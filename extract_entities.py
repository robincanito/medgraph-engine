"""Extractor masivo de entidades medicas con LLM (Gemini).

Procesa chunks de texto medico y extrae entidades + relaciones
para poblar el grafo semantico de Neo4j.

Modulo importable. Funciones principales:
  - extract_from_chunk(model, chunk) -> dict
  - extract_libro(libro_id, model_id, dev_mode) -> stats
  - upload_entities(entities, dev_mode) -> stats

CLI: python extract_entities.py <libro_id> [--preview] [--dev] [--limit N]
"""

import json
import os
import re
import time
import unicodedata
from datetime import datetime
from dotenv import load_dotenv
from db import run_write, run_query

load_dotenv()

PARSED_DIR = os.path.join(os.path.dirname(__file__), "parsed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "extracted")
DEFAULT_MODEL = "gemini-2.0-flash"
GCP_API_KEY = os.getenv("GCP_API_KEY", "")

# Rate limiting
BATCH_SIZE = 5          # chunks por batch (para no saturar)
DELAY_BETWEEN = 1.0     # segundos entre batches
MAX_RETRIES = 2

# Tipos de entidad validos
ENTITY_TYPES = {
    "patologia", "estructura_anatomica", "procedimiento", "farmaco",
    "grupo_farmacologico", "agente", "signo", "sintoma",
    "metodo_dx", "hallazgo", "parametro",
}

# Relaciones validas
RELATION_TYPES = {
    "CAUSADA_POR", "SE_MANIFIESTA_CON", "SE_DIAGNOSTICA_CON",
    "SE_TRATA_CON", "PARTE_DE", "EVALUA", "PERTENECE_A",
    "PUEDE_PRODUCIR", "DIFERENCIAL_DE", "ASOCIADA_A",
    "IRRIGA", "INERVA", "DRENA_EN", "SE_ORIGINA_EN",
    "FACTOR_DE_RIESGO", "COMPLICACION_DE", "VARIANTE_DE",
}

# Mapa tipo -> label Neo4j
TYPE_TO_LABEL = {
    "patologia": "Patologia",
    "estructura_anatomica": "EstructuraAnatomica",
    "procedimiento": "Procedimiento",
    "farmaco": "Farmaco",
    "grupo_farmacologico": "GrupoFarmacologico",
    "agente": "Agente",
    "signo": "Signo",
    "sintoma": "Sintoma",
    "metodo_dx": "MetodoDx",
    "hallazgo": "Hallazgo",
    "parametro": "Parametro",
}

EXTRACTION_PROMPT = """Eres un extractor de entidades medicas. Analiza el siguiente texto de un libro medico y extrae TODAS las entidades y relaciones medicas que encuentres.

REGLAS:
- Extrae SOLO entidades medicas concretas (no conceptos vagos como "tratamiento" sin especificar cual)
- Usa nombres canonicos en espanol (ej: "otitis media aguda", no "OMA" ni "acute otitis media")
- Incluye sinonimos y abreviaturas comunes
- Las relaciones deben conectar entidades que aparecen en el texto
- Si no hay entidades medicas relevantes, devuelve listas vacias
- NO inventes relaciones que no esten implicitas o explicitas en el texto

TIPOS DE ENTIDAD:
- patologia: enfermedades, sindromes, trastornos
- estructura_anatomica: organos, tejidos, estructuras
- procedimiento: tecnicas diagnosticas o terapeuticas
- farmaco: medicamentos especificos
- grupo_farmacologico: familias de farmacos
- agente: microorganismos, virus, parasitos
- signo: hallazgos objetivos del examen fisico
- sintoma: manifestaciones subjetivas del paciente
- metodo_dx: estudios complementarios, laboratorio
- hallazgo: resultados de estudios o examenes
- parametro: valores medibles (presion arterial, frecuencia, etc.)

TIPOS DE RELACION:
- CAUSADA_POR: patologia <- agente/causa
- SE_MANIFIESTA_CON: patologia -> signo/sintoma
- SE_DIAGNOSTICA_CON: patologia -> metodo_dx/procedimiento
- SE_TRATA_CON: patologia -> farmaco/procedimiento
- PARTE_DE: estructura -> estructura mayor
- EVALUA: procedimiento -> estructura/parametro
- PERTENECE_A: farmaco -> grupo_farmacologico
- PUEDE_PRODUCIR: farmaco -> signo/efecto adverso
- DIFERENCIAL_DE: patologia <-> patologia
- ASOCIADA_A: entidad <-> entidad (relacion general)
- FACTOR_DE_RIESGO: entidad -> patologia
- COMPLICACION_DE: patologia -> patologia
- VARIANTE_DE: patologia -> patologia

TEXTO:
\"\"\"
{texto}
\"\"\"

CONTEXTO: Libro: {libro}, Capitulo: {capitulo}, Seccion: {seccion}

Responde SOLO con JSON valido, sin markdown ni explicaciones:
{{"entidades": [{{"nombre": "...", "tipo": "...", "sinonimos": ["..."]}}], "relaciones": [{{"desde": "...", "relacion": "...", "hasta": "..."}}]}}"""


def normalize_name(name: str) -> str:
    """Normaliza nombre de entidad: lowercase, sin acentos extra, trim."""
    if not name:
        return ""
    name = name.strip().lower()
    # Quitar puntos finales, parentesis sueltos
    name = re.sub(r'[\.;,]+$', '', name).strip()
    # Colapsar espacios
    name = re.sub(r'\s+', ' ', name)
    return name


def init_model(model_id: str = DEFAULT_MODEL):
    """Inicializa cliente de Google GenAI con API key."""
    from google import genai

    client = genai.Client(api_key=GCP_API_KEY)
    print(f"  Modelo inicializado: {model_id}")
    return {"client": client, "model_id": model_id}


def extract_from_chunk(model, chunk: dict, libro_titulo: str = "") -> dict:
    """Extrae entidades y relaciones de un chunk usando el LLM.

    Args:
        model: dict con client (genai.Client) y model_id
        chunk: dict con text, titulo_capitulo, titulo_seccion, libro_id
        libro_titulo: Titulo del libro para contexto

    Returns:
        dict con entidades y relaciones validadas
    """
    prompt = EXTRACTION_PROMPT.format(
        texto=chunk["text"][:3000],  # Limitar largo
        libro=libro_titulo or chunk.get("libro_id", ""),
        capitulo=chunk.get("titulo_capitulo", ""),
        seccion=chunk.get("titulo_seccion", ""),
    )

    client = model["client"]
    model_id = model["model_id"]

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json",
                },
            )
            text = response.text.strip()

            # Parsear JSON
            data = json.loads(text)

            # Validar y limpiar
            return _validate_extraction(data, chunk)

        except json.JSONDecodeError:
            # Intentar extraer JSON de respuesta con markdown
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return _validate_extraction(data, chunk)
                except json.JSONDecodeError:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return {"entidades": [], "relaciones": [], "error": "json_parse_error"}

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(3)
                continue
            return {"entidades": [], "relaciones": [], "error": str(e)}


def _validate_extraction(data: dict, chunk: dict) -> dict:
    """Valida y limpia entidades/relaciones extraidas."""
    valid_entities = []
    entity_names = set()

    for ent in data.get("entidades", []):
        nombre = normalize_name(ent.get("nombre", ""))
        tipo = ent.get("tipo", "").lower().strip()

        # Validar
        if not nombre or len(nombre) < 2 or len(nombre) > 100:
            continue
        if tipo not in ENTITY_TYPES:
            continue

        # Normalizar sinonimos
        sinonimos = []
        for s in ent.get("sinonimos", []):
            sn = normalize_name(s)
            if sn and sn != nombre and len(sn) >= 2:
                sinonimos.append(sn)

        valid_entities.append({
            "nombre": nombre,
            "tipo": tipo,
            "sinonimos": sinonimos,
        })
        entity_names.add(nombre)

    valid_relations = []
    for rel in data.get("relaciones", []):
        desde = normalize_name(rel.get("desde", ""))
        hasta = normalize_name(rel.get("hasta", ""))
        relacion = rel.get("relacion", "").upper().strip()

        if not desde or not hasta or not relacion:
            continue
        if relacion not in RELATION_TYPES:
            continue
        # Al menos uno de los extremos debe ser una entidad extraida
        if desde not in entity_names and hasta not in entity_names:
            continue

        valid_relations.append({
            "desde": desde,
            "relacion": relacion,
            "hasta": hasta,
        })

    return {
        "entidades": valid_entities,
        "relaciones": valid_relations,
        "chunk_id": chunk.get("id", ""),
        "libro_id": chunk.get("libro_id", ""),
    }


def extract_libro(libro_id: str, model_id: str = DEFAULT_MODEL,
                  limit: int = 0, skip_first: int = 0,
                  on_progress: callable = None) -> dict:
    """Extrae entidades de todos los chunks de un libro.

    Args:
        libro_id: ID del libro
        model_id: Model ID de Gemini
        limit: Si > 0, solo procesar N chunks (para testing)
        skip_first: Saltear los primeros N chunks (prologo, indice)
        on_progress: Callback(pct, msg)

    Returns:
        dict con estadisticas y path al archivo de salida
    """
    # Cargar chunks v2
    chunks_path = os.path.join(PARSED_DIR, f"{libro_id}_v2_chunks.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"No hay chunks v2 para {libro_id}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Filtrar chunks de contenido (skip prologo/indice/colaboradores)
    chunks = [c for c in all_chunks if c.get("word_count", 0) >= 100]
    if skip_first:
        chunks = chunks[skip_first:]
    if limit:
        chunks = chunks[:limit]

    print(f"  Procesando {len(chunks)} chunks de {libro_id}")

    # Inicializar modelo
    model = init_model(model_id)

    # Cargar titulo del libro
    from parser_v2 import load_catalog
    catalog = load_catalog()
    libro_entry = next((l for l in catalog["libros"] if l["id"] == libro_id), None)
    libro_titulo = libro_entry["titulo"] if libro_entry else libro_id

    # Procesar chunks
    all_extractions = []
    total_entities = 0
    total_relations = 0
    errors = 0
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        extraction = extract_from_chunk(model, chunk, libro_titulo)
        all_extractions.append(extraction)

        n_ent = len(extraction["entidades"])
        n_rel = len(extraction["relaciones"])
        total_entities += n_ent
        total_relations += n_rel

        if extraction.get("error"):
            errors += 1

        # Progress
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            pct = int((i + 1) / len(chunks) * 100)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
            msg = f"  [{pct}%] {i+1}/{len(chunks)} chunks | {total_entities} entidades | {total_relations} relaciones | ETA: {eta:.0f}s"
            print(msg)
            if on_progress:
                on_progress(pct, msg)

        # Rate limiting
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(DELAY_BETWEEN)

    # Guardar resultado
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{libro_id}_entities.json")

    result = {
        "libro_id": libro_id,
        "model_id": model_id,
        "fecha": datetime.now().isoformat(),
        "chunks_procesados": len(chunks),
        "total_entidades": total_entities,
        "total_relaciones": total_relations,
        "errors": errors,
        "extractions": all_extractions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  Extraccion completa: {total_entities} entidades, {total_relations} relaciones")
    print(f"  Errores: {errors}")
    print(f"  Guardado en: {output_path}")

    return {
        "total_entities": total_entities,
        "total_relations": total_relations,
        "errors": errors,
        "output_path": output_path,
        "duration": time.time() - start_time,
    }


def canonicalize_entities(extractions: list) -> tuple:
    """Consolida entidades de multiples chunks en un set canonico.

    Resuelve sinonimos, agrupa duplicados, cuenta frecuencias.

    Returns:
        (canonical_entities, canonical_relations)
    """
    # Mapa nombre -> entidad canonica
    entity_map = {}      # nombre normalizado -> {nombre, tipo, sinonimos, freq, chunk_ids}
    synonym_map = {}     # sinonimo -> nombre canonico

    for ext in extractions:
        chunk_id = ext.get("chunk_id", "")
        for ent in ext.get("entidades", []):
            nombre = ent["nombre"]
            tipo = ent["tipo"]

            # Verificar si ya existe como sinonimo
            canon_name = synonym_map.get(nombre, nombre)

            if canon_name in entity_map:
                # Ya existe: incrementar frecuencia, agregar chunk
                existing = entity_map[canon_name]
                existing["freq"] += 1
                existing["chunk_ids"].add(chunk_id)
                # Agregar sinonimos nuevos
                for s in ent.get("sinonimos", []):
                    if s not in existing["sinonimos"] and s != canon_name:
                        existing["sinonimos"].append(s)
                        synonym_map[s] = canon_name
            else:
                # Nueva entidad
                entity_map[nombre] = {
                    "nombre": nombre,
                    "tipo": tipo,
                    "sinonimos": list(ent.get("sinonimos", [])),
                    "freq": 1,
                    "chunk_ids": {chunk_id},
                }
                # Registrar sinonimos
                for s in ent.get("sinonimos", []):
                    synonym_map[s] = nombre

    # Consolidar relaciones
    relation_set = set()  # (desde, relacion, hasta)
    canonical_relations = []

    for ext in extractions:
        for rel in ext.get("relaciones", []):
            desde = synonym_map.get(rel["desde"], rel["desde"])
            hasta = synonym_map.get(rel["hasta"], rel["hasta"])
            key = (desde, rel["relacion"], hasta)

            if key not in relation_set:
                relation_set.add(key)
                canonical_relations.append({
                    "desde": desde,
                    "relacion": rel["relacion"],
                    "hasta": hasta,
                })

    # Convertir chunk_ids set a list para JSON
    canonical_entities = []
    for ent in entity_map.values():
        ent["chunk_ids"] = list(ent["chunk_ids"])
        canonical_entities.append(ent)

    # Ordenar por frecuencia
    canonical_entities.sort(key=lambda x: x["freq"], reverse=True)

    return canonical_entities, canonical_relations


def upload_entities(entities: list, relations: list, libro_id: str,
                    dev_mode: bool = True) -> dict:
    """Sube entidades y relaciones a Neo4j en batch (UNWIND).

    Args:
        entities: Lista de entidades canonicas
        relations: Lista de relaciones canonicas
        libro_id: ID del libro fuente
        dev_mode: Si True, usa labels con sufijo Dev (PatologiaDev, etc.)

    Returns:
        dict con estadisticas
    """
    suffix = "Dev" if dev_mode else ""
    created_entities = 0
    created_relations = 0
    BATCH = 200

    print(f"  Subiendo {len(entities)} entidades{' (DEV mode)' if dev_mode else ''}...")

    # Agrupar entidades por tipo (cada label necesita su propia query UNWIND)
    by_type = {}
    for ent in entities:
        tipo = ent["tipo"]
        if tipo not in by_type:
            by_type[tipo] = []
        by_type[tipo].append(ent)

    # Subir entidades en batch por tipo
    for tipo, ents in by_type.items():
        label = TYPE_TO_LABEL.get(tipo, "Entidad") + suffix
        for i in range(0, len(ents), BATCH):
            batch = ents[i:i + BATCH]
            batch_data = [{
                "nombre": e["nombre"],
                "tipo": e["tipo"],
                "sinonimos": e.get("sinonimos", []),
                "freq": e.get("freq", 1),
                "libro_id": libro_id,
            } for e in batch]
            try:
                run_write(f"""
                UNWIND $batch AS ent
                MERGE (e:{label} {{nombre: ent.nombre}})
                SET e.tipo = ent.tipo,
                    e.sinonimos = ent.sinonimos,
                    e.freq = ent.freq,
                    e.fuente_libro = ent.libro_id
                """, {"batch": batch_data})
                created_entities += len(batch)
            except Exception as e:
                print(f"    Error batch {label}: {str(e)[:80]}")

        print(f"    {label}: {len(ents)} entidades")

    # Subir relaciones MENCIONA en batch (chunk -> entidad)
    print(f"  Conectando chunks con entidades (MENCIONA)...")
    menciona_batch = []
    for ent in entities:
        label = TYPE_TO_LABEL.get(ent["tipo"], "Entidad") + suffix
        for chunk_id in ent.get("chunk_ids", [])[:10]:
            if chunk_id:
                menciona_batch.append({
                    "nombre": ent["nombre"],
                    "label": label,
                    "chunk_id": chunk_id,
                })

    # MENCIONA necesita match por label, agrupar por label
    menciona_by_label = {}
    for m in menciona_batch:
        if m["label"] not in menciona_by_label:
            menciona_by_label[m["label"]] = []
        menciona_by_label[m["label"]].append(m)

    total_menciona = 0
    for label, items in menciona_by_label.items():
        for i in range(0, len(items), BATCH):
            batch = [{"nombre": m["nombre"], "chunk_id": m["chunk_id"]} for m in items[i:i + BATCH]]
            try:
                run_write(f"""
                UNWIND $batch AS m
                MATCH (e:{label} {{nombre: m.nombre}})
                MATCH (c:Chunk {{id: m.chunk_id}})
                MERGE (c)-[:MENCIONA]->(e)
                """, {"batch": batch})
                total_menciona += len(batch)
            except Exception as e:
                print(f"    Error MENCIONA {label}: {str(e)[:80]}")

    print(f"    {total_menciona} relaciones MENCIONA")

    # Subir relaciones entre entidades en batch, agrupadas por tipo de relacion
    print(f"  Subiendo {len(relations)} relaciones entre entidades...")

    # Pre-build lookup de nombre -> label
    entity_label = {}
    for ent in entities:
        entity_label[ent["nombre"]] = TYPE_TO_LABEL.get(ent["tipo"], "Entidad") + suffix

    # Agrupar por (desde_label, relacion, hasta_label)
    rel_groups = {}
    for rel in relations:
        desde_l = entity_label.get(rel["desde"])
        hasta_l = entity_label.get(rel["hasta"])
        if desde_l and hasta_l:
            key = (desde_l, rel["relacion"], hasta_l)
            if key not in rel_groups:
                rel_groups[key] = []
            rel_groups[key].append({"desde": rel["desde"], "hasta": rel["hasta"]})

    for (desde_l, rel_type, hasta_l), items in rel_groups.items():
        for i in range(0, len(items), BATCH):
            batch = items[i:i + BATCH]
            try:
                run_write(f"""
                UNWIND $batch AS r
                MATCH (a:{desde_l} {{nombre: r.desde}})
                MATCH (b:{hasta_l} {{nombre: r.hasta}})
                MERGE (a)-[:{rel_type}]->(b)
                """, {"batch": batch})
                created_relations += len(batch)
            except Exception as e:
                print(f"    Error {desde_l}-[{rel_type}]->{hasta_l}: {str(e)[:80]}")

    print(f"  Resultado: {created_entities} entidades, {total_menciona} MENCIONA, {created_relations} relaciones")
    return {"entities": created_entities, "menciona": total_menciona, "relations": created_relations}


def promote_dev_entities():
    """Renombra labels Dev a produccion (ej: PatologiaDev -> Patologia)."""
    for tipo, label in TYPE_TO_LABEL.items():
        dev_label = label + "Dev"
        count = run_query(f"MATCH (n:{dev_label}) RETURN count(n) AS n")
        n = count[0]["n"] if count else 0
        if n > 0:
            print(f"  Promoviendo {n} nodos {dev_label} -> {label}...")
            run_write(f"MATCH (n:{dev_label}) SET n:{label} REMOVE n:{dev_label}")
    print("  Promocion completa.")


# --- CLI ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\nextract_entities -- Extraccion masiva de entidades medicas con LLM")
        print("\nUso:")
        print("  python extract_entities.py <libro_id>                    - Extraer todo")
        print("  python extract_entities.py <libro_id> --preview          - Solo 10 chunks, sin subir")
        print("  python extract_entities.py <libro_id> --limit 50         - Solo N chunks")
        print("  python extract_entities.py <libro_id> --skip 30          - Saltear primeros N")
        print("  python extract_entities.py <libro_id> --dev              - Subir con labels Dev")
        print("  python extract_entities.py <libro_id> --upload           - Subir extracciones guardadas")
        print("  python extract_entities.py promote                       - Mover Dev -> produccion")
        print("  python extract_entities.py stats                         - Ver stats del grafo")
        sys.exit(0)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "promote":
        promote_dev_entities()
        sys.exit(0)

    if cmd == "stats":
        for tipo, label in TYPE_TO_LABEL.items():
            for suffix in ["", "Dev"]:
                full_label = label + suffix
                count = run_query(f"MATCH (n:{full_label}) RETURN count(n) AS n")
                n = count[0]["n"] if count else 0
                if n > 0:
                    print(f"  {full_label}: {n}")
        # Relaciones
        rels = run_query("MATCH ()-[r:MENCIONA]->() RETURN count(r) AS n")
        print(f"  MENCIONA: {rels[0]['n'] if rels else 0}")
        sys.exit(0)

    libro_id = cmd
    preview = "--preview" in args
    dev_mode = "--dev" in args or preview
    upload_only = "--upload" in args

    limit = 0
    if "--limit" in args:
        idx = args.index("--limit")
        limit = int(args[idx + 1]) if idx + 1 < len(args) else 10

    skip = 0
    if "--skip" in args:
        idx = args.index("--skip")
        skip = int(args[idx + 1]) if idx + 1 < len(args) else 0

    if preview:
        limit = limit or 10

    if upload_only:
        # Solo subir extracciones ya guardadas
        output_path = os.path.join(OUTPUT_DIR, f"{libro_id}_entities.json")
        if not os.path.exists(output_path):
            print(f"No hay extracciones guardadas para {libro_id}")
            sys.exit(1)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entities, relations = canonicalize_entities(data["extractions"])
        print(f"  Canonicalizadas: {len(entities)} entidades, {len(relations)} relaciones")
        upload_entities(entities, relations, libro_id, dev_mode=dev_mode)
        sys.exit(0)

    # Extraer
    print(f"\n{'='*60}")
    print(f"  EXTRACCION: {libro_id}")
    print(f"  Modo: {'PREVIEW' if preview else 'DEV' if dev_mode else 'PRODUCCION'}")
    print(f"  Chunks: {'todos' if not limit else limit}")
    print(f"{'='*60}\n")

    stats = extract_libro(libro_id, limit=limit, skip_first=skip)

    # Canonicalizar
    output_path = stats["output_path"]
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities, relations = canonicalize_entities(data["extractions"])
    print(f"\n  Canonicalizadas: {len(entities)} entidades unicas, {len(relations)} relaciones unicas")

    # Top 20 entidades por frecuencia
    print(f"\n  Top 20 entidades:")
    for ent in entities[:20]:
        sins = f" ({', '.join(ent['sinonimos'][:3])})" if ent.get('sinonimos') else ""
        print(f"    [{ent['tipo']}] {ent['nombre']}{sins} (freq: {ent['freq']})")

    if not preview:
        print(f"\n  Subiendo a Neo4j...")
        upload_entities(entities, relations, libro_id, dev_mode=dev_mode)
