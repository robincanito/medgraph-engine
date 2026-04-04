"""Ontología médica para MedGraph — ATC (fármacos) + SNOMED simplificado (patologías/anatomía).
Mapea entidades existentes del grafo a jerarquías taxonómicas usando Gemini LLM.
"""

import json
import os
import sys
import time
import re
from db import run_write, run_query

# Config
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GEMINI_API_KEY = os.getenv("GCP_API_KEY", "")
def init_gemini():
    """Init Gemini via Google GenAI (same pattern as extract_entities.py)."""
    from extract_entities import init_model
    return init_model()


# ============================================================
#  SEED: Cargar jerarquías estáticas a Neo4j
# ============================================================

def seed_atc_hierarchy(json_path=None):
    """Carga el árbol ATC como nodos CategoriaATC + relaciones ES_UN."""
    if json_path is None:
        json_path = os.path.join(DATA_DIR, "atc_hierarchy.json")

    with open(json_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    nodes = []
    relations = []

    def walk(subtree, parent_code=None, nivel=1):
        for code, data in subtree.items():
            nombre = data["nombre"]
            nodes.append({"codigo": code, "nombre": nombre, "nivel": nivel})
            if parent_code:
                relations.append({"hijo": code, "padre": parent_code})
            if "children" in data and data["children"]:
                walk(data["children"], code, nivel + 1)

    walk(tree)

    print(f"  ATC: {len(nodes)} nodos, {len(relations)} relaciones ES_UN")

    # Upload nodes in batch
    for i in range(0, len(nodes), 200):
        batch = nodes[i:i+200]
        run_write("""
        UNWIND $batch AS n
        MERGE (c:CategoriaATC {codigo: n.codigo})
        SET c.nombre = n.nombre, c.nivel = n.nivel
        """, {"batch": batch})

    # Upload ES_UN relations
    for i in range(0, len(relations), 200):
        batch = relations[i:i+200]
        run_write("""
        UNWIND $batch AS r
        MATCH (hijo:CategoriaATC {codigo: r.hijo})
        MATCH (padre:CategoriaATC {codigo: r.padre})
        MERGE (hijo)-[:ES_UN]->(padre)
        """, {"batch": batch})

    print(f"  ATC seeded: {len(nodes)} categorias")
    return len(nodes)


def seed_snomed_hierarchy(json_path=None):
    """Carga el árbol SNOMED simplificado como nodos CategoriaSNOMED + relaciones ES_UN."""
    if json_path is None:
        json_path = os.path.join(DATA_DIR, "snomed_systems.json")

    with open(json_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    nodes = []
    relations = []

    for sistema_code, sistema_data in tree.items():
        # Nivel 1: Sistema
        nodes.append({
            "codigo": sistema_code,
            "nombre": sistema_data["nombre"],
            "nivel": 1,
            "sistema": sistema_code
        })

        for subcat_code, subcat_data in sistema_data.get("subcategorias", {}).items():
            # Nivel 2: Subcategoría
            full_code = f"{sistema_code}_{subcat_code}"
            nodes.append({
                "codigo": full_code,
                "nombre": subcat_data["nombre"],
                "nivel": 2,
                "sistema": sistema_code
            })
            relations.append({"hijo": full_code, "padre": sistema_code})

            for grupo in subcat_data.get("grupos", []):
                # Nivel 3: Grupo específico
                grupo_code = f"{full_code}_{re.sub(r'[^a-z0-9]', '_', grupo.lower())[:30]}"
                nodes.append({
                    "codigo": grupo_code,
                    "nombre": grupo.lower(),
                    "nivel": 3,
                    "sistema": sistema_code
                })
                relations.append({"hijo": grupo_code, "padre": full_code})

    print(f"  SNOMED: {len(nodes)} nodos, {len(relations)} relaciones ES_UN")

    # Upload nodes
    for i in range(0, len(nodes), 200):
        batch = nodes[i:i+200]
        run_write("""
        UNWIND $batch AS n
        MERGE (c:CategoriaSNOMED {codigo: n.codigo})
        SET c.nombre = n.nombre, c.nivel = n.nivel, c.sistema = n.sistema
        """, {"batch": batch})

    # Upload ES_UN
    for i in range(0, len(relations), 200):
        batch = relations[i:i+200]
        run_write("""
        UNWIND $batch AS r
        MATCH (hijo:CategoriaSNOMED {codigo: r.hijo})
        MATCH (padre:CategoriaSNOMED {codigo: r.padre})
        MERGE (hijo)-[:ES_UN]->(padre)
        """, {"batch": batch})

    print(f"  SNOMED seeded: {len(nodes)} categorias")
    return len(nodes)


# ============================================================
#  MAPPING: Entidades existentes → Ontología con LLM
# ============================================================

def _llm_batch(model, prompt, max_retries=3):
    """Llama a Gemini (Google GenAI) y parsea JSON. Retry on failure."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 4096}
            )
            text = response.text.strip()
            # Limpiar markdown
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            return json.loads(text)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"    LLM error after {max_retries} retries: {str(e)[:80]}")
                return []


def map_farmacos_to_atc(batch_size=50, limit=0):
    """Mapea nodos Farmaco a códigos ATC usando Gemini."""
    model = init_gemini()

    # Get farmacos sin mapear
    query = """
    MATCH (f:Farmaco)
    WHERE NOT (f)-[:ES_UN]->(:CategoriaATC)
    RETURN f.nombre AS nombre
    ORDER BY f.nombre
    """
    if limit > 0:
        query += f" LIMIT {limit}"

    farmacos = run_query(query)
    print(f"  Farmacos sin mapear: {len(farmacos)}")

    # Get valid ATC codes
    atc_codes = run_query("MATCH (c:CategoriaATC) RETURN c.codigo AS codigo")
    valid_codes = {r["codigo"] for r in atc_codes}

    total_mapped = 0
    total_skipped = 0

    for i in range(0, len(farmacos), batch_size):
        batch = farmacos[i:i+batch_size]
        nombres = [f["nombre"] for f in batch]

        prompt = f"""Dado estos nombres de farmacos, asigna el codigo ATC mas especifico posible.
Si no existe codigo exacto de nivel 5, asigna el nivel mas especifico que conozcas (nivel 3 o 4).
Si no es un farmaco real o no tiene codigo ATC, responde "SKIP".
Responde SOLO JSON array, sin texto adicional.
Formato: [{{"nombre": "...", "atc": "C03CA01"}}]

Farmacos: {json.dumps(nombres, ensure_ascii=False)}"""

        results = _llm_batch(model, prompt)
        if not results:
            continue

        mappings = []
        for r in results:
            nombre = r.get("nombre", "").lower().strip()
            atc = r.get("atc", "SKIP").upper().strip()

            if atc == "SKIP" or not nombre:
                total_skipped += 1
                continue

            # Buscar el código ATC más cercano que exista en nuestra jerarquía
            matched_code = None
            for length in [7, 5, 4, 3, 1]:  # De más específico a más general
                candidate = atc[:length]
                if candidate in valid_codes:
                    matched_code = candidate
                    break

            if matched_code:
                mappings.append({"nombre": nombre, "atc_code": matched_code})
            else:
                total_skipped += 1

        # Upload batch
        if mappings:
            run_write("""
            UNWIND $batch AS m
            MATCH (f:Farmaco) WHERE toLower(f.nombre) = m.nombre
            MATCH (c:CategoriaATC {codigo: m.atc_code})
            MERGE (f)-[:ES_UN]->(c)
            """, {"batch": mappings})
            total_mapped += len(mappings)

        pct = int((i + batch_size) / len(farmacos) * 100)
        print(f"  [{min(pct,100)}%] {total_mapped} mapeados, {total_skipped} skipped")
        time.sleep(1)

    print(f"  ATC mapping done: {total_mapped} mapeados, {total_skipped} skipped")
    return {"mapped": total_mapped, "skipped": total_skipped}


def map_entities_to_snomed(label, batch_size=30, limit=0):
    """Mapea entidades de un label a categorías SNOMED."""
    model = init_gemini()

    # Get sistemas válidos
    sistemas = run_query("MATCH (c:CategoriaSNOMED {nivel: 1}) RETURN c.codigo AS codigo, c.nombre AS nombre")
    sistema_list = [s["codigo"] for s in sistemas]
    sistema_nombres = {s["codigo"]: s["nombre"] for s in sistemas}

    # Get entidades sin mapear
    query = f"""
    MATCH (e:{label})
    WHERE NOT (e)-[:ES_UN]->(:CategoriaSNOMED)
    RETURN e.nombre AS nombre
    ORDER BY e.nombre
    """
    if limit > 0:
        query += f" LIMIT {limit}"

    entities = run_query(query)
    print(f"  {label} sin mapear: {len(entities)}")

    # Get subcategorías válidas
    subcats = run_query("MATCH (c:CategoriaSNOMED {nivel: 2}) RETURN c.codigo AS codigo, c.nombre AS nombre, c.sistema AS sistema")
    subcat_map = {s["codigo"]: s for s in subcats}

    total_mapped = 0
    total_skipped = 0

    tipo_texto = {
        "Patologia": "patologias medicas",
        "EstructuraAnatomica": "estructuras anatomicas del cuerpo humano",
        "Procedimiento": "procedimientos medicos o diagnosticos"
    }

    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        nombres = [e["nombre"] for e in batch]

        prompt = f"""Clasifica estos {tipo_texto.get(label, 'conceptos medicos')} en un sistema del cuerpo humano.
Sistemas validos: {json.dumps(sistema_list, ensure_ascii=False)}
Responde SOLO JSON array, sin texto adicional.
Formato: [{{"nombre": "...", "sistema": "cardiovascular", "subcategoria": "nombre descriptivo"}}]
Si no es un concepto medico real o no se puede clasificar, responde "SKIP" como sistema.

Conceptos: {json.dumps(nombres, ensure_ascii=False)}"""

        results = _llm_batch(model, prompt)
        if not results:
            continue

        mappings = []
        for r in results:
            nombre = r.get("nombre", "").lower().strip()
            sistema = r.get("sistema", "SKIP").lower().strip()

            if sistema == "skip" or not nombre or sistema not in sistema_list:
                total_skipped += 1
                continue

            # Buscar la subcategoría más cercana
            subcat_code = None
            subcat_nombre = r.get("subcategoria", "").lower().strip()

            # Intentar matchear por nombre de subcategoría
            for code, data in subcat_map.items():
                if data["sistema"] == sistema and (
                    subcat_nombre in data["nombre"].lower() or
                    data["nombre"].lower() in subcat_nombre
                ):
                    subcat_code = code
                    break

            # Si no matchea subcategoría, vincular al sistema directamente
            target_code = subcat_code if subcat_code else sistema
            mappings.append({"nombre": nombre, "target": target_code})

        if mappings:
            run_write(f"""
            UNWIND $batch AS m
            MATCH (e:{label}) WHERE toLower(e.nombre) = m.nombre
            MATCH (c:CategoriaSNOMED {{codigo: m.target}})
            MERGE (e)-[:ES_UN]->(c)
            """, {"batch": mappings})
            total_mapped += len(mappings)

        pct = int((i + batch_size) / len(entities) * 100)
        print(f"  [{min(pct,100)}%] {total_mapped} mapeados, {total_skipped} skipped")
        time.sleep(1)

    print(f"  {label} SNOMED mapping done: {total_mapped} mapeados, {total_skipped} skipped")
    return {"mapped": total_mapped, "skipped": total_skipped}


# ============================================================
#  VALIDATE: Reportar cobertura
# ============================================================

def validate_mappings():
    """Reporta porcentaje de entidades mapeadas a ontología."""
    labels = ["Farmaco", "Patologia", "EstructuraAnatomica", "Procedimiento"]

    print("\n" + "=" * 60)
    print("  COBERTURA ONTOLÓGICA")
    print("=" * 60)

    for label in labels:
        total = run_query(f"MATCH (e:{label}) RETURN count(e) AS n")[0]["n"]
        if label == "Farmaco":
            mapped = run_query(f"MATCH (e:{label})-[:ES_UN]->(:CategoriaATC) RETURN count(DISTINCT e) AS n")[0]["n"]
        else:
            mapped = run_query(f"MATCH (e:{label})-[:ES_UN]->(:CategoriaSNOMED) RETURN count(DISTINCT e) AS n")[0]["n"]

        pct = (mapped * 100 // total) if total > 0 else 0
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"  {label:25s} {bar} {pct}% ({mapped}/{total})")

    # Stats ontología
    atc_nodes = run_query("MATCH (c:CategoriaATC) RETURN count(c) AS n")[0]["n"]
    snomed_nodes = run_query("MATCH (c:CategoriaSNOMED) RETURN count(c) AS n")[0]["n"]
    es_un = run_query("MATCH ()-[r:ES_UN]->() RETURN count(r) AS n")[0]["n"]

    print(f"\n  CategoriaATC: {atc_nodes} nodos")
    print(f"  CategoriaSNOMED: {snomed_nodes} nodos")
    print(f"  Relaciones ES_UN: {es_un}")


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python ontology.py <comando> [opciones]")
        print("Comandos:")
        print("  seed-atc         Cargar jerarquía ATC")
        print("  seed-snomed      Cargar jerarquía SNOMED")
        print("  seed-all         Cargar ambas jerarquías")
        print("  map-farmacos     Mapear Farmaco → ATC (--limit N)")
        print("  map-patologias   Mapear Patologia → SNOMED (--limit N)")
        print("  map-anatomia     Mapear EstructuraAnatomica → SNOMED (--limit N)")
        print("  map-procedimientos  Mapear Procedimiento → SNOMED (--limit N)")
        print("  map-all          Mapear todo (--limit N)")
        print("  validate         Reportar cobertura")
        sys.exit(1)

    cmd = sys.argv[1]
    limit = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    if cmd == "seed-atc":
        seed_atc_hierarchy()
    elif cmd == "seed-snomed":
        seed_snomed_hierarchy()
    elif cmd == "seed-all":
        seed_atc_hierarchy()
        seed_snomed_hierarchy()
    elif cmd == "map-farmacos":
        map_farmacos_to_atc(limit=limit)
    elif cmd == "map-patologias":
        map_entities_to_snomed("Patologia", limit=limit)
    elif cmd == "map-anatomia":
        map_entities_to_snomed("EstructuraAnatomica", limit=limit)
    elif cmd == "map-procedimientos":
        map_entities_to_snomed("Procedimiento", limit=limit)
    elif cmd == "map-all":
        print("=== Mapeando Farmacos → ATC ===")
        map_farmacos_to_atc(limit=limit)
        print("\n=== Mapeando Patologias → SNOMED ===")
        map_entities_to_snomed("Patologia", limit=limit)
        print("\n=== Mapeando EstructuraAnatomica → SNOMED ===")
        map_entities_to_snomed("EstructuraAnatomica", limit=limit)
        print("\n=== Mapeando Procedimientos → SNOMED ===")
        map_entities_to_snomed("Procedimiento", limit=limit)
    elif cmd == "validate":
        validate_mappings()
    else:
        print(f"Comando desconocido: {cmd}")
        sys.exit(1)
