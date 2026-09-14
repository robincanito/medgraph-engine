"""Endpoint comprehensive: todo sobre un tema en un solo call.

PODADO EN EL ESPEJO OSS (14-sep-2026). La version de la instancia privada devolvia tambien el
detalle del `:Tema` y las ACTIVIDADES de catedra (`:Actividad`, `:Documento`) con su material.
Esos nodos los escriben cargadores privados —una cursada, un cronograma—: ningun script de este
repo los crea, asi que aca esas dos secciones devolvian listas vacias en cualquier grafo. Quedan
las tres que este repo SI puede llenar: entidades del grafo (las que extrae `extract_entities.py`
con el perfil activo), bibliografia (los chunks del pipeline) y ontologia (`ontology.py`).
"""

from fastapi import APIRouter, Query

from services import graph, vector

router = APIRouter(tags=["comprehensive"])


@router.get("/topic/{tema}/comprehensive")
async def topic_comprehensive(
    tema: str,
    top_k: int = Query(default=20, description="Chunks de bibliografia a devolver"),
):
    """Devuelve TODO sobre un tema: grafo + bibliografia + ontologia.

    Orquesta multiples consultas internas y devuelve un paquete completo
    para que el LLM no necesite hacer multiples calls.
    """

    result = {
        "tema": tema,
        "grafo": {},
        "bibliografia": [],
    }

    # 1. Buscar en el grafo semantico (nodos + relaciones)
    # Buscar como Patologia
    patologias = graph.get_pathology(tema)
    if patologias:
        result["grafo"]["patologias"] = patologias

    # Buscar como Procedimiento
    procedimientos = graph.get_procedure(tema)
    if procedimientos:
        result["grafo"]["procedimientos"] = procedimientos

    # Buscar entidades Dev (extraidas por LLM)
    entidades_dev = _search_dev_entities(tema)
    if entidades_dev:
        result["grafo"]["entidades_extraidas"] = entidades_dev

    # 2. Buscar chunks de bibliografia (hybrid search)
    try:
        search_result = vector.search_hybrid(tema, top_k=top_k)
        if search_result.get("results"):
            result["bibliografia"] = [
                {
                    "libro": r.get("libro"),
                    "paginas": f"{r.get('pag_inicio', '?')}-{r.get('pag_fin', '?')}",
                    "capitulo": r.get("capitulo", ""),
                    "seccion": r.get("seccion", ""),
                    "tipo": r.get("tipo", ""),
                    "texto": r.get("texto", ""),
                    "score": r.get("rrf_score", 0),
                }
                for r in search_result["results"]
            ]
            result["busqueda"] = {
                "intencion": search_result.get("intencion", ""),
                "query_expandida": search_result.get("query_expandida", ""),
            }
    except Exception:
        pass

    # 3. Ontología (ATC + SNOMED)
    ontologia = {"farmacos_atc": [], "patologias_snomed": [], "query_cruzada": []}
    try:
        # Fármacos → ATC
        farmacos_atc = graph.query("""
        MATCH (f:Farmaco)-[:ES_UN*1..5]->(cat:CategoriaATC)
        WHERE toLower(f.nombre) CONTAINS toLower($tema)
        WITH f.nombre AS farmaco, collect(DISTINCT {codigo: cat.codigo, nombre: cat.nombre, nivel: cat.nivel}) AS jerarquia
        RETURN farmaco, jerarquia
        ORDER BY farmaco LIMIT 10
        """, {"tema": tema})
        for r in farmacos_atc:
            ontologia["farmacos_atc"].append({
                "nombre": r["farmaco"],
                "jerarquia": sorted(r["jerarquia"], key=lambda x: x.get("nivel", 0))
            })

        # Patologías → SNOMED
        pato_snomed = graph.query("""
        MATCH (p:Patologia)-[:ES_UN*1..3]->(cat:CategoriaSNOMED)
        WHERE toLower(p.nombre) CONTAINS toLower($tema)
        WITH p.nombre AS patologia, collect(DISTINCT {nombre: cat.nombre, sistema: cat.sistema, nivel: cat.nivel}) AS jerarquia
        RETURN patologia, jerarquia
        ORDER BY patologia LIMIT 10
        """, {"tema": tema})
        for r in pato_snomed:
            ontologia["patologias_snomed"].append({
                "nombre": r["patologia"],
                "jerarquia": sorted(r["jerarquia"], key=lambda x: x.get("nivel", 0))
            })

        # Query cruzada
        if ontologia["patologias_snomed"]:
            sistemas = set()
            for p in ontologia["patologias_snomed"]:
                for j in p["jerarquia"]:
                    if j.get("sistema"):
                        sistemas.add(j["sistema"])
            for sistema in list(sistemas)[:2]:
                cruzada = graph.query("""
                MATCH (p:Patologia)-[:ES_UN*1..3]->(sno:CategoriaSNOMED {sistema: $sistema})
                WHERE toLower(p.nombre) CONTAINS toLower($tema)
                MATCH (p)-[:SE_TRATA_CON]->(f:Farmaco)
                OPTIONAL MATCH (f)-[:ES_UN*1..5]->(atc:CategoriaATC)
                RETURN DISTINCT f.nombre AS farmaco, p.nombre AS patologia,
                       collect(DISTINCT atc.nombre)[0] AS clase_atc, $sistema AS sistema
                LIMIT 10
                """, {"tema": tema, "sistema": sistema})
                for r in cruzada:
                    ontologia["query_cruzada"].append({
                        "farmaco": r["farmaco"],
                        "patologia": r["patologia"],
                        "clase_atc": r.get("clase_atc"),
                        "sistema_snomed": r["sistema"]
                    })
    except Exception:
        pass

    result["ontologia"] = ontologia

    # 4. Resumen de fuentes encontradas
    fuentes = set()
    for bib in result["bibliografia"]:
        if bib.get("libro"):
            fuentes.add(bib["libro"])
    result["fuentes"] = list(fuentes)

    # 5. Flag de completitud
    result["tiene_grafo"] = bool(result["grafo"])
    result["tiene_bibliografia"] = bool(result["bibliografia"])
    result["tiene_ontologia"] = bool(ontologia["farmacos_atc"] or ontologia["patologias_snomed"])

    return result


@router.get("/topic/{tema}/pathways")
async def topic_pathways(tema: str):
    """Cadena causal/fisiopatologica: agente -> mecanismo -> efecto -> signo -> complicacion."""
    pathways = graph.query("""
    MATCH (a)-[r:PATHWAY]->(b)
    WHERE toLower(a.nombre) CONTAINS toLower($tema)
       OR toLower(b.nombre) CONTAINS toLower($tema)
       OR r.nombre_dag CONTAINS toLower($tema)
    RETURN DISTINCT r.nombre_dag AS dag, a.nombre AS desde, labels(a)[0] AS tipo_desde,
           b.nombre AS hasta, labels(b)[0] AS tipo_hasta,
           r.orden AS orden, r.tipo AS tipo_paso, r.nota AS nota
    ORDER BY r.nombre_dag, r.orden
    """, {"tema": tema})

    dags = {}
    for p in pathways:
        dag_name = p["dag"]
        if dag_name not in dags:
            dags[dag_name] = {"nombre": dag_name, "pasos": []}
        dags[dag_name]["pasos"].append({
            "orden": p["orden"],
            "desde": p["desde"],
            "tipo_desde": p["tipo_desde"],
            "hasta": p["hasta"],
            "tipo_hasta": p["tipo_hasta"],
            "tipo_paso": p.get("tipo_paso"),
            "nota": p.get("nota"),
        })

    return {"tema": tema, "pathways": list(dags.values()), "count": len(dags)}


@router.get("/topic/{tema}/clinical")
async def topic_clinical(tema: str):
    """Arbol de decision clinica con bifurcaciones y condiciones."""
    steps = graph.query("""
    MATCH (a)-[r:CLINICAL]->(b)
    WHERE toLower(a.nombre) CONTAINS toLower($tema)
       OR toLower(b.nombre) CONTAINS toLower($tema)
       OR r.nombre_dag CONTAINS toLower($tema)
    RETURN DISTINCT r.nombre_dag AS dag, a.nombre AS desde, labels(a)[0] AS tipo_desde,
           b.nombre AS hasta, labels(b)[0] AS tipo_hasta,
           r.orden AS orden, r.tipo_paso AS tipo_paso,
           r.condicion AS condicion, r.nota AS nota
    ORDER BY r.nombre_dag, r.orden, r.condicion
    """, {"tema": tema})

    dags = {}
    for s in steps:
        dag_name = s["dag"]
        if dag_name not in dags:
            dags[dag_name] = {"nombre": dag_name, "pasos": []}
        dags[dag_name]["pasos"].append({
            "orden": s["orden"],
            "desde": s["desde"],
            "tipo_desde": s["tipo_desde"],
            "hasta": s["hasta"],
            "tipo_hasta": s["tipo_hasta"],
            "tipo_paso": s.get("tipo_paso"),
            "condicion": s.get("condicion"),
            "nota": s.get("nota"),
        })

    return {"tema": tema, "clinical": list(dags.values()), "count": len(dags)}


def _search_dev_entities(tema: str) -> list:
    """Busca entidades extraidas (Dev) relacionadas con el tema."""
    # Buscar en todos los labels Dev
    dev_labels = [
        "Patologia", "EstructuraAnatomica", "Procedimiento",
        "Farmaco", "Agente", "Signo", "Sintoma",
        "MetodoDx", "Hallazgo", "GrupoFarmacologico",
        "PatologiaDev", "EstructuraAnatomicaDev", "ProcedimientoDev",
        "FarmacoDev", "AgenteDev", "SignoDev", "SintomaDev",
        "MetodoDxDev", "HallazgoDev", "GrupoFarmacologicoDev",
    ]

    results = []
    for label in dev_labels:
        try:
            matches = graph.query(f"""
            MATCH (e:{label})
            WHERE toLower(e.nombre) CONTAINS toLower($tema)
            OPTIONAL MATCH (e)-[r]-(related)
            WHERE NOT related:Chunk
            RETURN e.nombre AS nombre, labels(e)[0] AS tipo,
                   e.sinonimos AS sinonimos, e.freq AS freq,
                   collect(DISTINCT {{
                       nombre: related.nombre,
                       tipo: labels(related)[0],
                       relacion: type(r)
                   }})[0..10] AS relaciones
            LIMIT 5
            """, {"tema": tema})
            results.extend(matches)
        except Exception:
            pass

    return results
