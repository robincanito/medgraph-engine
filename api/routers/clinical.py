"""Endpoints de razonamiento: pathways (fisiopatologia) y clinical (arbol de decision)."""

from fastapi import APIRouter
from services.graph import query

router = APIRouter(tags=["clinical"])


@router.get("/topic/{tema}/pathways")
async def get_pathways(tema: str):
    """Cadenas causales/fisiopatologicas de un tema. Deterministas, lineales."""

    # Buscar todos los DAGs PATHWAY que involucren este tema
    paths = query("""
    MATCH (start)-[r:PATHWAY*1..12]->(end)
    WHERE ANY(n IN nodes((start)-[r*1..12]->(end))
          WHERE toLower(n.nombre) CONTAINS toLower($tema))
    WITH start, r, end,
         [rel IN r | {
           nombre_dag: rel.nombre_dag,
           orden: rel.orden,
           nota: rel.nota
         }] AS rels,
         [n IN nodes((start)-[r*1..12]->(end)) | {
           nombre: n.nombre,
           tipo: labels(n)[0]
         }] AS nodos
    RETURN DISTINCT rels[0].nombre_dag AS nombre_dag,
           nodos, rels
    """, {"tema": tema})

    if not paths:
        # Fallback: buscar nodos directamente conectados con PATHWAY
        paths = query("""
        MATCH (a)-[r:PATHWAY]->(b)
        WHERE toLower(a.nombre) CONTAINS toLower($tema)
           OR toLower(b.nombre) CONTAINS toLower($tema)
        RETURN r.nombre_dag AS nombre_dag,
               a.nombre AS desde, labels(a)[0] AS tipo_desde,
               b.nombre AS hasta, labels(b)[0] AS tipo_hasta,
               r.orden AS orden, r.nota AS nota
        ORDER BY r.nombre_dag, r.orden
        """, {"tema": tema})

        # Agrupar por nombre_dag
        dags = {}
        for p in paths:
            dag_name = p["nombre_dag"]
            if dag_name not in dags:
                dags[dag_name] = []
            dags[dag_name].append({
                "desde": {"nombre": p["desde"], "tipo": p["tipo_desde"]},
                "hasta": {"nombre": p["hasta"], "tipo": p["tipo_hasta"]},
                "orden": p["orden"],
                "nota": p.get("nota", ""),
            })

        return {
            "tema": tema,
            "tipo": "pathways",
            "descripcion": "Cadenas causales y fisiopatologicas",
            "pathways": [{"nombre": k, "pasos": v} for k, v in dags.items()],
        }

    return {
        "tema": tema,
        "tipo": "pathways",
        "descripcion": "Cadenas causales y fisiopatologicas",
        "pathways": paths,
    }


@router.get("/topic/{tema}/clinical")
async def get_clinical(tema: str):
    """Arbol de decision clinica ante un tema. Bifurcaciones con condiciones."""

    # Buscar relaciones CLINICAL
    steps = query("""
    MATCH (a)-[r:CLINICAL]->(b)
    WHERE toLower(a.nombre) CONTAINS toLower($tema)
       OR toLower(b.nombre) CONTAINS toLower($tema)
       OR r.nombre_dag CONTAINS toLower($tema)
    RETURN r.nombre_dag AS nombre_dag,
           a.nombre AS desde, labels(a)[0] AS tipo_desde,
           b.nombre AS hasta, labels(b)[0] AS tipo_hasta,
           r.orden AS orden, r.tipo_paso AS tipo_paso,
           r.condicion AS condicion, r.nota AS nota
    ORDER BY r.nombre_dag, r.orden
    """, {"tema": tema})

    # Agrupar por nombre_dag
    dags = {}
    for s in steps:
        dag_name = s["nombre_dag"]
        if dag_name not in dags:
            dags[dag_name] = {"nombre": dag_name, "pasos": []}

        dags[dag_name]["pasos"].append({
            "desde": {"nombre": s["desde"], "tipo": s["tipo_desde"]},
            "hasta": {"nombre": s["hasta"], "tipo": s["tipo_hasta"]},
            "orden": s["orden"],
            "tipo_paso": s.get("tipo_paso", ""),
            "condicion": s.get("condicion", ""),
            "nota": s.get("nota", ""),
        })

    return {
        "tema": tema,
        "tipo": "clinical",
        "descripcion": "Arbol de decision clinica",
        "arboles": list(dags.values()),
    }
