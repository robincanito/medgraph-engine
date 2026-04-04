"""Router inteligente unificado — POST /query.
Un solo endpoint que analiza la pregunta, activa las capas necesarias,
y devuelve un paquete completo de conocimiento.
"""

import asyncio
from fastapi import APIRouter
from pydantic import BaseModel
from services.analyzer import analyze_query
from services.layers import (
    execute_ontology,
    execute_graph,
    execute_bibliography,
    execute_activities,
    execute_dags,
)

router = APIRouter(tags=["unified"])

LAYER_MAP = {
    "ONTOLOGY": ("ontologia", execute_ontology),
    "GRAPH": ("grafo", execute_graph),
    "BIBLIOGRAPHY": ("bibliografia", None),  # special handling for top_k
    "ACTIVITIES": ("actividades", execute_activities),
    "DAGS": ("dags", execute_dags),
}


class QueryRequest(BaseModel):
    pregunta: str
    top_k: int = 8


@router.post("/query")
async def unified_query(req: QueryRequest):
    """Consulta inteligente unificada. Analiza la pregunta, detecta entidades,
    activa las capas necesarias (ontología, grafo, bibliografía, actividades, DAGs),
    y devuelve un paquete completo."""

    # 1. Analyzer (Gemini orquestador)
    analysis = await asyncio.to_thread(analyze_query, req.pregunta)

    # 2. Build task list based on activated layers
    tasks = {}
    task_keys = []

    for layer_code in analysis.get("capas", ["BIBLIOGRAPHY"]):
        if layer_code in LAYER_MAP:
            key, fn = LAYER_MAP[layer_code]
            if layer_code == "BIBLIOGRAPHY":
                tasks[key] = asyncio.to_thread(execute_bibliography, analysis, req.top_k)
            elif fn:
                tasks[key] = asyncio.to_thread(fn, analysis)
            task_keys.append(key)

    # 3. Execute all layers in parallel
    if tasks:
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    else:
        results = []

    # 4. Compose response
    response = {
        "pregunta": req.pregunta,
        "analisis": {
            "intencion": analysis.get("intencion", "general"),
            "entidades_detectadas": analysis.get("entidades_detectadas", []),
            "capas_activadas": analysis.get("capas", []),
            "sub_queries": analysis.get("sub_queries", []),
        },
    }

    for key, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            response[key] = {"error": str(result)}
        else:
            response[key] = result

    # 5. Collect bibliographic sources
    fuentes = set()
    bib = response.get("bibliografia", {})
    if isinstance(bib, dict):
        for r in bib.get("results", []):
            if isinstance(r, dict) and r.get("libro"):
                fuentes.add(r["libro"])
    response["fuentes"] = sorted(fuentes)

    # 6. Flags
    response["tiene_ontologia"] = bool(
        response.get("ontologia", {}).get("farmacos_atc")
        or response.get("ontologia", {}).get("categorias_encontradas")
        or response.get("ontologia", {}).get("entidades_snomed")
    )
    response["tiene_grafo"] = bool(
        response.get("grafo", {}).get("patologias")
        or response.get("grafo", {}).get("procedimientos")
        or response.get("grafo", {}).get("relaciones")
    )
    response["tiene_bibliografia"] = bool(
        bib.get("results") if isinstance(bib, dict) else False
    )
    response["tiene_actividades"] = bool(
        response.get("actividades", {}).get("actividades")
    )
    response["tiene_dags"] = bool(
        response.get("dags", {}).get("pathways")
        or response.get("dags", {}).get("clinical")
    )

    # 7. Clarificación si Gemini detectó ambigüedad
    if analysis.get("ambigua"):
        response["necesita_clarificacion"] = True
        response["clarificacion"] = analysis.get("clarificacion")
    else:
        response["necesita_clarificacion"] = False

    return response
