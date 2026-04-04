import os
"""MCP Server para MedGraph — conecta Claude directamente a la base de conocimiento medica."""

import json
import httpx
from mcp.server.fastmcp import FastMCP

API_URL = os.getenv("MEDGRAPH_API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

mcp = FastMCP(
    "MedGraph",
    instructions="""Sos un asistente de estudio medico. Tenes acceso a MedGraph, una base de conocimiento
con bibliografia indexada, un grafo de conceptos medicos interconectados, y material academico.

Reglas:
- SIEMPRE usar medgraph_query como primera opcion. Es el endpoint unificado que analiza la pregunta y activa automaticamente las capas necesarias (ontologia, grafo, bibliografia, actividades, flujos clinicos).
- Solo usar las tools especificas (medgraph_search, medgraph_activity, etc.) si necesitas algo muy puntual que medgraph_query no cubrio.
- Siempre citar fuentes (libro, paginas).
- No inventar. Si no encontras datos, decirlo.
- Responder en español.""",
)


async def _api_get(path: str) -> dict:
    async with httpx.AsyncClient(verify=True, timeout=60) as client:
        r = await client.get(f"{API_URL}{path}", headers=HEADERS)
        r.raise_for_status()
        return r.json()


async def _api_post(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(verify=True, timeout=60) as client:
        r = await client.post(f"{API_URL}{path}", headers=HEADERS, json=body)
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def medgraph_query(pregunta: str, top_k: int = 8) -> str:
    """Consulta inteligente unificada a MedGraph. Analiza la pregunta automaticamente
    y activa las capas necesarias: ontologia (clasificacion ATC/SNOMED), grafo de
    conocimiento (relaciones entre entidades), bibliografia (busqueda en libros),
    actividades academicas (TPs, seminarios), y flujos clinicos (DAGs).
    USAR SIEMPRE COMO PRIMERA OPCION para cualquier pregunta medica."""
    data = await _api_post("/query", {"pregunta": pregunta, "top_k": top_k})
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def medgraph_comprehensive(tema: str) -> str:
    """Obtener TODO sobre un tema medico: grafo de conocimiento, actividades academicas,
    course materials y bibliografia con citas exactas. Usar para preguntas amplias
    como 'contame sobre otitis', 'todo sobre HTA', 'preparame para el TP de otoscopia'."""
    data = await _api_get(f"/topic/{tema}/comprehensive")
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def medgraph_search(query: str, top_k: int = 5, libro_id: str = "") -> str:
    """Buscar en la bibliografia medica indexada. Busqueda hibrida (semantica + keywords).
    Usar para preguntas puntuales como 'dosis de amoxicilina', 'valores normales de hemograma',
    'definicion de fovea'. Libros disponibles: farreras-2020, harrison-manual, garcia-feijoo-oftalmo,
    diamante-orl, balcells-laboratorio, goodman-gilman-farma, sanguinetti-semiologia,
    and more. Configure your own books via the pipeline.
    your indexed books appear here after running the pipeline.
    
    """
    body = {"query": query, "top_k": top_k}
    if libro_id:
        body["libro_id"] = libro_id
    data = await _api_post("/search/hybrid", body)
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def medgraph_pathways(tema: str) -> str:
    """Obtener la cadena causal/fisiopatologica de un tema. Secuencia determinista:
    agente -> mecanismo -> efecto -> signo -> complicacion.
    Usar cuando pregunten 'por que se produce X', 'fisiopatologia de X', 'mecanismo de X'."""
    data = await _api_get(f"/topic/{tema}/pathways")
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def medgraph_clinical(tema: str) -> str:
    """Obtener el arbol de decision clinica ante un tema. Bifurcaciones con condiciones.
    Usar cuando pregunten 'que hago ante un paciente con X', 'como manejo X',
    'diagnostico diferencial de X'."""
    data = await _api_get(f"/topic/{tema}/clinical")
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def medgraph_activity(nombre: str) -> str:
    """Buscar una actividad academica (TP, Seminario, Taller, Acreditacion) por nombre.
    Usar cuando mencionen un TP, seminario o taller especifico."""
    data = await _api_get(f"/activity/search/{nombre}")
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def medgraph_activity_material(activity_id: str) -> str:
    """Obtener el contenido completo de los documentos de una actividad (guias, procedimientos).
    Usar despues de medgraph_activity para leer el material."""
    data = await _api_get(f"/activity/{activity_id}/material")
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def medgraph_pathology(nombre: str) -> str:
    """Informacion estructurada de una patologia del grafo: definicion, diagnostico,
    tratamiento, signos, agentes, fuentes bibliograficas."""
    data = await _api_get(f"/pathology/{nombre}")
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def medgraph_procedure(nombre: str) -> str:
    """Pasos, insumos, hallazgos y parametros de un procedimiento medico."""
    data = await _api_get(f"/procedure/{nombre}")
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
async def medgraph_ontology(tema: str) -> str:
    """Clasificacion ontologica de un tema medico: jerarquia ATC de farmacos,
    jerarquia SNOMED de patologias/anatomia/procedimientos, y queries cruzadas
    entre sistemas farmacologicos y sistemas corporales.
    Usar cuando pregunten 'a que clase pertenece X', 'que tipo de farmaco es X',
    'que sistema afecta X', 'farmacos de clase X para patologias de sistema Y'."""
    data = await _api_get(f"/topic/{tema}/ontology")
    return json.dumps(data, ensure_ascii=False, indent=2)


SCHEDULE_API_URL = os.getenv("SCHEDULE_API_URL", "http://localhost:8001")
SCHEDULE_API_KEY = os.getenv("SCHEDULE_API_KEY", "")
SCHEDULE_HEADERS = {"X-Medgraph-Key": SCHEDULE_API_KEY}


@mcp.tool()
async def medgraph_cronograma(semana_numero: int = 0) -> str:
    """Obtener el cronograma semanal de la carrera de medicina. Incluye todas las
    actividades con horarios, tipo, area, si es obligatoria,
    docente y lugar. Si semana_numero es 0, devuelve la semana actual."""
    async with httpx.AsyncClient(verify=True, timeout=60) as client:
        params = {"group": "A", "subgroup": "A1"}
        if semana_numero > 0:
            params["semana_numero"] = semana_numero
        r = await client.get(
            f"{SCHEDULE_API_URL}/api/v1/cronograma" # Configure your schedule API,
            headers=SCHEDULE_HEADERS,
            params=params,
        )
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
