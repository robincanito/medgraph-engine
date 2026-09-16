"""Endpoints de búsqueda: full-text, semántica e híbrida con RRF."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from pydantic import BaseModel

from services import vector

router = APIRouter(prefix="/search", tags=["search"])

#: Cuanto esperar cuando el grafo no contesta. No es una constante de estilo: viaja en el header
#: `Retry-After` y en el cuerpo, y es lo que separa un "volve en medio minuto" de un "esto esta
#: roto". Si el despliegue de quien opere esto tarda mas en recuperarse, este es el numero a mover.
RETRY_AFTER_GRAFO = 30


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    libro_id: str | None = None


@router.post("/semantic")
async def semantic_search(req: SearchRequest):
    """Búsqueda por significado usando embeddings (vector KNN)."""
    results = vector.search_semantic(req.query, req.top_k, req.libro_id)
    return {"query": req.query, "results": results, "count": len(results)}


@router.post("/keyword")
async def keyword_search(req: SearchRequest):
    """Búsqueda full-text con scoring BM25 (Lucene via Neo4j)."""
    results = vector.search_keyword(req.query, req.top_k, req.libro_id)
    return {"query": req.query, "results": results, "count": len(results)}


def _503(code: str, detail: str, retry_after: int | None = None, **extra) -> JSONResponse:
    """503 con `code` enrutable, `detail` legible y, cuando hay algo que esperar, `Retry-After`."""
    cuerpo = {"status": code, "code": code, "detail": detail}
    if retry_after is not None:
        cuerpo["retry_after_seconds"] = retry_after
    cuerpo.update(extra)
    return JSONResponse(cuerpo, status_code=503,
                        headers={"Retry-After": str(retry_after)} if retry_after else None)


def _detalle_de_fallas(fallas: list) -> str:
    """"No se pudo buscar" con nombre y apellido: que ruta se cayo y con que error."""
    if not fallas:
        return ("No se pudo consultar el corpus: ninguna ruta de busqueda devolvio resultados. "
                "Esto NO significa que no haya material sobre el tema.")
    detalle = ", ".join(f"{f.get('ruta', '?')} ({f.get('error', '?')})" for f in fallas)
    return (f"No se pudo consultar el corpus: fallaron las rutas de busqueda {detalle}. "
            "Esto NO significa que no haya material sobre el tema: no se pudo preguntar.")


@router.post("/hybrid")
async def hybrid_search(req: SearchRequest):
    """Búsqueda híbrida: full-text + semántica fusionadas con Reciprocal Rank Fusion. Usar esta por defecto.

    UN 200 CON `results: []` SIGNIFICA UNA SOLA COSA: no hay material (16-sep-2026). Hasta hoy
    significaba dos --eso, y "no se pudo consultar el corpus"-- y desde afuera eran
    indistinguibles: el consumidor leia el vacio como ausencia, respondia de su propio conocimiento
    y no avisaba que la bibliografia no se habia consultado. En una herramienta cuya promesa es
    "cita fuente y pagina", eso es un fallo de CORRECCION, no de disponibilidad.

      · **503 `graph_unavailable`** — el grafo no contesta (`search_hybrid` lanza). Con
        `Retry-After`: es una dependencia caida, no un bug del servicio, y un 500 le diria al
        cliente "no reintentes".
      · **503 `retrieval_unavailable`** — se cayeron todas las rutas por otra causa (el proveedor
        de embeddings, el parser de la consulta). El grafo puede estar sano; el corpus no se
        consulto igual.
      · **200** — hay resultados (con `degradado: "parcial"` y `fallas` si una ruta se cayo), o no
        hay material.

    El 503 se arma ACA y no en un handler global porque `api/main.py` de este repo no tiene
    ninguno: la ruta se basta sola y quien monte esta app no necesita saber que existe.
    """
    try:
        result = vector.search_hybrid(req.query, req.top_k, req.libro_id)
    except (ServiceUnavailable, SessionExpired) as e:
        return _503("graph_unavailable",
                    "El grafo no responde en este momento. No es que no haya material sobre el "
                    f"tema: no se pudo consultar ({type(e).__name__}). "
                    f"Reintenta la misma consulta en ~{RETRY_AFTER_GRAFO} segundos.",
                    retry_after=RETRY_AFTER_GRAFO)

    if result.get("degradado") == "total":
        return _503("retrieval_unavailable", _detalle_de_fallas(result.get("fallas") or []),
                    fallas=result.get("fallas") or [])

    salida = {
        "query": req.query,
        "keyword_count": result["keyword_count"],
        "semantic_count": result["semantic_count"],
        "intencion": result.get("intencion", "general"),
        "query_expandida": result.get("query_expandida", req.query),
        "results": result["results"],
        "count": len(result["results"]),
    }
    # Solo cuando hay algo que decir: un aviso que aparece siempre deja de leerse.
    if result.get("degradado"):
        salida["degradado"] = result["degradado"]
        if result.get("fallas"):
            salida["fallas"] = result["fallas"]
    return salida
