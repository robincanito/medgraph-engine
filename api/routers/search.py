"""Endpoints de búsqueda: full-text, semántica e híbrida con RRF."""

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from pydantic import BaseModel, ConfigDict, Field

from services import vector

router = APIRouter(prefix="/search", tags=["search"])

#: Cuanto esperar cuando el grafo no contesta. No es una constante de estilo: viaja en el header
#: `Retry-After` y en el cuerpo, y es lo que separa un "volve en medio minuto" de un "esto esta
#: roto". Si el despliegue de quien opere esto tarda mas en recuperarse, este es el numero a mover.
RETRY_AFTER_GRAFO = 30


class FiltrosFuente(BaseModel):
    """QUE FUENTES PUEDE MIRAR LA BUSQUEDA (16-sep-2026).

    Hasta hoy el modelo de petición era `query`, `top_k` y **un** `libro_id`: para acotar la
    búsqueda a un subconjunto de fuentes había que hacer una llamada por fuente y pegar los
    resultados a mano. Con listas es una sola llamada.

    Las cinco dimensiones se combinan con AND. Las tres últimas son propiedades de la FUENTE
    (`:Book`): `collections` (etiquetas), `areas` (`b.area`) y `tipos` (`b.source_kind`).
    `excluir` es una lista negra de fuentes.

    `extra="forbid"` A PROPOSITO: un `libro_id` mal escrito adentro de `filtros` sería un filtro
    silenciosamente ignorado, o sea una respuesta de todo el corpus cuando se pidió una fuente.
    Mejor un 422 que diga el nombre correcto.
    """

    model_config = ConfigDict(extra="forbid")

    libro_ids: list[str] | None = None
    collections: list[str] | None = None
    areas: list[str] | None = None
    tipos: list[str] | None = None
    excluir: list[str] | None = None


class Garantia(BaseModel):
    """QUE FUENTES DEBEN ENTRAR SI TIENEN ALGO (16-sep-2026).

    Además de la pasada general se corre UNA pasada restringida a estas fuentes, y hasta `cupo`
    lugares del `top_k` se les reservan **si pasan el piso de score**
    (`vector.PISO_GARANTIA_FRACCION`). Si la colección no tiene nada del tema el cupo queda vacío
    y `procedencia.garantizados` lo dice: la garantía no inventa material.

    `cupo` por defecto: la mitad del `top_k`, mínimo 1 (`vector.cupo_por_defecto`).
    """

    model_config = ConfigDict(extra="forbid")

    collections: list[str] | None = None
    libro_ids: list[str] | None = None
    cupo: int | None = Field(default=None, ge=1, le=vector.TOPE_TOP_K)


class SearchRequest(BaseModel):
    query: str
    # EL TOPE (16-sep-2026): `top_k` no tenía ninguno y un `top_k: 5000` devolvía el pool entero.
    # El número vive en `vector.TOPE_TOP_K`, para que todas las puertas del despliegue lean el
    # mismo.
    top_k: int = Field(default=20, ge=1, le=vector.TOPE_TOP_K)
    libro_id: str | None = None
    filtros: FiltrosFuente | None = None
    garantizar: Garantia | None = None
    # `None` por defecto: esta ruta NO agrupa salvo que se lo pidan. Un consumidor que ya existe
    # espera un pasaje por lugar.
    agrupar: Literal["padre", "pagina"] | None = None
    max_por_fuente: int | None = Field(default=None, ge=1, le=vector.TOPE_TOP_K)


def _dict(modelo) -> dict | None:
    """El modelo Pydantic como dict sin los campos que no vinieron (`None` = no filtrar)."""
    return modelo.model_dump(exclude_none=True) if modelo is not None else None


@router.post("/semantic")
async def semantic_search(req: SearchRequest):
    """Búsqueda por significado usando embeddings (vector KNN)."""
    results = vector.search_semantic(req.query, req.top_k, req.libro_id, _dict(req.filtros))
    return {"query": req.query, "results": results, "count": len(results)}


@router.post("/keyword")
async def keyword_search(req: SearchRequest):
    """Búsqueda full-text con scoring BM25 (Lucene via Neo4j)."""
    results = vector.search_keyword(req.query, req.top_k, req.libro_id, _dict(req.filtros))
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

    LO QUE SE PUEDE PEDIR DESDE EL 16-sep-2026: `filtros` (acotar por fuente, colección, área,
    tipo o lista negra), `garantizar` (reservar cupo para las fuentes que el cliente nombra) y
    `agrupar` (colapsar por `:ParentChunk` o por página). Y lo que la respuesta devuelve de más:
    `procedencia` —conteo por fuente en el `top_k` y en el pool— y `calidad` por resultado.
    """
    try:
        result = vector.search_hybrid(req.query, req.top_k, req.libro_id,
                                      filtros=_dict(req.filtros),
                                      garantizar=_dict(req.garantizar),
                                      agrupar=req.agrupar,
                                      max_por_fuente=req.max_por_fuente)
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
        # LA PROCEDENCIA VIAJA SIEMPRE (16-sep-2026). Sale gratis: los dos conteos ya se calculaban
        # para la telemetria y morian en el log. `pool` es lo que distingue "el ranking la
        # entierra" de "no tiene material", que son dos problemas con curas opuestas.
        "procedencia": result.get("procedencia", {}),
    }
    # Solo cuando hay algo que decir: un aviso que aparece siempre deja de leerse.
    if result.get("degradado"):
        salida["degradado"] = result["degradado"]
        if result.get("fallas"):
            salida["fallas"] = result["fallas"]
    return salida
