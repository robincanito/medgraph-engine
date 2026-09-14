"""Rutas de /admin/v1 (contrato nomos-contracts, openapi/admin-v1.yaml). SOLO LECTURA.

Las cinco que implementa este repo: descriptor, health, stats, sources, sources/{id}. Las demas
del contrato (facets, units, PATCH, DELETE, vectorize, reingest, jobs, uploads) no existen aca y
`admin_v1.CAPABILITIES` las publica en false con su nota, que es como la consola se entera sin
probar. El contrato es explicito: publicar una capacidad sin endpoint es mentirle a la consola.

AUTENTICACION: la misma API key que el resto de la API, en el middleware de `main.py`
(`Authorization: Bearer` o `X-API-Key`). No hay roles: admin/v1 aca no escribe nada, asi que no
hay escritura que separar de una lectura. El descriptor lo publica tal cual.

`AdminError` PASA DE LARGO en cada ruta y no se envuelve: ya trae el codigo del contrato (404
not_found, 422 validation_error) y la consola enruta por ese codigo. Cualquier otra excepcion si
es un grafo que no contesta, y sale como 503 graph_unavailable.
"""
from fastapi import APIRouter, Query, Request

from services import admin_v1
from services.autorizacion import AdminError

router = APIRouter(prefix="/admin/v1", tags=["admin/v1"])

PREFIJO_FILTRO = "filter["


def _filtros_de(request: Request) -> dict:
    """`filter[autor]=Autora De Ejemplo` -> {"autor": "Autora De Ejemplo"}.

    Los filtros por campo no se declaran uno por uno en la firma porque dependen del descriptor,
    que depende del perfil: se leen crudos y `admin_v1` rechaza los que no son filtrables.
    """
    return {clave[len(PREFIJO_FILTRO):-1]: valor
            for clave, valor in request.query_params.items()
            if clave.startswith(PREFIJO_FILTRO) and clave.endswith("]")}


@router.get("/descriptor")
def get_descriptor():
    """Lo primero que pide la consola: vocabulario, campos, pasos y capacidades."""
    return admin_v1.descriptor()


@router.get("/health")
def get_health():
    """Estado de la API y del grafo. No falla si el grafo no contesta: lo dice en el cuerpo."""
    return admin_v1.health()


@router.get("/stats")
def get_stats():
    try:
        return admin_v1.stats()
    except AdminError:
        raise
    except Exception as e:
        raise AdminError(503, "graph_unavailable",
                         "The graph is not answering: statistics cannot be computed.") from e


@router.get("/sources")
def get_sources(request: Request,
                q: str | None = None,
                kind: str | None = None,
                status: str | None = None,
                sort: str | None = None,
                limit: int = Query(50, ge=1, le=admin_v1.LIMITE_MAX),
                cursor: str | None = None):
    try:
        return admin_v1.listar_fuentes(q=q, kind=kind, status=status,
                                       filtros=_filtros_de(request), sort=sort,
                                       limit=limit, cursor=cursor)
    except AdminError:
        raise
    except Exception as e:
        raise AdminError(503, "graph_unavailable", "The graph is not answering.") from e


@router.get("/sources/{source_id}")
def get_source(source_id: str):
    try:
        return admin_v1.obtener_fuente(source_id)
    except AdminError:
        raise
    except Exception as e:
        raise AdminError(503, "graph_unavailable", "The graph is not answering.") from e
