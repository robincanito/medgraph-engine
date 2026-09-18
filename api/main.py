"""MedGraph Engine API — consulta del grafo y administracion (admin/v1) de solo lectura.

QUE ES ESTE SERVICIO. La cara HTTP del pipeline: lo que ya esta cargado en el grafo se busca
(`/search/*`, `/query`, `/topic/*`, `/pathology/*`, `/procedure/*`) y se administra
(`/admin/v1/*`, contrato de nomos-contracts, solo lectura). La INGESTA no vive aca: se corre
llamando a `pipeline/` desde tu propio script, que es lo que documenta el README.

LO QUE SE PODO EL 14-sep-2026, cuando esta API paso de "no arranca" a arrancar: las rutas de la
instancia privada que dependen de nodos que ningun script de este repo escribe —`:UP` y `:Tema`
(temas por unidad problematica), `:Actividad` y `:Documento` (actividades de catedra y su
material)— y `POST /admin/ingest`, que traia su PROPIO parser y una carga destructiva, o sea la
contracara exacta de `pipeline/carga.py`. Una API chica que arranca vale mas que una grande que
promete rutas vacias; el README lo lista endpoint por endpoint.
"""

import logging
import secrets
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# El pipeline (`pipeline/parseo.py`, `pipeline/perfiles.py`) vive FUERA de api/, y esta API lo
# importa: el retrieval filtra por los tipos de contenido que declara el parseo y el descriptor de
# admin/v1 lee el perfil activo. En la imagen, el Dockerfile de la raiz lo copia al lado de este
# archivo; corriendo local desde api/ hay que sumar la raiz del repo al path.
_RAIZ = Path(__file__).resolve().parent.parent
if not (Path(__file__).resolve().parent / "pipeline").exists() and str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from pipeline import perfiles
from routers import (
    admin_v1,
    comprehensive,
    mcp_remote,
    ontology_router,
    pathology,
    procedure,
    search,
    unified,
)
from services import graph
from services.autorizacion import AdminError
from services.settings import get_settings

settings = get_settings()

# FALLA CERRADA. `os.getenv("API_KEY", "")` dejaba la API ABIERTA cuando la variable faltaba: sin
# header, `api_key` valia "" y el `!=` de abajo comparaba "" contra "" y daba acceso a todo. Un
# olvido en el despliegue no puede ser "sin auth": o hay clave o no hay servicio. Se aborta al
# IMPORTAR -no en el primer request- para que el fallo sea del arranque, que es donde se mira.
API_KEY = settings.api_key
if not API_KEY:
    raise RuntimeError(
        "API_KEY no esta definida. La API no arranca sin clave: con la variable vacia "
        "cualquier request sin credencial quedaba autorizado. Copia .env.example a .env y "
        "poné un valor (o exportá API_KEY en el entorno del contenedor)."
    )

# EL PERFIL ACTIVO SE LEE AL ARRANCAR, no en el primer request. Con `PROFILE` apuntando a un YAML
# que no existe, la primera pantalla de la consola -que pide GET /admin/v1/descriptor- seria un 500
# sin explicacion. `perfiles.cargar` falla cerrado y con un mensaje que dice que archivo falta.
perfiles.cargar(settings.profile)

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Corre el session manager del MCP remoto (17-sep-2026).

    Montar un sub-app ASGI NO ejecuta su lifespan, asi que hay que encadenarlo aca o `/mcp` falla en
    el primer request con "Task group is not initialized".
    """
    async with mcp_remote.mcp_lifespan():
        yield


app = FastAPI(
    title="MedGraph Engine API",
    description="Consulta del grafo y administracion admin/v1 sobre lo que ingesto el pipeline",
    version="2.1.0",
    servers=[{"url": settings.public_base_url}],
    lifespan=lifespan,
    # /docs solo con ENVIRONMENT=development EXACTO (y detras de la API key igual, porque el
    # middleware no lo exime): `environment_declarable` mapea lo desconocido a development para
    # que el descriptor valide contra el contrato, y eso no puede decidir que se publica.
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS restrictivo: los origenes se declaran en CORS_ORIGINS (separados por coma). NUNCA "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted({*settings.cors_origins, settings.public_base_url}),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# === AUTH ===
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Unica ruta libre: el health del servicio, que es lo que sondea un balanceador.
    # OJO: /admin/v1/health NO esta exento — el contrato lo declara con 401.
    if request.url.path == "/health":
        return await call_next(request)

    api_key = (
        request.headers.get("X-API-Key")
        or request.headers.get("Authorization", "").replace("Bearer ", "")
    )
    # `compare_digest` y no `==`: la comparacion normal corta en el primer byte que difiere, y ese
    # tiempo filtra el secreto. Se comparan BYTES porque `compare_digest` de dos `str` revienta con
    # un ValueError si alguno trae un caracter no ASCII, y el header lo escribe quien llama: eso
    # convertiria una credencial mal tipeada en un 500.
    if not (api_key and secrets.compare_digest(api_key.encode("utf-8"), API_KEY.encode("utf-8"))):
        # SE DEVUELVE LA RESPUESTA, NO SE LEVANTA LA EXCEPCION (arreglado 14-sep-2026). Aca habia
        # un `raise HTTPException(401)`, y un HTTPException levantado DENTRO de un middleware no
        # pasa por los handlers de FastAPI: sube hasta el ServerErrorMiddleware, que responde
        # 500. O sea que TODA request sin credencial recibia "500 Internal Server Error" en vez
        # de un 401 — el acceso quedaba denegado igual, pero ningun cliente podia distinguir
        # "te falta la clave" de "el servidor se rompio", y `tests/test_admin_v1.py` lo fija.
        # El `WWW-Authenticate` lo pide el contrato admin/v1 para el 401.
        #
        # EN `/mcp` EL CUERPO DICE COMO ENTRAR (17-sep-2026). Un cliente MCP que recibe un 401 busca
        # en el header a que servidor de autorizacion ir (`resource_metadata`, RFC 9728) y acá NO HAY
        # NINGUNO: esta instancia autentica con una clave compartida. Anunciar un flujo de OAuth que
        # no existe mandaria al cliente a un descubrimiento que termina en 404; callar deja a quien
        # conecta sin saber que le falta. Asi que el header queda como esta --sin `resource_metadata`
        # y sin `scope`-- y el cuerpo explica la credencial, que es lo unico honesto que se puede
        # decir. Un despliegue que quiera OAuth por persona pone un authorization server adelante y
        # publica su propia metadata: el contrato mcp/v1 lo contempla (`auth.mcp.oauth` es opcional).
        if request.url.path.startswith("/mcp"):
            return JSONResponse(
                {"detail": "This MCP endpoint requires the instance API key. Send it as "
                           "'Authorization: Bearer <key>' or as the 'X-API-Key' header. There is no "
                           "OAuth authorization server for this deployment: the key is issued by "
                           "whoever operates it.",
                 "code": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="medgraph-engine"'},
            )
        return JSONResponse(
            {"detail": "Invalid or missing API key.", "code": "unauthorized"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="medgraph-engine"'},
        )

    return await call_next(request)


# === LOGGING ===
@app.middleware("http")
async def log_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    # `logging` y no `print`: asi quien despliega elige formato y destino (y en Cloud Logging
    # cada linea es un registro con su severidad, no texto suelto en stdout).
    logging.info("%s %s -> %s (%.2fs)", request.method, request.url.path,
                 response.status_code, duration)
    return response


# === ERRORES DE admin/v1 ===
@app.exception_handler(AdminError)
async def admin_error_handler(request: Request, exc: AdminError):
    """El contrato pide `{detail, code}` en la RAIZ: la consola enruta por `code`."""
    return JSONResponse({"detail": exc.detail, "code": exc.code}, status_code=exc.status)


# === ROUTERS ===
app.include_router(search.router)
app.include_router(pathology.router)
app.include_router(procedure.router)
app.include_router(comprehensive.router)
app.include_router(ontology_router.router)
app.include_router(unified.router)
app.include_router(admin_v1.router)  # contrato admin/v1 (nomos-contracts)


# === HEALTH ===
@app.get("/health")
async def health():
    """Sonda del servicio, sin credencial. El health del contrato es /admin/v1/health."""
    try:
        graph.query("RETURN 1")
        return {"status": "ok", "service": "medgraph-engine-api", "db": "connected"}
    except Exception:
        return JSONResponse(
            {"status": "degraded", "service": "medgraph-engine-api", "db": "disconnected"}, 503)


@app.get("/stats")
async def stats():
    return graph.get_stats()


# === MCP REMOTO (Streamable HTTP) ===
# DOS RUTAS EXACTAS, NO UN MOUNT (18-sep-2026). Montarlo en "/mcp" da 307 en "/mcp/" y un redirect
# en POST es fragil (varios clientes descartan el body); montarlo en la RAIZ resolvia eso pero se
# comia los 405 de TODA la API, porque un mount matchea por path y no por metodo. Con `Route`
# exactas no hay ni 307 ni captura: ver `mcp_remote.rutas_mcp`.
#
# YA NO HACE FALTA QUE SEA LO ULTIMO DEL ARCHIVO: una ruta declarada despues ya no queda tapada.
#
# La puerta es el middleware de arriba: `/mcp` no esta exento, asi que exige la misma clave que el
# resto de la API.
app.router.routes.extend(mcp_remote.rutas_mcp())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
