"""MedGraph API — Knowledge base medica para estudio."""

import os
import secrets
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from routers import topics, search, pathology, procedure, activity, admin, comprehensive, ontology_router, unified
from services import graph

load_dotenv()

# FALLA CERRADA, Y ES LO QUE ARREGLA EL ITEM M-4 (13-sep-2026). `os.getenv("API_KEY", "")` dejaba
# la API ABIERTA cuando la variable faltaba: sin header, `api_key` valia "" y el `!=` de abajo
# comparaba "" contra "" y daba acceso a todo. Un olvido en el despliegue no puede ser "sin auth":
# o hay clave o no hay servicio. Se aborta al IMPORTAR -no en el primer request- para que el fallo
# sea del arranque, que es donde se mira.
API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "API_KEY no esta definida. La API no arranca sin clave: con la variable vacia "
        "cualquier request sin credencial quedaba autorizado. Copia .env.example a .env y "
        "poné un valor (o exportá API_KEY en el entorno del contenedor)."
    )

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

app = FastAPI(
    title="MedGraph API",
    description="Knowledge base medica con grafo de conocimiento y busqueda semantica",
    version="2.0.0",
    servers=[{"url": os.getenv("API_BASE_URL", "http://localhost:8000")}],
    docs_url="/docs" if os.getenv("ENVIRONMENT") == "development" else None,
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS restrictivo
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("CORS_ORIGIN_1", "http://localhost:3000"),
        # Add your frontend origins here
        
        os.getenv("API_BASE_URL", "http://localhost:8000"),
        # Add your frontend origins here
        
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# === AUTH ===
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Skip auth for health and schema
    if request.url.path in ("/health", "/chatgpt-schema"):
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
        raise HTTPException(status_code=401, detail="Invalid API key")

    return await call_next(request)


# === LOGGING ===
@app.middleware("http")
async def log_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"{request.method} {request.url.path} -> {response.status_code} ({duration:.2f}s)")
    return response


# === ROUTERS ===
app.include_router(topics.router)
app.include_router(search.router)
app.include_router(pathology.router)
app.include_router(procedure.router)
app.include_router(activity.router)
app.include_router(admin.router)
app.include_router(comprehensive.router)
app.include_router(ontology_router.router)
app.include_router(unified.router)


# === HEALTH ===
@app.get("/health")
async def health():
    try:
        graph.query("RETURN 1")
        return {"status": "ok", "service": "medgraph-api", "db": "connected"}
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "degraded", "service": "medgraph-api", "db": "disconnected"}, 503)


@app.get("/chatgpt-schema")
async def chatgpt_schema():
    """Schema reducido (5 endpoints) para importar en ChatGPT Actions."""
    from fastapi.responses import JSONResponse
    import json
    for path in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatgpt_action_schema.json"),
        "/app/chatgpt_action_schema.json",
        "chatgpt_action_schema.json",
    ]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return JSONResponse(content=json.load(f))
    return {"error": "schema not found"}


@app.get("/stats")
async def stats():
    return graph.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
