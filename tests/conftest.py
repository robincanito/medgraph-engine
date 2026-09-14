"""Fixtures compartidas de la suite del engine.

UNA SOLA RAIZ DE IMPORT. El repo no es un paquete instalable: `pipeline/` y los scripts de
la raiz se importan por ruta. Se agrega la raiz del repo a `sys.path` aca, antes de
cualquier import de la suite, para que `from pipeline import parseo` funcione igual si
pytest se corre desde la raiz o desde otro directorio.

CERO RED, CERO GRAFO, CERO KEYS (la doctrina de la casa). Ningun test de esta suite abre
una conexion ni llama a un proveedor: el cliente de embeddings es un doble, el grafo es un
doble de `query`/`write` y los PDF se generan en el test con PyMuPDF. Las variables de
entorno se siembran con valores de MENTIRA porque `db.py` y `api/services/graph.py` leen
NEO4J_* al importarse; `bolt://127.0.0.1:1` apunta a un puerto cerrado a proposito, asi que
si algun test intentara conectarse de verdad falla rapido en vez de colgarse.
"""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
# Y la raiz de la API, que NO es un paquete: `main`, `services` y `routers` se importan como
# modulos de primer nivel (es como los resuelve `uvicorn main:app` desde api/ y como los copia el
# Dockerfile). Va DESPUES de la raiz del repo para que `from pipeline import ...` siga saliendo de
# la raiz, que es donde vive el espejo del pipeline.
sys.path.insert(1, str(RAIZ / "api"))

os.environ.setdefault("NEO4J_URI", "bolt://127.0.0.1:1")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password-not-real")
os.environ.setdefault("NEO4J_DATABASE", "neo4j")
os.environ.setdefault("API_KEY", "test-api-key-not-real")
os.environ.setdefault("GCP_API_KEY", "test-key-not-real")
os.environ.setdefault("ENVIRONMENT", "test")
