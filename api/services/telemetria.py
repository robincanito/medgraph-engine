"""Telemetria de uso del MCP: que herramientas se piden y que devuelven.

POR QUE EXISTE (10-sep-2026). MedGraph tiene capas caras de construir y de mantener —entidades,
ontologia ATC/SNOMED, DAGs de razonamiento, busqueda por facetas— y **no sabiamos cuales se
usan**. La noche del 9-sep hubo 7 llamadas al MCP desde ChatGPT y lo unico que se pudo leer en
los logs fue "hubo 7 llamadas": ni que herramienta, ni si el resultado traia entidades o solo
pasajes. Decidir si profundizar la ontologia o los DAGs sin ese dato es apostar.

Esto es un LOG, no un proyecto: una linea JSON por llamada. Con dos semanas de datos se sabe que
capa paga su costo. Ver la nota de diseno sobre degradacion en el README (seccion Known gaps).

QUE SE REGISTRA Y QUE NO. Se registra la FORMA de la llamada y del resultado: herramienta,
duracion, si hubo error, cuantos resultados, y si vinieron entidades o subgrafo. **NO se registra
el texto de la consulta.** Son preguntas de quien opera la instancia sobre su propio corpus, y mandarlas a
Cloud Logging es una decision de privacidad que nadie tomo; para la pregunta que se quiere
responder —que capa se usa— el nombre de la herramienta y la forma del resultado alcanzan. Si
algun dia hace falta el texto, se activa a proposito y se dice.

CANAL PROPIO A PROPOSITO. La API no configura logging (lo hace uvicorn, y su formato antepone
nivel y logger). Una linea que no es JSON puro llega a Cloud Logging como `textPayload` y deja de
ser consultable por campo. Con handler propio, `propagate = False` y formato `%(message)s`, cada
evento sale como JSON puro y Cloud Logging lo parsea a `jsonPayload`:

    gcloud logging read 'jsonPayload.evento="mcp_tool" AND jsonPayload.tool="deep_dive"'

FALLA ABIERTO, no cerrado. Es lo contrario a la doctrina de la casa, y es deliberado: la
telemetria no puede romper una consulta. Si registrar falla, se traga la excepcion y la
herramienta responde igual.
"""
import functools
import json
import logging
import sys
import time

_log = logging.getLogger("medgraph.telemetria")
_log.propagate = False  # el formato de uvicorn romperia el JSON puro
if not _log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)

# Claves donde los tools devuelven listas de resultados. Se cuenta la primera que aparezca.
_CLAVES_RESULTADOS = ("results", "items", "passages", "chunks", "entities", "sources", "facets")


def _forma_del_resultado(valor) -> dict:
    """Cuantos resultados y que capas vinieron. Nunca mira el contenido, solo la forma."""
    datos = valor
    # Los tools del router devuelven modelos de pydantic; los de services, dicts.
    if hasattr(datos, "model_dump"):
        try:
            datos = datos.model_dump()
        except Exception:
            return {}
    if not isinstance(datos, dict):
        return {"n": len(datos)} if isinstance(datos, list) else {}

    forma = {}
    for clave in _CLAVES_RESULTADOS:
        v = datos.get(clave)
        if isinstance(v, list):
            forma[f"n_{clave}"] = len(v)
        elif isinstance(v, dict):
            forma[f"n_{clave}"] = len(v)
    # Las dos preguntas que motivaron todo esto: ¿se uso el grafo, o fue RAG a secas?
    forma["con_entidades"] = bool(datos.get("entities"))
    forma["con_grafo"] = bool(datos.get("graph") or datos.get("subgraph") or datos.get("subgrafo"))
    return forma


def registrar(evento: str, **campos) -> None:
    """Una linea JSON. Se traga cualquier error: la telemetria no rompe una consulta."""
    try:
        _log.info(json.dumps({"evento": evento, **campos}, default=str, ensure_ascii=False))
    except Exception:
        pass


def medir(nombre: str):
    """Envuelve un tool del MCP para registrar su uso.

    OJO CON EL ORDEN DE LOS DECORADORES. Este va DEBAJO de `@mcp_server.tool(...)`, o sea mas
    cerca de la funcion: FastMCP tiene que registrar el wrapper para que la medicion ocurra.
    `functools.wraps` copia nombre, docstring y `__wrapped__`, y `inspect.signature` sigue
    `__wrapped__`, asi que el schema que FastMCP le publica al cliente no cambia — hay un test
    que lo fija, porque si cambiara, ChatGPT dejaria de ver los argumentos correctos.
    """
    def envoltura(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                salida = fn(*args, **kwargs)
            except Exception as e:
                registrar("mcp_tool", tool=nombre, ok=False,
                          ms=round((time.perf_counter() - t0) * 1000),
                          error=type(e).__name__)
                raise
            registrar("mcp_tool", tool=nombre, ok=True,
                      ms=round((time.perf_counter() - t0) * 1000),
                      **_forma_del_resultado(salida))
            return salida
        return wrapper
    return envoltura
