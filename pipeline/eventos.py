"""Eventos nombrados del pipeline: una corrida se puede juzgar desde AFUERA.

POR QUE (agujero G7 de docs/DISENO-qa-pipeline-13sep.md, tanda 1, 13-sep-2026). La bitacora
emitia prosa ("  [3] 41 embeddings"): nadie podia atribuir una llamada paga a un documento,
ni saber por que paso iba una ingesta, ni contar reintentos. Un oraculo de QA necesita
EVENTOS con nombre y campos, no texto libre.

COMO. Sobre `logging` de la biblioteca estandar y NADA MAS. Este modulo vive en `pipeline/`,
que se replica byte a byte al engine OSS (medgraph-engine), y el engine NO tiene `bitacora.py`:
por eso aca no se importa. Quien quiera ver los eventos formateados configura un handler; en
medgraph lo hace `bitacora.configurar()` (el formato JSON aplana los campos al nivel raiz para
Cloud Logging, y el de consola agrega ` k=v` al final de la linea).

VOCABULARIO. Es un CONTRATO con el harness de QA (tests/harness): no se renombra ni se recorta
un campo sin actualizar los oraculos que lo leen.

  ingest_paso   libro_id, paso, origen, estado, ms, n?, tipo?, detalle?
                paso   in {download, classify, parse, upload, vectorize, fuente, extract}
                estado in {inicio, fin, error}       origen in {cli, api}
                `ms` (entero) va en fin y en error. `n` es el conteo natural del paso
                (chunks, embeddings) y lo pone el cuerpo del `with` en `handle.campos`.
                `tipo` / `detalle` (<=200 chars) solo en error.

  embed_lote    libro_id, n_textos, intento, estado, ms, detalle?
                estado in {ok, reintento, fallo}; `intento` arranca en 1.
                `detalle` (<=200 chars) va en reintento y en fallo.

  llm_llamada   libro_id, paso, modelo, n_chunks, estado, ms
                paso in {classify, extract}; estado in {ok, error}. `libro_id` puede ser
                None: el clasificador corre antes de que el libro exista.

  decodificacion_fuente libro_id, fuente, metodo, k, codigos, confianza, chars, estado
                metodo in {corrimiento, sustitucion}; estado in {aplicada, descartada}.
                UNO POR FUENTE TIPOGRAFICA CON EL MAPEO ROTO (16-sep-2026, tanda 6 del diseño
                de uso real). `pipeline/decodificacion.py` recupera el texto de un PDF cuyos
                glifos no se pueden traducir, y eso CAMBIA EL TEXTO que entra al corpus: tiene
                que quedar dicho que fuente se toco, con que metodo (`k` es el corrimiento, o
                None si fue sustitucion), cuantos codigos resolvio y con que confianza —la
                fraccion de palabras reconocidas—. `estado=descartada` es el caso en que no
                llego al umbral: ahi el texto queda como estaba y la calidad lo marca.

  extraccion_descarte   libro_id, chunk_id, motivo, tipo?, relacion?, desde_tipo?, hasta_tipo?
                motivo in {tipo_entidad, tipo_relacion, nombre_corto, nombre_largo, from_to,
                extremo_ausente}. UNO POR ITEM DESCARTADO (14-sep-2026, decision E). Hasta hoy
                lo que el validador tiraba se perdia en silencio y nadie podia medir la
                diferencia entre lo que el modelo devolvio y lo que quedo en el grafo.
                `from_to` y `extremo_ausente` se emiten SIN rechazar: son la evidencia con la
                que se decide si el rechazo estricto se enciende (ver extraccion.py).

Las claves del sobre JSON (severity, message, logger, time, evento, exception) estan PROHIBIDAS
como nombre de campo: el formateador aplana los campos al nivel raiz y una colision pisaria el
sobre en silencio. `emitir` levanta ValueError antes de que eso pase.
"""
import time
from contextlib import contextmanager

# Las claves que FormatoJsonCloud (bitacora.py) escribe en el sobre. Un campo con
# uno de estos nombres desaparece al aplanarse: se prohibe en origen.
CLAVES_RESERVADAS = frozenset({"severity", "message", "logger", "time", "evento", "exception"})

# El vocabulario, ejecutable: los oraculos y los tests leen de aca, no de una lista propia.
PASOS = ("download", "classify", "parse", "upload", "vectorize", "fuente", "extract")
ESTADOS_PASO = ("inicio", "fin", "error")
ORIGENES = ("cli", "api")
ESTADOS_LOTE = ("ok", "reintento", "fallo")
MAX_DETALLE = 200


def emitir(log, nombre: str, **campos) -> None:
    """Un evento nombrado: `record.evento` es el nombre y `record.campos` los campos.

    El mensaje del log ES el nombre del evento, asi que un handler sin enterarse de nada
    igual imprime algo legible. Nivel INFO a proposito: los eventos son el rastro normal
    de una corrida, no una anomalia (la anomalia viaja en `estado`).
    """
    reservadas = sorted(set(campos) & CLAVES_RESERVADAS)
    if reservadas:
        raise ValueError(
            f"evento '{nombre}': {reservadas} son claves del sobre JSON y se aplanarian "
            "encima de el; elegi otro nombre de campo")
    log.info(nombre, extra={"evento": nombre, "campos": campos})


class _Paso:
    """Lo que el cuerpo del `with` recibe. Lo que deje en `campos` viaja al evento de cierre."""

    __slots__ = ("campos",)

    def __init__(self):
        self.campos = {}


def _ms(arranque: float) -> int:
    """Milisegundos enteros desde `arranque` (perf_counter: monotono, inmune a la hora del sistema)."""
    return int(round((time.perf_counter() - arranque) * 1000))


def _con_extras(fijos: dict, extras: dict) -> dict:
    """Los fijos primero y GANANDO: un `n` del cuerpo no puede pisar `estado` ni `ms`."""
    salida = dict(fijos)
    for clave, valor in extras.items():
        if clave not in salida:
            salida[clave] = valor
    return salida


@contextmanager
def paso(log, libro_id, paso: str, origen: str):
    """Envuelve un paso del orquestador: emite `ingest_paso` inicio / fin / error.

    Uso:
        with eventos.paso(log, libro_id, "vectorize", origen="cli") as p:
            resultado = ...
            p.campos["n"] = resultado["embebidos"]

    Ante excepcion emite estado="error" con `tipo` y `detalle` y RE-LANZA: este modulo
    observa, no decide. `ms` se mide con perf_counter y va en fin y en error.
    """
    fijos = {"libro_id": libro_id, "paso": paso, "origen": origen}
    emitir(log, "ingest_paso", **fijos, estado="inicio")
    handle = _Paso()
    arranque = time.perf_counter()
    try:
        yield handle
    except BaseException as exc:
        emitir(log, "ingest_paso", **_con_extras(
            {**fijos, "estado": "error", "ms": _ms(arranque),
             "tipo": exc.__class__.__name__, "detalle": str(exc)[:MAX_DETALLE]},
            handle.campos))
        raise
    emitir(log, "ingest_paso", **_con_extras(
        {**fijos, "estado": "fin", "ms": _ms(arranque)}, handle.campos))
