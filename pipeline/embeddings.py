"""Embeddings canonicos de MedGraph: el modelo, el texto que se embebe y la llamada.

UNA SOLA FUENTE DE VERDAD (Fase 2, tanda 2, 6-sep-2026). Hasta hoy:
  - `EMBEDDING_MODEL = "gemini-embedding-2"` estaba escrito en CUATRO archivos.
  - El prefijo contextual que se embebe ("Capitulo: X. Seccion: Y. Tipo: Z. <texto>")
    existia dos veces: en vectorize.py CON tildes y con Tipo, y en
    api/services/ingest.py SIN tildes y sin Tipo. Todo libro vectorizado por la API
    quedo en un espacio vectorial ligeramente corrido respecto del corpus. El CLI
    habia arreglado su copia el 3-ago; la de la API nadie la vio, porque era copia.
  - La llamada a Vertex vivia tres veces. Vertex acepta UN contenido por llamada
    (verificado en vivo el 6-sep: "only supports one content at a time"); la API
    sobrevivia solo porque EMBED_BATCH_SIZE valia 1.

Quien lo usa: vectorize.py (CLI, shim), ingest.py (CLI), api/services/ingest.py
(ingesta por API) y api/services/vector.py (embedding de la consulta).
Codigo de las funciones: el de vectorize.py del 6-sep, movido verbatim por AST.
`google.genai` se importa perezoso: la API no debe pagarlo en el arranque.

TANDA 1 DE LA QA (13-sep-2026): tambien EL BUCLE es uno solo. Lo que seguia duplicado era la
politica de vectorizacion: `ingest.py::step_vectorize` reintentaba 3 veces con espera 5/10 s y
truncaba a 2.000 caracteres, y `api/services/ingest.py::vectorize_chunks` no reintentaba nada,
dormia 3 s y no truncaba. El MISMO documento terminaba con distinto numero de embeddings segun
por donde entro (agujero G3 del diseno). `vectorizar_faltantes` es la unica politica: la del CLI,
que es la que Ivan aprobo, mas el conteo explicito de los chunks que quedaron SIN vector, que
antes nadie reportaba.
"""
import logging
import time

from pipeline import eventos

MAX_CHARS = 2000  # techo de caracteres por texto antes de embeber (ver generate_embeddings)

EMBEDDING_MODEL = "gemini-embedding-2"


EMBEDDING_DIMS = 3072
# Tanda 2b/3 (6 y 7-sep-2026): todo vector que se guarda lleva Chunk.embedding_forma con el
# nombre de la forma de texto con que se embebio. "canonico" = build_embedding_text. La
# auditoria (audit_embedding_prefix.py) lee esa marca en vez de re-embeber muestras, y el
# 40% del corpus que estuvo mezclado durante semanas se hubiera detectado en el primer run.
EMBEDDING_FORMA = "canonico"
# La UNICA sentencia que guarda embeddings. Los cuatro writers (CLI vectorize/ingest, API
# ingest, reembed) la importan: si alguien vuelve a escribir "SET c.embedding" a mano sin
# la forma, tests/test_pipeline_embeddings.py lo frena.
CYPHER_GUARDAR_EMBEDDINGS = f"""
UNWIND $updates AS u
MATCH (c:Chunk {{id: u.id}})
SET c.embedding = u.embedding, c.embedding_forma = '{EMBEDDING_FORMA}'
"""

# La UNICA consulta de "que falta vectorizar". Estaba copiada letra por letra en
# ingest.py::step_vectorize y en api/services/ingest.py::vectorize_chunks. El ORDER BY
# importa: hace la corrida reproducible (los lotes siempre se arman igual) y hace que un
# corte a mitad deje vectorizado un prefijo del libro, no un salpicado.
CYPHER_CHUNKS_SIN_EMBEDDING = """
MATCH (c:Chunk {libro_id: $lid})
WHERE c.embedding IS NULL
RETURN c.id AS id, c.text AS text,
       c.titulo_capitulo AS titulo_capitulo, c.titulo_seccion AS titulo_seccion,
       c.tipo_contenido AS tipo_contenido
ORDER BY c.chunk_index
"""

# Chunks por lote de ESCRITURA y de reintento (Vertex recibe 1 texto por llamada igual, ver
# generate_embeddings). 10 y no los 20 que tenia el CLI: el lote es la unidad que se pierde
# cuando un reintento agota, asi que mas chico es menos trabajo pago tirado. La API ya usaba 10.
EMBED_BATCH = 10
# Mismo techo que generate_embeddings, con nombre propio porque aca es un parametro.
MAX_CHARS_EMBED = MAX_CHARS


def crear_cliente(project: str, location: str):
    """Cliente google-genai contra Vertex AI. gemini-embedding-2 vive en `global`,
    no en us-central1 (404 "Publisher model ... was not found"). Import perezoso."""
    from google import genai

    return genai.Client(vertexai=True, project=project, location=location)


def build_embedding_text(chunk: dict) -> str:
    """Prepend metadata al texto para embedding contextual (v2).

    Esto mejora la calidad del embedding porque reduce ambigüedad semántica.
    Ej: "Capítulo: Nefrología. Sección: Inmunodepresores. <texto>"
    """
    parts = []
    if chunk.get("titulo_capitulo"):
        parts.append(f"Capítulo: {chunk['titulo_capitulo']}")
    if chunk.get("titulo_seccion"):
        parts.append(f"Sección: {chunk['titulo_seccion']}")
    if chunk.get("tipo_contenido") and chunk["tipo_contenido"] != "body":
        parts.append(f"Tipo: {chunk['tipo_contenido']}")
    prefix = ". ".join(parts)
    return f"{prefix}. {chunk['text']}" if prefix else chunk["text"]


def generate_embeddings(client, texts: list) -> list:
    """Genera embeddings con gemini-embedding-2 via google-genai.

    NOTA 2026-06-16: cuando gemini-embedding-2 pasó a GA, el formato
    `contents=[str, str, ...]` empezó a devolver solo 1 embedding por call.
    El formato correcto para batch es `contents=[{"parts": [{"text": t}]}, ...]`.
    Con strings simples solo funciona para 1 input a la vez.
    """
    truncated = [t[:2000] if len(t) > 2000 else t for t in texts]

    # UNO POR LLAMADA, a proposito (18-ago-2026).
    # Al migrar de la API key de AI Studio a Vertex AI aparecio una diferencia
    # de contrato entre los dos backends con el MISMO modelo:
    #     "The embedContent API for this model only supports one content at a time."
    # AI Studio aceptaba el lote completo; Vertex lo rechaza. Como la unica
    # fabrica de cliente ahora apunta a Vertex (ver init_vertex), el batch
    # murio y hay que iterar.
    #
    # Costo: una llamada HTTP por chunk en vez de una por lote. Para material
    # de catedra (decenas de chunks) es irrelevante; para un libro de miles
    # se nota, y ahi conviene paralelizar con un pool de workers antes que
    # volver al batch.
    #
    # La verificacion de que sigue habiendo 1 vector por texto (que es lo que
    # el bug de junio rompio en silencio) la hace el llamador comparando
    # len(vectores) == len(batch).
    vectores = []
    for t in truncated:
        r = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[{"parts": [{"text": t}]}],
        )
        vectores.append(r.embeddings[0].values)
    return vectores


def _ms(arranque: float) -> int:
    """Milisegundos enteros desde `arranque` (perf_counter, monotono)."""
    return int(round((time.perf_counter() - arranque) * 1000))


def vectorizar_faltantes(query, write, client, libro_id, on_progress=None, *, log=None,
                         lote=EMBED_BATCH, reintentos=3, esperas=(5, 10), pausa=0.3,
                         max_chars=MAX_CHARS_EMBED, dormir=time.sleep) -> dict:
    """Embebe los chunks de `libro_id` que no tienen vector. UNA politica para CLI y API.

    LA POLITICA (la del CLI, aprobada el 13-sep-2026 para los dos caminos):
      - hasta `reintentos` intentos por lote, con espera creciente entre intentos
        (`esperas`; la ultima se repite si faltan). Lo que reintenta es el LOTE, no el chunk:
        un 429 de Vertex no distingue.
      - cada texto truncado a `max_chars`. La API no truncaba: un chunk de 12.000 caracteres
        se embebia distinto segun por donde entro el documento.
      - `pausa` entre lotes, para no gatillar la cuota.
      - si vuelven menos vectores que textos NO SE GUARDA NADA del lote (el bug de junio-2026:
        `contents=[str, str]` devolvia un vector mezcla y zip() truncaba en silencio).
      - un lote que agota los reintentos NO corta la corrida: los demas siguen y los chunks
        del lote perdido se CUENTAN en `sin_vector`. Antes nadie los contaba y un libro
        quedaba a medias sin que ninguna metrica lo dijera.

    Argumentos:
      query / write   `(cypher, params)` — db.run_query/run_write en el CLI, services.graph
                      query/write en la API. Este modulo no sabe de conexiones.
      client          cliente de Vertex (crear_cliente).
      on_progress     `(step, pct, msg)` opcional, igual que pipeline.carga.
      log             logger del llamador, para que el evento salga con SU nombre de modulo.
      dormir          inyectable: los tests no esperan 15 segundos de verdad.

    Devuelve {"embebidos", "sin_vector", "lotes_fallidos", "llamadas"}. `llamadas` es el eje
    economico: cuantas veces se llamo a generate_embeddings, reintentos incluidos.
    """
    log = log if log is not None else logging.getLogger(__name__)
    chunks = query(CYPHER_CHUNKS_SIN_EMBEDDING, {"lid": libro_id})
    if not chunks:
        return {"embebidos": 0, "sin_vector": 0, "lotes_fallidos": 0, "llamadas": 0}

    embebidos = sin_vector = lotes_fallidos = llamadas = 0
    for i in range(0, len(chunks), lote):
        batch = chunks[i:i + lote]
        textos = [build_embedding_text(c)[:max_chars] for c in batch]

        for intento in range(1, reintentos + 1):
            arranque = time.perf_counter()
            try:
                llamadas += 1
                vectores = generate_embeddings(client, textos)
                if len(vectores) != len(batch):
                    raise RuntimeError(
                        f"la API devolvio {len(vectores)} embeddings para {len(batch)} "
                        f"chunks; no se guarda nada para no desalinear")
                updates = [{"id": c["id"], "embedding": v}
                           for c, v in zip(batch, vectores, strict=True)]
                write(CYPHER_GUARDAR_EMBEDDINGS, {"updates": updates})
            except Exception as e:
                ultimo = intento >= reintentos
                eventos.emitir(log, "embed_lote", libro_id=libro_id, n_textos=len(batch),
                               intento=intento, estado="fallo" if ultimo else "reintento",
                               ms=_ms(arranque), detalle=str(e)[:eventos.MAX_DETALLE])
                if ultimo:
                    lotes_fallidos += 1
                    sin_vector += len(batch)
                    log.error(f"lote {i} sin vector despues de {reintentos} intentos: {str(e)[:80]}")
                    break
                espera = esperas[min(intento - 1, len(esperas) - 1)] if esperas else 0
                log.warning(f"reintento {intento}/{reintentos} del lote {i} en {espera}s: {str(e)[:60]}")
                dormir(espera)
            else:
                embebidos += len(batch)
                eventos.emitir(log, "embed_lote", libro_id=libro_id, n_textos=len(batch),
                               intento=intento, estado="ok", ms=_ms(arranque))
                break

        if on_progress:
            pct = int((i + len(batch)) / len(chunks) * 100)
            on_progress("vectorize", pct, f"{embebidos}/{len(chunks)} embeddings")
        if i + lote < len(chunks):
            dormir(pausa)

    if sin_vector:
        log.warning(f"{libro_id}: {embebidos} embeddings, {sin_vector} chunks SIN vector "
                    f"({lotes_fallidos} lotes fallidos, {llamadas} llamadas)")
    return {"embebidos": embebidos, "sin_vector": sin_vector,
            "lotes_fallidos": lotes_fallidos, "llamadas": llamadas}
