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
"""

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
