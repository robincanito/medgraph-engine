"""Servicio de busqueda: semantica, full-text y hybrid con RRF."""


from dotenv import load_dotenv

from pipeline.parseo import NO_CONTENIDO
from services.graph import query
from services.settings import get_settings

load_dotenv()

_client = None

# Pool size para retrieval inicial (se fusionan despues).
#
# 60 -> 150 el 10-sep-2026, MEDIDO con eval/eval_retrieval.py sobre 101 consultas reales (corridas
# `base2` vs `pool150` en eval/resultados/). Cuatro indicadores apuntaron al mismo lado, y el que
# no es ruido es el tercero:
#   · consultas SIN un solo pasaje relevante en el pool: 3 -> 1 de 101
#   · concentracion (precision@top_k / precision@pool): 1.33 -> 1.45
#   · 17 consultas mejoran contra 9 que empeoran (75 sin cambio)
#   · contaminacion por `cie10` en el top-10: 7 -> 5 consultas
# Costo: +54 ms en la mediana y +60 ms en p95 (~6%), plano en todos los percentiles.
# `precision@top_k` sube apenas +0.009: eso SOLO es ruido, no es el argumento.
RETRIEVAL_POOL = 150

# Constante k para Reciprocal Rank Fusion.
#
# SE PROBO BAJARLA A 20 el 10-sep-2026 Y SALIO PEOR: 8 consultas mejoran contra 19 que empeoran,
# concentracion 1.33 -> 1.27, y la contaminacion por `cie10` subio de 7 a 9 consultas (corrida
# `rrf20`). Queda en 60 con la hipotesis refutada anotada, para no volver a gastar la medicion.
#
# POR QUE FALLO, que es lo util: se sospechaba que k=60 con un pool de 60 "aplana" el ranking --el
# puesto 1 pesa apenas el doble que el 60-- y que bajarla haria discriminar mas. Pasa lo contrario.
# Con k alto lo que decide es la MULTIPLICIDAD (aparecer en varios de los rankings que se fusionan),
# que es justamente la señal robusta. Bajar k le sube el peso al puesto 1 de CADA ranking por
# separado, o sea que amplifica el error de cada uno --incluido el de BM25, que pone primero al
# indice de codigos CIE-10-- en vez de corregirlo.
RRF_K = 60
# Embedding config

# VERTEX AI, no API key (migrado 18-ago-2026).
# Esta es la ruta de CONSULTA: cada busqueda semantica embebe el texto de la
# pregunta. Cuando el saldo prepago de AI Studio llego a cero, el 429 se
# convirtio en un 500 y tumbo /search/semantic, /search/hybrid y /query a la
# vez — todos dependen de este vector. El /health seguia devolviendo 200,
# porque no toca embeddings: el servicio parecia sano y no lo estaba.
#
# OJO CON LA LOCATION: gemini-embedding-2 no existe en us-central1 (404).
# Vive en `global`.
_s = get_settings()
GCP_PROJECT = _s.gcp_project
GCP_LOCATION = _s.gcp_location


def get_embedding_client():
    global _client
    if _client is None:
        # Cliente unico via el pipeline canonico (Vertex, location global). Ver
        # pipeline/embeddings.py: gemini-embedding-2 NO existe en us-central1.
        from pipeline import embeddings
        _client = embeddings.crear_cliente(GCP_PROJECT, GCP_LOCATION)
    return _client


def generate_embedding(text: str) -> list:
    """Embedding de la consulta: misma llamada y mismo modelo que la ingesta.

    Delegado a pipeline.embeddings (6-sep-2026): un contenido por llamada, dict
    por texto, techo de 2000 chars — el contrato que rompio en junio y el unico
    que Vertex acepta, escrito una sola vez.
    """
    from pipeline import embeddings

    return embeddings.generate_embeddings(get_embedding_client(), [text])[0]


def search_semantic(query_text: str, top_k: int = 40, libro_id: str = None) -> list:
    """Búsqueda semántica con vector KNN sobre embeddings."""
    embedding = generate_embedding(query_text)

    if libro_id:
        # POST-FILTRO, no pre-filtro: el indice vectorial de Neo4j no sabe filtrar por
        # propiedad, asi que trae los K vecinos GLOBALES y despues se descarta lo que
        # no es del libro. Con $top_k chico eso devuelve casi siempre VACIO: si ninguno
        # de los 60 vecinos globales es de ese libro, el filtro deja 0.
        # Detectado 2026-07-30 al implementar el routing por fuente sugerida: la pasada
        # restringida a meneghello no aportaba nada y solo trabajaba la pata keyword.
        # Fix: pedir un pool mucho mas grande antes de filtrar.
        pool = min(max(top_k * 60, 3000), 20000)
        results = query("""
        CALL db.index.vector.queryNodes('chunk_embeddings', $pool, $embedding)
        YIELD node AS c, score
        WHERE c.libro_id = $libro_id AND NOT coalesce(c.tipo_contenido, 'body') IN $excluidos
        RETURN c.id AS id, c.libro_id AS libro, c.page_start AS pag_inicio,
               c.page_end AS pag_fin, c.text AS texto, c.word_count AS palabras,
               c.titulo_capitulo AS capitulo, c.titulo_seccion AS seccion,
               c.tipo_contenido AS tipo, c.parent_id AS parent_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """, {"embedding": embedding, "top_k": top_k, "libro_id": libro_id, "pool": pool,
              "excluidos": list(NO_CONTENIDO)})
    else:
        # LO QUE NO ES CONTENIDO SE FILTRA DESPUES del indice (13-sep-2026): el indice vectorial
        # devuelve los K vecinos y no sabe filtrar por propiedad. Si se pidieran exactamente top_k y
        # despues se sacaran las referencias y el indice CIE-10, quedarian menos de top_k. Se piden de
        # mas -el no-contenido es ~3% del corpus, pero en una consulta puntual llego a ser 4 de 10- y
        # se corta despues. Mismo principio que el post-filtro por libro de arriba, con menos margen.
        results = query("""
        CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
        YIELD node AS c, score
        WHERE NOT coalesce(c.tipo_contenido, 'body') IN $excluidos
        RETURN c.id AS id, c.libro_id AS libro, c.page_start AS pag_inicio,
               c.page_end AS pag_fin, c.text AS texto, c.word_count AS palabras,
               c.titulo_capitulo AS capitulo, c.titulo_seccion AS seccion,
               c.tipo_contenido AS tipo, c.parent_id AS parent_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """, {"embedding": embedding, "top_k": top_k, "k": top_k + max(50, top_k // 2),
              "excluidos": list(NO_CONTENIDO)})

    return results


def _normalize(text: str) -> str:
    """Sin acentos, lowercase — los campos *_busqueda del índice v2 están normalizados así."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# La sintaxis del QueryParser clásico de Lucene. Ninguno de estos caracteres puede llegar crudo
# desde una consulta de usuario (ver `_terminos_lucene`).
_SINTAXIS_LUCENE = str.maketrans({c: " " for c in '+-&|!(){}[]^"~*?:\\/'})


def _terminos_lucene(texto: str) -> str:
    """Consulta de usuario -> términos planos para Lucene, sin sintaxis que la reinterprete.

    BUG 2026-09-10, encontrado por eval/eval_retrieval.py: **37 de las 101 consultas del conjunto
    de evaluación traían caracteres de sintaxis Lucene**, y el texto iba crudo al parser. Dos modos
    de falla, y el segundo es el peligroso:

      · `/` y `[ ]` -> excepción del parser. `search_hybrid` la atrapa, tira la mitad keyword y
        sigue con semántica sola. Al menos queda registrado (`fallas` -> `degradado`).
      · `:` -> NO da error: Lucene lo lee como `campo:término`. "sindrome: condensacion pulmonar"
        se convierte en "buscá 'condensacion' en el campo 'sindrome'" -- campo que no existe, así
        que ese término se pierde y sólo queda "pulmonar". La búsqueda devuelve resultados
        plausibles de los libros equivocados (Harrison/Farreras en vez de Argente) sin una sola
        señal de que algo salió mal. Eso es peor que el error: mentira silenciosa.

    POR QUÉ REEMPLAZAR POR ESPACIO Y NO ESCAPAR CON `\\`: el índice usa el analizador
    `standard-no-stop-words`, cuyo tokenizer (UAX#29) ya descarta la puntuación AL INDEXAR -- los
    tokens guardados nunca tienen `:` ni `/` ni `-`. Entonces partir en espacios es exactamente la
    tokenización del índice: "anti-inflamatorios" -> "anti inflamatorios", que es el par de tokens
    que el índice realmente contiene. Escapar dejaría que el analizador los borre igual, pero
    agrega los problemas del `\\` (final de cadena, doble escape) sin ganar nada.
    """
    return " ".join(texto.translate(_SINTAXIS_LUCENE).split())


def search_keyword(query_text: str, top_k: int = 40, libro_id: str = None) -> list:
    """Búsqueda full-text con scoring BM25 sobre los campos normalizados del Chunk (Lucene via Neo4j)."""
    if not query_text.strip():
        return []

    # Lucene query: OR entre términos para mayor recall,
    # el scoring BM25 se encarga de rankear mejor los que tienen más matches.
    # Normalizada porque el índice v2 indexa text_busqueda (sin acentos, lowercase).
    lucene_query = _terminos_lucene(_normalize(query_text))  # OR implícito en Lucene
    if not lucene_query:
        # La consulta era pura sintaxis ("///", "?!"). No hay nada que buscar, y mandarla vacía
        # hace explotar el parser.
        return []

    params = {"query": lucene_query, "top_k": top_k}
    # BUG 2026-07-30: el Cypher de abajo usa $libro_id pero nunca se agregaba a params
    # -> ParameterMissing, error tragado por search_hybrid, y el filtro por libro
    # degradaba en silencio a semantica-sola. Afectaba a medgraph_search(libro_id=...)
    # y al routing por fuente sugerida.
    if libro_id:
        params["libro_id"] = libro_id
    params["excluidos"] = list(NO_CONTENIDO)

    if libro_id:
        results = query("""
        CALL db.index.fulltext.queryNodes('busqueda_chunks_v2', $query)
        YIELD node AS c, score
        WHERE c.libro_id = $libro_id AND NOT coalesce(c.tipo_contenido, 'body') IN $excluidos
        RETURN c.id AS id, c.libro_id AS libro, c.page_start AS pag_inicio,
               c.page_end AS pag_fin, c.text AS texto, c.word_count AS palabras,
               c.titulo_capitulo AS capitulo, c.titulo_seccion AS seccion,
               c.tipo_contenido AS tipo, c.parent_id AS parent_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """, params)
    else:
        results = query("""
        CALL db.index.fulltext.queryNodes('busqueda_chunks_v2', $query)
        YIELD node AS c, score
        WHERE NOT coalesce(c.tipo_contenido, 'body') IN $excluidos
        RETURN c.id AS id, c.libro_id AS libro, c.page_start AS pag_inicio,
               c.page_end AS pag_fin, c.text AS texto, c.word_count AS palabras,
               c.titulo_capitulo AS capitulo, c.titulo_seccion AS seccion,
               c.tipo_contenido AS tipo, c.parent_id AS parent_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """, params)

    return results


def _rrf_fusion(rankings: list[list], k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion: combina múltiples rankings en uno solo.

    Para cada resultado en cada ranking, calcula score = 1/(k + rank).
    Suma los scores de todos los rankings donde aparece cada chunk.
    Chunks que aparecen en múltiples rankings suben al top.
    """
    scores = {}
    chunk_data = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking):
            chunk_id = result["id"]
            rrf_score = 1.0 / (k + rank + 1)

            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                chunk_data[chunk_id] = result
            scores[chunk_id] += rrf_score

    # Ordenar por score RRF combinado
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for chunk_id in sorted_ids:
        result = chunk_data[chunk_id]
        result["rrf_score"] = round(scores[chunk_id], 6)
        results.append(result)

    return results


def search_hybrid(query_text: str, top_k: int = 8, libro_id: str = None) -> dict:
    """Búsqueda híbrida con query rewriting + full-text + semántica + RRF.

    1. Query preprocessing (expansión sinónimos, clasificación intención, decomposición)
    2. Para cada sub-query: full-text top POOL + dense top POOL
    3. RRF fusion de todos los rankings
    4. Devuelve top_k resultados finales
    """
    from services.query import preprocess

    # 1. Preprocess query
    processed = preprocess(query_text)
    sub_queries = processed["sub_queries"]

    all_rankings = []
    total_keyword = 0
    total_semantic = 0

    # 2. Multi-retrieval EN PARALELO: N keyword (una por sub-query) + 1 semantica.
    # Antes corrian en secuencia (hasta 4 viajes seriales a Neo4j/Gemini).
    import logging
    from concurrent.futures import ThreadPoolExecutor

    from services import telemetria

    # Las dos rutas de busqueda se tragan sus errores para degradar en vez de romper. Bien, pero
    # sin registrarlo, una caida del grafo devuelve `results: []` — indistinguible de "no hay
    # nada sobre esto". El cliente entonces contesta de su propio conocimiento SIN AVISAR que la
    # bibliografia no se consulto, que en una herramienta cuya promesa es "cita libro y pagina"
    # es un fallo de correccion, no de disponibilidad. Se anotan para poder distinguirlas.
    fallas: list = []

    with ThreadPoolExecutor(max_workers=len(sub_queries) + 1) as pool:
        kw_futures = [(sq, pool.submit(search_keyword, sq, RETRIEVAL_POOL, libro_id))
                      for sq in sub_queries]
        sem_future = pool.submit(search_semantic, processed["expandida"], RETRIEVAL_POOL, libro_id)

        for sq, fut in kw_futures:
            try:
                kw = fut.result()
                if kw:
                    all_rankings.append(kw)
                    total_keyword += len(kw)
            except Exception as e:
                logging.error(f"Keyword search failed for '{sq}': {type(e).__name__}: {str(e)[:100]}")
                fallas.append(("keyword", type(e).__name__))

        try:
            sem = sem_future.result()
            if sem:
                all_rankings.append(sem)
                total_semantic += len(sem)
        except Exception as e:
            logging.error(f"Semantic search failed: {type(e).__name__}: {str(e)[:100]}")
            fallas.append(("semantic", type(e).__name__))

    if not all_rankings:
        if fallas:
            telemetria.registrar("mcp_degradado", tool="search_hybrid", capa="retrieval",
                                 rutas_caidas=[f"{r}:{e}" for r, e in fallas],
                                 detalle="sin resultados PORQUE fallo el retrieval, no porque no haya material")
        return {
            "keyword_count": 0,
            "semantic_count": 0,
            "intencion": processed["intencion"],
            "query_expandida": processed["expandida"],
            "results": [],
            # SUBE CON LA RESPUESTA, no se queda solo en el log (10-sep-2026): quien pregunta
            # tiene derecho a saber que el corpus no se pudo consultar. Sin esto, una lista vacia
            # por caida se lee igual que "no hay material".
            "degradado": "total" if fallas else None,
        }

    if fallas:
        # Degradacion PARCIAL: una ruta respondio y la otra no. Hay resultados, pero peores que
        # los normales, y sin esto nadie se entera nunca.
        telemetria.registrar("mcp_degradado", tool="search_hybrid", capa="retrieval_parcial",
                             rutas_caidas=[f"{r}:{e}" for r, e in fallas])

    # 3. RRF fusion de todos los rankings
    fused = _rrf_fusion(all_rankings)

    # 4. Top-k final
    results = fused[:top_k]

    # CUANTO MATERIAL SE ESTA TIRANDO (paso 0 de MEDGRAPH_RERANKER_PLAN.md, 10-sep-2026).
    # Se recuperan hasta RETRIEVAL_POOL por ranking y se cortan `top_k`: hoy eso es descartar
    # ~92-96% de lo traido sin volver a mirarlo. Un reranker solo sirve si lo relevante ESTA en el
    # pool pero mal rankeado; si ya viene en el top_k, no hay nada que ganar. Estos tres numeros
    # son lo que permite decidirlo sin comprometerse a nada.
    #
    # `rrf_top` va porque `RRF_K = 60` con un pool de 60 aplana el ranking -- el puesto 1 pesa
    # 1/61 y el 60 pesa 1/120, o sea que el primero vale apenas el doble que el ultimo-- y lo que
    # termina decidiendo es la MULTIPLICIDAD. Ver la distribucion es lo que confirma o refuta eso.
    telemetria.registrar(
        "retrieval",
        n_sub_queries=len(sub_queries),
        n_rankings=len(all_rankings),
        n_candidatos_unicos=len(fused),
        top_k=top_k,
        descartados=max(0, len(fused) - len(results)),
        rrf_top=[round(r.get("rrf_score", 0), 5) for r in results[:5]],
        rrf_k=RRF_K,
        pool=RETRIEVAL_POOL,
    )

    return {
        "keyword_count": total_keyword,
        "semantic_count": total_semantic,
        "intencion": processed["intencion"],
        "query_expandida": processed["expandida"],
        "results": results,
        # "parcial": hay resultados, pero una de las dos rutas de busqueda no contesto, asi que
        # son peores que los normales. Vale decirlo aunque no este vacio.
        "degradado": "parcial" if fallas else None,
    }
