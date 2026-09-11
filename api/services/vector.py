"""Servicio de busqueda: semantica, full-text y hybrid con RRF."""

import os
from dotenv import load_dotenv
from services.graph import query

load_dotenv()

_client = None

# Pool size para retrieval inicial (se fusionan despues)
RETRIEVAL_POOL = 60
# Constante k para Reciprocal Rank Fusion
RRF_K = 60
# Embedding config
EMBEDDING_MODEL = "gemini-embedding-2"
GCP_API_KEY = os.getenv("GCP_API_KEY", "")


def get_embedding_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=GCP_API_KEY)
    return _client


def generate_embedding(text: str) -> list:
    client = get_embedding_client()
    truncated = text[:2000] if len(text) > 2000 else text
    r = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=truncated,
    )
    return r.embeddings[0].values


def search_semantic(query_text: str, top_k: int = 40, libro_id: str = None) -> list:
    """Búsqueda semántica con vector KNN sobre embeddings."""
    embedding = generate_embedding(query_text)

    if libro_id:
        results = query("""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
        YIELD node AS c, score
        WHERE c.libro_id = $libro_id
        RETURN c.id AS id, c.libro_id AS libro, c.page_start AS pag_inicio,
               c.page_end AS pag_fin, c.text AS texto, c.word_count AS palabras,
               c.titulo_capitulo AS capitulo, c.titulo_seccion AS seccion,
               c.tipo_contenido AS tipo, c.parent_id AS parent_id, score
        ORDER BY score DESC
        """, {"embedding": embedding, "top_k": top_k, "libro_id": libro_id})
    else:
        results = query("""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
        YIELD node AS c, score
        RETURN c.id AS id, c.libro_id AS libro, c.page_start AS pag_inicio,
               c.page_end AS pag_fin, c.text AS texto, c.word_count AS palabras,
               c.titulo_capitulo AS capitulo, c.titulo_seccion AS seccion,
               c.tipo_contenido AS tipo, c.parent_id AS parent_id, score
        ORDER BY score DESC
        """, {"embedding": embedding, "top_k": top_k})

    return results


# La sintaxis del QueryParser clásico de Lucene: ninguno de estos caracteres puede llegar crudo
# desde una consulta de usuario.
_SINTAXIS_LUCENE = str.maketrans({c: " " for c in '+-&|!(){}[]^"~*?:\\/'})


def _terminos_lucene(texto: str) -> str:
    """Consulta de usuario -> términos planos, sin sintaxis que Lucene reinterprete.

    BUG 2026-09-10, encontrado al medir el retrieval con un conjunto de evaluación real: 37 de 101
    consultas traían caracteres de sintaxis Lucene y el texto iba crudo al parser. Dos modos de
    falla, y el segundo es el peligroso:

      · `/` y `[ ]` -> excepción del parser. `search_hybrid` la atrapa, descarta la mitad keyword
        y sigue con semántica sola.
      · `:` -> NO da error: Lucene lo lee como `campo:término`. "sindrome: condensacion pulmonar"
        pasa a ser "buscá 'condensacion' en el campo 'sindrome'", campo que no existe, así que el
        término se pierde y queda sólo "pulmonar". Devuelve resultados plausibles de las fuentes
        equivocadas sin una sola señal de que algo salió mal.

    Se reemplaza por espacio en vez de escapar con `\\` porque el analizador del índice
    (`standard-no-stop-words`, tokenizer UAX#29) ya descarta la puntuación AL INDEXAR: los tokens
    guardados nunca tienen `:` ni `-`. Partir en espacios es exactamente esa tokenización --
    "anti-inflamatorios" -> "anti inflamatorios", el par que el índice realmente contiene.
    """
    return " ".join(texto.translate(_SINTAXIS_LUCENE).split())


def search_keyword(query_text: str, top_k: int = 40, libro_id: str = None) -> list:
    """Búsqueda full-text con scoring BM25 sobre Chunk.text (Lucene via Neo4j)."""
    if not query_text.strip():
        return []

    # Lucene query: OR entre términos para mayor recall,
    # el scoring BM25 se encarga de rankear mejor los que tienen más matches
    lucene_query = _terminos_lucene(query_text)  # OR implícito en Lucene
    if not lucene_query:
        # La consulta era pura sintaxis. Mandarla vacía hace explotar el parser.
        return []

    params = {"query": lucene_query, "top_k": top_k}

    if libro_id:
        results = query("""
        CALL db.index.fulltext.queryNodes('busqueda_chunks', $query)
        YIELD node AS c, score
        WHERE c.libro_id = $libro_id
        RETURN c.id AS id, c.libro_id AS libro, c.page_start AS pag_inicio,
               c.page_end AS pag_fin, c.text AS texto, c.word_count AS palabras,
               c.titulo_capitulo AS capitulo, c.titulo_seccion AS seccion,
               c.tipo_contenido AS tipo, c.parent_id AS parent_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """, params)
    else:
        results = query("""
        CALL db.index.fulltext.queryNodes('busqueda_chunks', $query)
        YIELD node AS c, score
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
    from concurrent.futures import ThreadPoolExecutor
    import logging

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

        try:
            sem = sem_future.result()
            if sem:
                all_rankings.append(sem)
                total_semantic += len(sem)
        except Exception as e:
            logging.error(f"Semantic search failed: {type(e).__name__}: {str(e)[:100]}")

    if not all_rankings:
        return {
            "keyword_count": 0,
            "semantic_count": 0,
            "intencion": processed["intencion"],
            "query_expandida": processed["expandida"],
            "results": [],
        }

    # 3. RRF fusion de todos los rankings
    fused = _rrf_fusion(all_rankings)

    # 4. Top-k final
    results = fused[:top_k]

    return {
        "keyword_count": total_keyword,
        "semantic_count": total_semantic,
        "intencion": processed["intencion"],
        "query_expandida": processed["expandida"],
        "results": results,
    }
