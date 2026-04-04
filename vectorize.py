"""Vectoriza chunks y los sube a Neo4j con embeddings para vector search."""

import json
import os
import sys
import time
from db import run_write, run_query
from dotenv import load_dotenv

load_dotenv()

PARSED_DIR = os.path.join(os.path.dirname(__file__), "parsed")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")

# Embedding config
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMS = 3072
GCP_API_KEY = os.getenv("GCP_API_KEY", "")
BATCH_SIZE = 10  # Smaller batches for gemini-embedding-2 (token limits)


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog):
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


def init_embeddings():
    """Inicializa cliente de embeddings (google-genai con API key)."""
    from google import genai
    client = genai.Client(api_key=GCP_API_KEY)
    print(f"  Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMS} dims)")
    return client


def create_vector_index():
    """Crea el indice vectorial en Neo4j si no existe."""
    try:
        run_write(f"""
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMS},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """)
        print(f"  Indice vectorial creado/verificado ({EMBEDDING_DIMS} dims)")
    except Exception as e:
        if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
            print("  Indice vectorial ya existe")
        else:
            print(f"  Advertencia indice: {e}")


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
    """Genera embeddings con gemini-embedding-2-preview via google-genai."""
    truncated = [t[:2000] if len(t) > 2000 else t for t in texts]
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=truncated,
    )
    return [e.values for e in result.embeddings]


def vectorize_libro(libro_id: str, model=None, use_v2: bool = None,
                    on_progress: callable = None) -> dict:
    """Vectoriza chunks de un libro que ya están en Neo4j (les agrega embedding).

    Soporta tanto chunks v1 (text plano) como v2 (con metadata contextual).

    Args:
        libro_id: ID del libro
        model: Modelo de embeddings (se inicializa si no se pasa)
        use_v2: Si True, usa build_embedding_text. Auto-detecta si None.
        on_progress: Callback(step, pct, msg)

    Returns:
        dict con estadísticas: {total, skipped, errors}
    """
    def report(pct, msg):
        if on_progress:
            on_progress("vectorize", pct, msg)
        print(f"  [{pct}%] {msg}")

    # Obtener chunks sin embedding de Neo4j
    chunks = run_query("""
    MATCH (c:Chunk {libro_id: $lid})
    WHERE c.embedding IS NULL
    RETURN c.id AS id, c.text AS text,
           c.titulo_capitulo AS titulo_capitulo,
           c.titulo_seccion AS titulo_seccion,
           c.tipo_contenido AS tipo_contenido
    ORDER BY c.chunk_index
    """, {"lid": libro_id})

    if not chunks:
        report(100, f"{libro_id}: todos los chunks ya tienen embedding")
        return {"total": 0, "skipped": 0, "errors": 0}

    # Auto-detectar v2: si tiene titulo_capitulo, es v2
    if use_v2 is None:
        use_v2 = any(c.get("titulo_capitulo") for c in chunks)

    report(0, f"{libro_id}: vectorizando {len(chunks)} chunks {'(v2 contextual)' if use_v2 else '(v1)'}")

    if model is None:
        model = init_embeddings()

    total = 0
    errors = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        # Preparar textos para embedding
        if use_v2:
            texts = [build_embedding_text(c) for c in batch]
        else:
            texts = [c["text"] for c in batch]

        # Generar embeddings
        try:
            embeddings = generate_embeddings(model, texts)
        except Exception as e:
            print(f"  Error generando embeddings batch {i}: {e}")
            time.sleep(5)
            try:
                embeddings = generate_embeddings(model, texts)
            except Exception as e2:
                print(f"  Error retry: {e2}. Saltando batch.")
                errors += len(batch)
                continue

        # Actualizar embedding en Neo4j (batch update)
        updates = [{"id": c["id"], "embedding": emb}
                    for c, emb in zip(batch, embeddings)]
        try:
            run_write("""
            UNWIND $updates AS u
            MATCH (c:Chunk {id: u.id})
            SET c.embedding = u.embedding
            """, {"updates": updates})
            total += len(batch)
        except Exception as e:
            print(f"  Error subiendo batch {i}: {e}")
            errors += len(batch)

        pct = min(99, int((i + len(batch)) / len(chunks) * 100))
        if (i + BATCH_SIZE) % 100 < BATCH_SIZE:
            report(pct, f"  {i + len(batch)}/{len(chunks)} embeddings")

        # Rate limiting
        if i + BATCH_SIZE < len(chunks):
            time.sleep(1)

    report(100, f"{libro_id}: {total} embeddings generados, {errors} errores")
    return {"total": total, "skipped": 0, "errors": errors}


def vectorize_all():
    """Vectoriza todos los libros parseados."""
    catalog = load_catalog()

    print("Inicializando Google GenAI...")
    model = init_embeddings()

    print("Creando indice vectorial en Neo4j...")
    create_vector_index()

    total = 0
    for libro in catalog["libros"]:
        if libro["estado"] == "parseado":
            print(f"\n{'='*60}")
            print(f"  {libro['titulo']} ({libro['chunks_generados']} chunks)")
            print(f"{'='*60}")
            n = vectorize_libro(libro["id"], model)
            total += n

    # Stats finales
    stats = run_query("MATCH (c:Chunk) RETURN c.libro_id AS libro, count(c) AS chunks ORDER BY chunks DESC")
    print(f"\n{'='*60}")
    print(f"  VECTORIZACION COMPLETA: {total} chunks nuevos")
    print(f"{'='*60}")
    for s in stats:
        print(f"  {s['libro']}: {s['chunks']} chunks en Neo4j")


def search_vector(query: str, top_k: int = 5, libro_id: str = None):
    """Busqueda semantica sobre los chunks vectorizados."""
    model = init_embeddings()
    query_embedding = generate_embeddings(model, [query])[0]

    filter_clause = ""
    params = {"embedding": query_embedding, "top_k": top_k}

    if libro_id:
        filter_clause = "WHERE c.libro_id = $libro_id"
        params["libro_id"] = libro_id

    results = run_query(f"""
    CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
    YIELD node AS c, score
    {filter_clause}
    RETURN c.id AS id, c.libro_id AS libro, c.page_start AS pag_inicio,
           c.page_end AS pag_fin, c.text AS texto, score
    ORDER BY score DESC
    LIMIT $top_k
    """, params)

    return results


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("\nUso:")
        print("  python vectorize.py all              - Vectorizar todos los libros")
        print("  python vectorize.py <libro_id>        - Vectorizar un libro")
        print('  python vectorize.py search "query"    - Busqueda semantica')
        print('  python vectorize.py search "query" --libro farreras-2020')
        sys.exit(0)

    if args[0] == "all":
        vectorize_all()
    elif args[0] == "search":
        query = args[1] if len(args) > 1 else ""
        libro_id = None
        if "--libro" in args:
            idx = args.index("--libro")
            libro_id = args[idx + 1] if idx + 1 < len(args) else None

        if not query:
            print("Error: query vacia")
            sys.exit(1)

        print(f"\n  Busqueda semantica: \"{query}\"")
        results = search_vector(query, top_k=5, libro_id=libro_id)

        if not results:
            print("  Sin resultados")
        else:
            for i, r in enumerate(results):
                print(f"\n  [{i+1}] {r['libro']} - pags {r['pag_inicio']}-{r['pag_fin']} (score: {r['score']:.4f})")
                preview = r['texto'][:300].replace('\n', ' ')
                try:
                    print(f"      {preview}...")
                except UnicodeEncodeError:
                    print(f"      {preview.encode('ascii', errors='replace').decode()}...")
