"""Migración de chunks v1 -> v2 en Neo4j.

Módulo importable. Funciones principales:
  - upload_chunks_for_libro(libro_id, children, parents) — sube chunks de un libro
  - delete_libro_chunks(libro_id) — borra chunks de un libro
  - normalize_chunks(children) — agrega text_busqueda, keywords, etc.
  - create_relationships(libro_id) — crea CHILD_OF y SIGUE_A
  - full_migration() — migración global v1 -> v2 con swap de labels

CLI: python migrate_chunks.py [upload|swap|verify|clean|full]
"""

import json
import os
import unicodedata
import time
from db import run_write, run_query

PARSED_DIR = os.path.join(os.path.dirname(__file__), "parsed")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")
BATCH_SIZE = 100


def normalize_for_search(text: str) -> str:
    """Quita acentos y pasa a minúsculas para full-text search."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def normalize_chunks(children: list) -> None:
    """Agrega propiedades normalizadas a los chunks (in-place).

    Agrega: text_busqueda, titulo_seccion_busqueda, titulo_capitulo_busqueda, keywords
    """
    for chunk in children:
        chunk["text_busqueda"] = normalize_for_search(chunk["text"])
        chunk["titulo_seccion_busqueda"] = normalize_for_search(chunk.get("titulo_seccion", ""))
        chunk["titulo_capitulo_busqueda"] = normalize_for_search(chunk.get("titulo_capitulo", ""))

        # Keywords: combinar título + primeras 50 palabras
        kw_parts = []
        if chunk.get("titulo_capitulo"):
            kw_parts.append(chunk["titulo_capitulo"])
        if chunk.get("titulo_seccion"):
            kw_parts.append(chunk["titulo_seccion"])
        first_words = " ".join(chunk["text"].split()[:50])
        kw_parts.append(first_words)
        chunk["keywords"] = normalize_for_search(" ".join(kw_parts))


def upload_chunks_for_libro(libro_id: str, children: list, parents: list,
                            label: str = "Chunk",
                            on_progress: callable = None) -> dict:
    """Sube children y parent chunks de un libro a Neo4j.

    Args:
        libro_id: ID del libro
        children: Lista de child chunks
        parents: Lista de parent chunks
        label: Label para children (default "Chunk", usar "ChunkV2" para migración)
        on_progress: Callback(step, pct, msg)

    Returns:
        dict con estadísticas
    """
    def report(pct, msg):
        if on_progress:
            on_progress("upload", pct, msg)
        print(f"  [{pct}%] {msg}")

    total_children = len(children)
    total_parents = len(parents)

    # Subir children en batches
    report(0, f"Subiendo {total_children} children como :{label}...")
    for i in range(0, total_children, BATCH_SIZE):
        batch = children[i:i + BATCH_SIZE]
        _upload_children_batch(batch, label)
        pct = min(60, int((i + len(batch)) / total_children * 60))
        if (i + BATCH_SIZE) % 500 < BATCH_SIZE:
            report(pct, f"  {i + len(batch)}/{total_children} children")

    # Subir parents en batches
    report(60, f"Subiendo {total_parents} parents como :ParentChunk...")
    for i in range(0, total_parents, BATCH_SIZE):
        batch = parents[i:i + BATCH_SIZE]
        _upload_parents_batch(batch)
        pct = 60 + min(30, int((i + len(batch)) / total_parents * 30))
        if (i + BATCH_SIZE) % 500 < BATCH_SIZE:
            report(pct, f"  {i + len(batch)}/{total_parents} parents")

    report(90, "Upload completo")
    return {"children": total_children, "parents": total_parents}


def _upload_children_batch(batch: list, label: str = "Chunk"):
    """Sube un batch de child chunks."""
    # Usar CREATE en vez de MERGE para velocidad (asumimos IDs únicos)
    cypher = f"""
    UNWIND $batch AS chunk
    CREATE (c:{label} {{
        id: chunk.id,
        libro_id: chunk.libro_id,
        text: chunk.text,
        page_start: chunk.page_start,
        page_end: chunk.page_end,
        word_count: chunk.word_count,
        titulo_capitulo: chunk.titulo_capitulo,
        titulo_seccion: chunk.titulo_seccion,
        tipo_contenido: chunk.tipo_contenido,
        parent_id: chunk.parent_id,
        chunk_index: chunk.chunk_index,
        version: chunk.version,
        text_busqueda: chunk.text_busqueda,
        titulo_seccion_busqueda: chunk.titulo_seccion_busqueda,
        titulo_capitulo_busqueda: chunk.titulo_capitulo_busqueda,
        keywords: chunk.keywords
    }})
    """
    # Preparar batch con defaults para propiedades que podrían faltar
    safe_batch = []
    for c in batch:
        safe_batch.append({
            "id": c["id"],
            "libro_id": c["libro_id"],
            "text": c["text"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "word_count": c["word_count"],
            "titulo_capitulo": c.get("titulo_capitulo", ""),
            "titulo_seccion": c.get("titulo_seccion", ""),
            "tipo_contenido": c.get("tipo_contenido", "body"),
            "parent_id": c.get("parent_id", ""),
            "chunk_index": c.get("chunk_index", 0),
            "version": c.get("version", 2),
            "text_busqueda": c.get("text_busqueda", ""),
            "titulo_seccion_busqueda": c.get("titulo_seccion_busqueda", ""),
            "titulo_capitulo_busqueda": c.get("titulo_capitulo_busqueda", ""),
            "keywords": c.get("keywords", ""),
        })
    run_write(cypher, {"batch": safe_batch})


def _upload_parents_batch(batch: list):
    """Sube un batch de parent chunks."""
    run_write("""
    UNWIND $batch AS parent
    CREATE (p:ParentChunk {
        id: parent.id,
        libro_id: parent.libro_id,
        text: parent.text,
        page_start: parent.page_start,
        page_end: parent.page_end,
        word_count: parent.word_count,
        titulo_capitulo: parent.titulo_capitulo,
        titulo_seccion: parent.titulo_seccion
    })
    """, {"batch": [{
        "id": p["id"],
        "libro_id": p["libro_id"],
        "text": p["text"],
        "page_start": p["page_start"],
        "page_end": p["page_end"],
        "word_count": p["word_count"],
        "titulo_capitulo": p.get("titulo_capitulo", ""),
        "titulo_seccion": p.get("titulo_seccion", ""),
    } for p in batch]})


def create_relationships(libro_id: str = None, label: str = "Chunk"):
    """Crea relaciones CHILD_OF y SIGUE_A para chunks de un libro (o todos).

    Args:
        libro_id: Si se da, solo para ese libro. Si None, para todos.
        label: Label de los children (default "Chunk")
    """
    libro_filter = "AND child.libro_id = $libro_id" if libro_id else ""
    params = {"libro_id": libro_id} if libro_id else {}

    # CHILD_OF
    print(f"  Creando relaciones CHILD_OF{f' para {libro_id}' if libro_id else ''}...")
    run_write(f"""
    MATCH (child:{label})
    WHERE child.parent_id IS NOT NULL AND child.parent_id <> '' {libro_filter}
    WITH child
    MATCH (parent:ParentChunk {{id: child.parent_id}})
    CREATE (child)-[:CHILD_OF]->(parent)
    """, params)

    # SIGUE_A (chunks consecutivos del mismo libro)
    print(f"  Creando relaciones SIGUE_A{f' para {libro_id}' if libro_id else ''}...")
    if libro_id:
        run_write(f"""
        MATCH (c1:{label} {{libro_id: $libro_id}})
        WITH c1 ORDER BY c1.chunk_index
        WITH collect(c1) AS chunks
        UNWIND range(0, size(chunks)-2) AS i
        WITH chunks[i] AS c1, chunks[i+1] AS c2
        CREATE (c1)-[:SIGUE_A]->(c2)
        """, {"libro_id": libro_id})
    else:
        # Para todos: agrupar por libro
        libros = run_query(f"MATCH (c:{label}) RETURN DISTINCT c.libro_id AS lid")
        for row in libros:
            lid = row["lid"]
            print(f"    SIGUE_A para {lid}...")
            run_write(f"""
            MATCH (c1:{label} {{libro_id: $libro_id}})
            WITH c1 ORDER BY c1.chunk_index
            WITH collect(c1) AS chunks
            UNWIND range(0, size(chunks)-2) AS i
            WITH chunks[i] AS c1, chunks[i+1] AS c2
            CREATE (c1)-[:SIGUE_A]->(c2)
            """, {"libro_id": lid})


def delete_libro_chunks(libro_id: str):
    """Borra todos los chunks y parents de un libro específico."""
    print(f"  Borrando chunks de {libro_id}...")

    # Borrar children
    run_write("""
    MATCH (c:Chunk {libro_id: $libro_id})
    DETACH DELETE c
    """, {"libro_id": libro_id})

    # Borrar parents
    run_write("""
    MATCH (p:ParentChunk {libro_id: $libro_id})
    DETACH DELETE p
    """, {"libro_id": libro_id})

    print(f"  Borrado completo para {libro_id}")


def verify_counts():
    """Verifica conteo de chunks en Neo4j."""
    results = {}

    # Chunks v1
    v1 = run_query("MATCH (c:Chunk) RETURN count(c) AS n")
    results["chunks"] = v1[0]["n"] if v1 else 0

    # Chunks v2 (si existen durante migración)
    v2 = run_query("MATCH (c:ChunkV2) RETURN count(c) AS n")
    results["chunks_v2"] = v2[0]["n"] if v2 else 0

    # Parents
    parents = run_query("MATCH (p:ParentChunk) RETURN count(p) AS n")
    results["parents"] = parents[0]["n"] if parents else 0

    # ChunkV1 (backup durante swap)
    v1_old = run_query("MATCH (c:ChunkV1) RETURN count(c) AS n")
    results["chunks_v1_backup"] = v1_old[0]["n"] if v1_old else 0

    # Relaciones
    child_of = run_query("MATCH ()-[r:CHILD_OF]->() RETURN count(r) AS n")
    results["child_of_rels"] = child_of[0]["n"] if child_of else 0

    sigue_a = run_query("MATCH ()-[r:SIGUE_A]->() RETURN count(r) AS n")
    results["sigue_a_rels"] = sigue_a[0]["n"] if sigue_a else 0

    # Por libro
    by_libro = run_query("MATCH (c:Chunk) RETURN c.libro_id AS libro, count(c) AS n ORDER BY n DESC")
    results["by_libro"] = {r["libro"]: r["n"] for r in by_libro}

    return results


def full_migration():
    """Migración completa v1 -> v2 con swap de labels.

    Pasos:
    1. Cargar y normalizar todos los chunks v2 de parsed/
    2. Subir como :ChunkV2 + :ParentChunk
    3. Verificar conteo
    4. Swap: Chunk -> ChunkV1, ChunkV2 -> Chunk
    5. Crear relaciones
    6. Limpiar ChunkV1
    """
    print("=" * 60)
    print("  MIGRACIÓN COMPLETA v1 -> v2")
    print("=" * 60)

    catalog = json.load(open(CATALOG_PATH, "r", encoding="utf-8"))

    # Paso 1: Cargar todos los chunks v2
    all_children = []
    all_parents = []

    for libro in catalog["libros"]:
        children_path = os.path.join(PARSED_DIR, f"{libro['id']}_v2_chunks.json")
        parents_path = os.path.join(PARSED_DIR, f"{libro['id']}_v2_parents.json")

        if not os.path.exists(children_path):
            print(f"  SKIP: {libro['titulo']} (no tiene v2 chunks)")
            continue

        with open(children_path, "r", encoding="utf-8") as f:
            children = json.load(f)
        with open(parents_path, "r", encoding="utf-8") as f:
            parents = json.load(f)

        # Normalizar para full-text
        normalize_chunks(children)

        all_children.extend(children)
        all_parents.extend(parents)
        print(f"  Cargado: {libro['titulo']} — {len(children)} children, {len(parents)} parents")

    print(f"\n  Total a subir: {len(all_children)} children, {len(all_parents)} parents")

    # Paso 2: Subir como ChunkV2
    print("\n  Paso 2: Subiendo chunks v2...")
    upload_chunks_for_libro("_all_", all_children, all_parents, label="ChunkV2")

    # Paso 3: Verificar conteo
    print("\n  Paso 3: Verificando...")
    counts = verify_counts()
    print(f"    Chunks v1 actuales: {counts['chunks']}")
    print(f"    Chunks v2 nuevos: {counts['chunks_v2']}")
    print(f"    Parents nuevos: {counts['parents']}")

    if counts['chunks_v2'] == 0:
        print("  ERROR: No se subieron chunks v2. Abortando.")
        return

    # Paso 4: Swap de labels
    print("\n  Paso 4: Swap de labels...")
    print("    Chunk -> ChunkV1...")
    run_write("MATCH (c:Chunk) SET c:ChunkV1 REMOVE c:Chunk")
    time.sleep(1)

    print("    ChunkV2 -> Chunk...")
    run_write("MATCH (c:ChunkV2) SET c:Chunk REMOVE c:ChunkV2")
    time.sleep(1)

    # Paso 5: Crear relaciones
    print("\n  Paso 5: Creando relaciones...")
    create_relationships(label="Chunk")

    # Paso 6: Verificar
    print("\n  Paso 6: Verificación final...")
    counts = verify_counts()
    print(f"    Chunks (nuevos): {counts['chunks']}")
    print(f"    Parents: {counts['parents']}")
    print(f"    CHILD_OF: {counts['child_of_rels']}")
    print(f"    SIGUE_A: {counts['sigue_a_rels']}")
    print(f"    ChunkV1 (backup): {counts['chunks_v1_backup']}")
    print(f"    Por libro: {counts['by_libro']}")

    print("\n  Migración completada. Los chunks v1 están en :ChunkV1 como backup.")
    print("  Para limpiar: python migrate_chunks.py clean")


def clean_v1():
    """Borra los chunks v1 (backup post-swap)."""
    counts = verify_counts()
    v1_count = counts['chunks_v1_backup']

    if v1_count == 0:
        print("  No hay chunks v1 para limpiar.")
        return

    print(f"  Borrando {v1_count} chunks v1 (backup)...")
    run_write("MATCH (c:ChunkV1) DETACH DELETE c")
    print("  Limpieza completada.")


# --- CLI ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\nmigrate_chunks — Migración de chunks v1 -> v2 en Neo4j")
        print("\nUso:")
        print("  python migrate_chunks.py verify   - Ver estado actual")
        print("  python migrate_chunks.py full     - Migración completa v1 -> v2")
        print("  python migrate_chunks.py clean    - Borrar backup v1")
        print("  python migrate_chunks.py libro <id> - Subir un libro específico (reemplaza)")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "verify":
        counts = verify_counts()
        print(f"\n{'='*60}")
        print(f"  ESTADO DE NEO4J")
        print(f"{'='*60}")
        for k, v in counts.items():
            if k != "by_libro":
                print(f"  {k}: {v}")
        if counts.get("by_libro"):
            print(f"\n  Por libro:")
            for libro, n in counts["by_libro"].items():
                print(f"    {libro}: {n}")

    elif cmd == "full":
        full_migration()

    elif cmd == "clean":
        clean_v1()

    elif cmd == "libro" and len(sys.argv) > 2:
        libro_id = sys.argv[2]
        children_path = os.path.join(PARSED_DIR, f"{libro_id}_v2_chunks.json")
        parents_path = os.path.join(PARSED_DIR, f"{libro_id}_v2_parents.json")

        if not os.path.exists(children_path):
            print(f"No se encontraron chunks v2 para '{libro_id}'")
            print(f"Primero correr: python parser_v2.py {libro_id}")
            sys.exit(1)

        with open(children_path, "r", encoding="utf-8") as f:
            children = json.load(f)
        with open(parents_path, "r", encoding="utf-8") as f:
            parents = json.load(f)

        normalize_chunks(children)
        delete_libro_chunks(libro_id)
        upload_chunks_for_libro(libro_id, children, parents)
        create_relationships(libro_id)

        counts = verify_counts()
        print(f"\n  Verificación: {counts['by_libro'].get(libro_id, 0)} chunks para {libro_id}")

    else:
        print(f"Comando desconocido: {cmd}")
