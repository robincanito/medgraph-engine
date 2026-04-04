"""Sube todos los chunks a Neo4j SIN embeddings. Rápido, para que keyword search funcione."""

import json
import os
from db import run_write, run_query

PARSED_DIR = os.path.join(os.path.dirname(__file__), "parsed")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")


def upload_all():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    total = 0

    for libro in catalog["libros"]:
        if libro["estado"] != "parseado":
            continue

        chunks_path = os.path.join(PARSED_DIR, f"{libro['id']}_chunks.json")
        if not os.path.exists(chunks_path):
            continue

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Check how many already exist
        existing = run_query(
            "MATCH (c:Chunk {libro_id: $lid}) RETURN count(c) AS n",
            {"lid": libro["id"]}
        )
        existing_count = existing[0]["n"] if existing else 0

        if existing_count >= len(chunks):
            print(f"  {libro['titulo']}: {existing_count} chunks ya en Neo4j (skip)")
            continue

        print(f"  {libro['titulo']}: subiendo {len(chunks)} chunks ({existing_count} ya existen)...")

        batch = []
        for chunk in chunks:
            batch.append({
                "id": chunk["id"],
                "libro_id": chunk["libro_id"],
                "text": chunk["text"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "word_count": chunk["word_count"]
            })

            if len(batch) >= 100:
                _upload_batch(batch)
                total += len(batch)
                print(f"    {total} chunks subidos...")
                batch = []

        if batch:
            _upload_batch(batch)
            total += len(batch)

        print(f"    {libro['titulo']} completo")

    print(f"\nTotal: {total} chunks subidos a Neo4j")


def _upload_batch(batch):
    """Sube un batch de chunks con UNWIND para eficiencia."""
    run_write("""
    UNWIND $batch AS chunk
    MERGE (c:Chunk {id: chunk.id})
    SET c.libro_id = chunk.libro_id,
        c.text = chunk.text,
        c.page_start = chunk.page_start,
        c.page_end = chunk.page_end,
        c.word_count = chunk.word_count
    """, {"batch": batch})


if __name__ == "__main__":
    upload_all()
