"""Carga de un libro al grafo: nodos :Chunk / :ParentChunk y sus relaciones. UNA copia.

POR QUE EXISTE (tanda 3, 7-sep-2026). El paso "upload" vivia dos veces y las dos estaban
mal, de formas distintas:
  - CLI (migrate_chunks.upload_chunks_for_libro): CREATE sin borrar lo previo. Re-ingestar
    un libro fallaba por la constraint de id, y el docstring decia "idempotente".
  - API (services/ingest.upload_to_neo4j): DETACH DELETE de todo el libro y CREATE de cero.
    Entre el borrado y la carga el libro no existia, y se perdian las relaciones que
    colgaban de sus chunks (entidades extraidas) y los embeddings ya pagados.

AHORA la carga es DETERMINISTA e IDEMPOTENTE. Los id son deterministas por construccion
(`<libro>_v2_00042`, pipeline/parseo.py), asi que MERGE por id + SET de propiedades deja
el grafo en el mismo estado sin importar cuantas veces se corra:
  - Un chunk cuyo TEXTO cambio pierde `embedding` y `embedding_forma`, para que
    vectorize_missing lo regenere: un vector de un texto viejo es peor que ninguno.
  - Los chunks del libro que ya no existen en el PDF nuevo se borran (huerfanos).
  - Las relaciones derivadas se recomputan: CHILD_OF con MERGE (y se sueltan las de un chunk
    re-parentado), SIGUE_A se borra y se vuelve a tejer por chunk_index.
  - Falla cerrado: sin chunks no se toca el grafo (un parse vacio NO puede vaciar un libro),
    y un chunk con libro_id ajeno aborta antes de escribir.

Sin driver propio: recibe `write(cypher, params)` — db.run_write en el CLI, services.graph.write
en la API. El modulo no sabe de conexiones y se testea con un doble (tests/test_pipeline_carga.py).
"""

BATCH_SIZE = 500

# nombre -> default si el parser no lo trae. None = obligatorio (KeyError si falta).
CAMPOS_CHUNK = {
    "libro_id": None, "text": None, "page_start": None, "page_end": None, "word_count": None,
    "titulo_capitulo": "", "titulo_seccion": "", "tipo_contenido": "body", "parent_id": "",
    "chunk_index": 0, "version": 2, "text_busqueda": "", "titulo_seccion_busqueda": "",
    "titulo_capitulo_busqueda": "", "keywords": "",
}
CAMPOS_PARENT = {
    "libro_id": None, "text": None, "page_start": None, "page_end": None, "word_count": None,
    "titulo_capitulo": "", "titulo_seccion": "",
}

# ON MATCH SET corre ANTES del SET final, con c.text todavia viejo: ahi se decide si el
# embedding sigue valiendo. En un chunk nuevo no hay nada que invalidar.
CYPHER_MERGE_CHUNKS = """
UNWIND $batch AS chunk
MERGE (c:Chunk {id: chunk.id})
ON MATCH SET c.embedding = CASE WHEN c.text <> chunk.props.text THEN null ELSE c.embedding END,
             c.embedding_forma = CASE WHEN c.text <> chunk.props.text THEN null ELSE c.embedding_forma END
SET c += chunk.props
"""
CYPHER_MERGE_PARENTS = """
UNWIND $batch AS parent
MERGE (p:ParentChunk {id: parent.id})
SET p += parent.props
"""
CYPHER_BORRAR_CHUNKS_HUERFANOS = """
MATCH (c:Chunk {libro_id: $lid}) WHERE NOT c.id IN $ids
DETACH DELETE c
"""
CYPHER_BORRAR_PARENTS_HUERFANOS = """
MATCH (p:ParentChunk {libro_id: $lid}) WHERE NOT p.id IN $ids
DETACH DELETE p
"""
CYPHER_SOLTAR_CHILD_OF_VIEJOS = """
MATCH (child:Chunk {libro_id: $lid})-[r:CHILD_OF]->(p:ParentChunk)
WHERE p.id <> child.parent_id
DELETE r
"""
CYPHER_CHILD_OF = """
MATCH (child:Chunk {libro_id: $lid})
WHERE child.parent_id IS NOT NULL AND child.parent_id <> ''
MATCH (parent:ParentChunk {id: child.parent_id})
MERGE (child)-[:CHILD_OF]->(parent)
"""
CYPHER_BORRAR_SIGUE_A = """
MATCH (c1:Chunk {libro_id: $lid})-[r:SIGUE_A]->(:Chunk)
DELETE r
"""
CYPHER_SIGUE_A = """
MATCH (c1:Chunk {libro_id: $lid})
WITH c1 ORDER BY c1.chunk_index
WITH collect(c1) AS chunks
UNWIND range(0, size(chunks) - 2) AS i
WITH chunks[i] AS c1, chunks[i + 1] AS c2
MERGE (c1)-[:SIGUE_A]->(c2)
"""


def preparar(items: list, campos: dict) -> list:
    """[{id, props}] con EXACTAMENTE los campos declarados: defaults para lo ausente,
    KeyError para un obligatorio que falta, y nada mas (ni embedding ni campos raros)."""
    salida = []
    for it in items:
        props = {k: (it[k] if d is None else it.get(k, d)) for k, d in campos.items()}
        salida.append({"id": it["id"], "props": props})
    return salida


def crear_relaciones(write, libro_id: str) -> None:
    """CHILD_OF y SIGUE_A de un libro, recomputadas (son derivadas de parent_id y chunk_index)."""
    p = {"lid": libro_id}
    write(CYPHER_SOLTAR_CHILD_OF_VIEJOS, p)
    write(CYPHER_CHILD_OF, p)
    write(CYPHER_BORRAR_SIGUE_A, p)
    write(CYPHER_SIGUE_A, p)


def cargar_libro(write, libro_id: str, children: list, parents: list, on_progress=None) -> dict:
    """Carga idempotente de un libro. `write(cypher, params)` es la unica puerta al grafo.

    on_progress(step, pct, msg) es opcional (la API lo usa para el job; el CLI imprime).
    Devuelve {"children": n, "parents": m}.
    """
    if not children:
        raise ValueError(f"{libro_id}: sin chunks; no se toca el grafo (un parse vacio no puede vaciar un libro)")
    ajenos = {c.get("libro_id") for c in children} | {p.get("libro_id") for p in parents}
    if ajenos != {libro_id}:
        raise ValueError(f"{libro_id}: chunks con libro_id ajeno {sorted(ajenos - {libro_id})}; no se escribe nada")

    def report(pct, msg):
        if on_progress:
            on_progress("upload", pct, msg)

    ch = preparar(children, CAMPOS_CHUNK)
    pa = preparar(parents, CAMPOS_PARENT)

    report(0, f"Cargando {len(ch)} chunks de {libro_id} (MERGE por id)...")
    for i in range(0, len(ch), BATCH_SIZE):
        write(CYPHER_MERGE_CHUNKS, {"batch": ch[i:i + BATCH_SIZE]})
        report(min(60, int((i + BATCH_SIZE) / len(ch) * 60)), f"  {min(i + BATCH_SIZE, len(ch))}/{len(ch)} chunks")
    report(60, f"Cargando {len(pa)} parents...")
    for i in range(0, len(pa), BATCH_SIZE):
        write(CYPHER_MERGE_PARENTS, {"batch": pa[i:i + BATCH_SIZE]})

    write(CYPHER_BORRAR_CHUNKS_HUERFANOS, {"lid": libro_id, "ids": [c["id"] for c in ch]})
    write(CYPHER_BORRAR_PARENTS_HUERFANOS, {"lid": libro_id, "ids": [p["id"] for p in pa]})
    report(80, "Relaciones CHILD_OF y SIGUE_A...")
    crear_relaciones(write, libro_id)
    report(90, "Carga completa")
    return {"children": len(ch), "parents": len(pa)}


# ---------------------------------------------------------------- fuente (:Book) y borrado
# Tanda B de admin/v1 (7-sep-2026). Hasta hoy la ingesta por API creaba un :Documento paralelo al
# :Book del CLI (backfill_books): dos modelos de "fuente" para la misma cosa, y la consola veia
# 163 libros "legacy" sin metadata. Ahora hay UNA fuente, :Book, y este es el unico lugar que la
# escribe y la borra. La relacion de mencion la pone el paso Extract; se nombra aca para que el
# borrado pueda reportar entidades huerfanas sin adivinar.
RELACION_MENCION = "MENCIONA"

CYPHER_REGISTRAR_FUENTE = """
MERGE (b:Book {id: $lid})
SET b += $props, b.legacy_libro_id = $lid
"""
# CONTAINS en lotes: un libro grande tiene 13k chunks y una sola transaccion se hace pesada.
CYPHER_VINCULAR_FUENTE = """
MATCH (b:Book {id: $lid})
MATCH (c:Chunk {libro_id: $lid})
WHERE NOT (b)-[:CONTAINS]->(c)
WITH b, c LIMIT $lote
MERGE (b)-[:CONTAINS]->(c)
RETURN count(c) AS vinculados
"""
CYPHER_CONTAR_FUENTE = """
MATCH (b:Book {id: $lid})
OPTIONAL MATCH (c:Chunk {libro_id: $lid})
WITH b, count(c) AS chunks, count(c.embedding) AS emb, min(c.page_start) AS p0, max(c.page_end) AS p1
SET b.chunk_count = chunks, b.embedded_count = emb, b.page_start = p0, b.page_end = p1
RETURN chunks, emb
"""

CYPHER_PREVIEW_UNITS = "MATCH (c:Chunk {libro_id: $lid}) RETURN count(c) AS units"
CYPHER_PREVIEW_PARENTS = "MATCH (p:ParentChunk {libro_id: $lid}) RETURN count(p) AS parents"
# Neo4j no admite mezclar una variable agrupada con una agregacion en el RETURN ("implicit
# grouping expressions", visto en vivo 7-sep-2026): las dos cuentas van en WITH separados.
CYPHER_PREVIEW_RELATIONS = """
MATCH (c:Chunk {libro_id: $lid})-[r]-()
WITH count(r) AS de_chunks
OPTIONAL MATCH (p:ParentChunk {libro_id: $lid})-[r2]-(x)
WHERE NOT x:Chunk
WITH de_chunks, count(r2) AS de_parents
RETURN de_chunks + de_parents AS relations
"""
CYPHER_PREVIEW_ORPHANS = f"""
MATCH (c:Chunk {{libro_id: $lid}})-[:{RELACION_MENCION}]->(e)
WHERE NOT EXISTS {{ MATCH (o:Chunk)-[:{RELACION_MENCION}]->(e) WHERE o.libro_id <> $lid }}
RETURN count(DISTINCT e) AS orphans
"""
CYPHER_BORRAR_CHUNKS_LOTE = "MATCH (c:Chunk {libro_id: $lid}) WITH c LIMIT $lote DETACH DELETE c RETURN count(*) AS borrados"
CYPHER_BORRAR_PARENTS_LOTE = "MATCH (p:ParentChunk {libro_id: $lid}) WITH p LIMIT $lote DETACH DELETE p RETURN count(*) AS borrados"
CYPHER_BORRAR_FUENTE = "MATCH (b:Book {id: $lid}) DETACH DELETE b"


def registrar_fuente(query, write, libro_id: str, props: dict, lote: int = 5000) -> dict:
    """MERGE del :Book con sus propiedades, CONTAINS a todos sus chunks (en lotes) y conteos
    frescos (chunk_count, embedded_count, paginas). Idempotente: re-registrar no duplica nada.
    `props` con valor None no se escriben (no pisan lo que ya habia)."""
    write(CYPHER_REGISTRAR_FUENTE, {"lid": libro_id, "props": {k: v for k, v in props.items() if v is not None}})
    while True:
        if query(CYPHER_VINCULAR_FUENTE, {"lid": libro_id, "lote": lote})[0]["vinculados"] == 0:
            break
    r = query(CYPHER_CONTAR_FUENTE, {"lid": libro_id})[0]
    return {"chunk_count": r["chunks"], "embedded_count": r["emb"]}


def borrar_libro(query, write, libro_id: str, dry_run: bool = False, lote: int = 1000) -> dict:
    """Borra una fuente completa (chunks, parents, :Book y sus relaciones) o, con dry_run, solo
    cuenta lo que se borraria. Devuelve el DeletePreview del contrato admin/v1.

    Las entidades que quedan sin ninguna mencion NO se borran (orphan_policy=keep): son
    conocimiento del dominio, no de un libro. Se reportan para que la consola lo muestre.
    Los chunks se borran en lotes de `lote`: un tratado tiene 13k chunks con ~45 relaciones
    cada uno y una sola transaccion se queda sin memoria.
    """
    preview = {
        "source_id": libro_id,
        "dry_run": dry_run,
        "units": query(CYPHER_PREVIEW_UNITS, {"lid": libro_id})[0]["units"],
        "parents": query(CYPHER_PREVIEW_PARENTS, {"lid": libro_id})[0]["parents"],
        "relations": query(CYPHER_PREVIEW_RELATIONS, {"lid": libro_id})[0]["relations"],
        "entities_orphaned": query(CYPHER_PREVIEW_ORPHANS, {"lid": libro_id})[0]["orphans"],
        "orphan_policy": "keep",
        "deleted_at": None,
    }
    if dry_run:
        return preview
    for cypher in (CYPHER_BORRAR_CHUNKS_LOTE, CYPHER_BORRAR_PARENTS_LOTE):
        while query(cypher, {"lid": libro_id, "lote": lote})[0]["borrados"] > 0:
            pass
    write(CYPHER_BORRAR_FUENTE, {"lid": libro_id})
    from datetime import UTC, datetime
    preview["deleted_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return preview
