"""Servicio de ingesta de libros: PDF -> parse -> upload -> vectorize.

Pipeline simplificado para correr dentro de Cloud Run.
Para libros grandes (>1000 pags), preferir correr desde CLI local.
"""

import os
import re
import json
import time
import unicodedata
import fitz  # PyMuPDF
from services.graph import query, write

# Chunking config
TARGET_SIZE = 280
MIN_SIZE = 150
MAX_SIZE = 380
OVERLAP_SIZE = 60
PARENT_WINDOW = 3
MAX_PARENT_WORDS = 1200
BATCH_SIZE = 100

# Jobs in memory (single instance is fine for this use case)
_jobs = {}


def get_job(job_id: str) -> dict:
    return _jobs.get(job_id)


def normalize_for_search(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^.*booksmedicos\.org.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def classify_content_type(text: str) -> str:
    lines = text.strip().split('\n')
    if not lines:
        return "body"
    tab_lines = sum(1 for l in lines if '|' in l or '\t' in l)
    if tab_lines > len(lines) * 0.3 and tab_lines >= 3:
        return "tabla"
    list_lines = sum(1 for l in lines if re.match(r'^\s*[-*]\s', l) or re.match(r'^\s*\d+[\.\)]\s', l))
    if list_lines > len(lines) * 0.3 and list_lines >= 3:
        return "lista"
    first_100 = text[:200].lower()
    if any(p in first_100 for p in ['concepto', 'definicion', 'se define como']):
        return "definicion"
    return "body"


def _find_sentence_boundary(words, target_idx):
    search_start = max(0, target_idx - 30)
    search_end = min(len(words), target_idx + 30)
    best = target_idx
    best_dist = 999
    for i in range(search_start, search_end):
        word = words[i]
        if word.endswith(('.', '?', '!', ':')) and not re.match(r'^\d+\.$', word):
            dist = abs(i - target_idx)
            if dist < best_dist:
                best = i
                best_dist = dist
    return best + 1


def parse_and_chunk(pdf_path: str, libro_id: str) -> tuple:
    """Parse PDF and generate v2 chunks. Returns (children, parents)."""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(doc.page_count):
        text = clean_text(doc[i].get_text())
        if len(text.strip()) > 20:
            pages.append({"page": i + 1, "text": text})
    doc.close()

    # Simple structure detection (generic patterns)
    cap_pats = [re.compile(p, re.MULTILINE) for p in [
        r'^SECCION\s+[IVXLCDM]+', r'^Capitulo\s+\d+', r'^PARTE\s+\d+',
    ]]
    sec_pats = [re.compile(p, re.MULTILINE) for p in [
        r'^\d+\.\d+[\s\.]+[A-Z]', r'^[A-Z][A-Z\s]{5,60}$',
    ]]

    current_cap = ""
    current_sec = ""
    structured = []
    for p in pages:
        for line in p["text"].split('\n'):
            ls = line.strip()
            if not ls or len(ls) < 3:
                continue
            is_cap = any(pat.match(ls) for pat in cap_pats)
            if is_cap:
                current_cap = ls[:120]
                current_sec = ""
            elif any(pat.match(ls) for pat in sec_pats) and len(ls) < 80:
                current_sec = ls[:100]
        structured.append({**p, "titulo_capitulo": current_cap, "titulo_seccion": current_sec})

    # Generate chunks
    all_words = []
    word_meta = []
    for sp in structured:
        for w in sp["text"].split():
            all_words.append(w)
            word_meta.append((sp["page"], sp["titulo_capitulo"], sp["titulo_seccion"]))

    if not all_words:
        return [], []

    children = []
    pos = 0
    chunk_index = 0
    while pos < len(all_words):
        end_target = pos + TARGET_SIZE
        if end_target >= len(all_words):
            end = len(all_words)
        else:
            end = _find_sentence_boundary(all_words, end_target)
            if end - pos > MAX_SIZE:
                end = pos + MAX_SIZE

        if end - pos < MIN_SIZE and children:
            prev = children[-1]
            prev["text"] = prev["text"] + " " + " ".join(all_words[pos:end])
            prev["word_count"] = len(prev["text"].split())
            prev["page_end"] = word_meta[end - 1][0]
            break

        chunk_text = " ".join(all_words[pos:end])
        capitulo = word_meta[pos][1]
        seccion = word_meta[pos][2]
        for i in range(pos, min(end, pos + 50)):
            if word_meta[i][1]:
                capitulo = word_meta[i][1]
            if word_meta[i][2]:
                seccion = word_meta[i][2]

        text_busqueda = normalize_for_search(chunk_text)
        kw_parts = [capitulo, seccion, " ".join(chunk_text.split()[:50])]
        keywords = normalize_for_search(" ".join(p for p in kw_parts if p))

        children.append({
            "id": f"{libro_id}_v2_{chunk_index:05d}",
            "libro_id": libro_id,
            "page_start": word_meta[pos][0],
            "page_end": word_meta[end - 1][0],
            "text": chunk_text,
            "word_count": end - pos,
            "titulo_capitulo": capitulo,
            "titulo_seccion": seccion,
            "tipo_contenido": classify_content_type(chunk_text),
            "parent_id": "",
            "chunk_index": chunk_index,
            "version": 2,
            "text_busqueda": text_busqueda,
            "titulo_seccion_busqueda": normalize_for_search(seccion),
            "titulo_capitulo_busqueda": normalize_for_search(capitulo),
            "keywords": keywords,
        })
        chunk_index += 1
        next_pos = end - OVERLAP_SIZE
        if next_pos <= pos:
            next_pos = end
        pos = next_pos

    # Parents
    parents = []
    for i in range(0, len(children), PARENT_WINDOW):
        window = children[i:i + PARENT_WINDOW]
        parent_text = " ".join(c["text"] for c in window)
        if len(parent_text.split()) > MAX_PARENT_WORDS:
            parent_text = " ".join(parent_text.split()[:MAX_PARENT_WORDS])

        parent_id = f"{libro_id}_v2_parent_{len(parents):05d}"
        parents.append({
            "id": parent_id,
            "libro_id": libro_id,
            "page_start": window[0]["page_start"],
            "page_end": window[-1]["page_end"],
            "text": parent_text,
            "word_count": len(parent_text.split()),
            "titulo_capitulo": window[0]["titulo_capitulo"],
            "titulo_seccion": window[0]["titulo_seccion"],
        })
        for c in window:
            c["parent_id"] = parent_id

    return children, parents


def upload_to_neo4j(libro_id: str, children: list, parents: list):
    """Upload chunks and parents to Neo4j, replacing existing ones for this libro."""
    # Delete existing
    write("MATCH (c:Chunk {libro_id: $lid}) DETACH DELETE c", {"lid": libro_id})
    write("MATCH (p:ParentChunk {libro_id: $lid}) DETACH DELETE p", {"lid": libro_id})

    # Upload children in batches
    for i in range(0, len(children), BATCH_SIZE):
        batch = children[i:i + BATCH_SIZE]
        write("""
        UNWIND $batch AS c
        CREATE (n:Chunk {
            id: c.id, libro_id: c.libro_id, text: c.text,
            page_start: c.page_start, page_end: c.page_end,
            word_count: c.word_count, titulo_capitulo: c.titulo_capitulo,
            titulo_seccion: c.titulo_seccion, tipo_contenido: c.tipo_contenido,
            parent_id: c.parent_id, chunk_index: c.chunk_index, version: c.version,
            text_busqueda: c.text_busqueda, titulo_seccion_busqueda: c.titulo_seccion_busqueda,
            titulo_capitulo_busqueda: c.titulo_capitulo_busqueda, keywords: c.keywords
        })
        """, {"batch": batch})

    # Upload parents
    for i in range(0, len(parents), BATCH_SIZE):
        batch = parents[i:i + BATCH_SIZE]
        write("""
        UNWIND $batch AS p
        CREATE (n:ParentChunk {
            id: p.id, libro_id: p.libro_id, text: p.text,
            page_start: p.page_start, page_end: p.page_end,
            word_count: p.word_count, titulo_capitulo: p.titulo_capitulo,
            titulo_seccion: p.titulo_seccion
        })
        """, {"batch": batch})

    # Create CHILD_OF relationships
    write("""
    MATCH (child:Chunk {libro_id: $lid})
    WHERE child.parent_id IS NOT NULL AND child.parent_id <> ''
    WITH child
    MATCH (parent:ParentChunk {id: child.parent_id})
    CREATE (child)-[:CHILD_OF]->(parent)
    """, {"lid": libro_id})

    # Create SIGUE_A relationships
    write("""
    MATCH (c1:Chunk {libro_id: $lid})
    WITH c1 ORDER BY c1.chunk_index
    WITH collect(c1) AS chunks
    UNWIND range(0, size(chunks)-2) AS i
    WITH chunks[i] AS c1, chunks[i+1] AS c2
    CREATE (c1)-[:SIGUE_A]->(c2)
    """, {"lid": libro_id})


def vectorize_chunks(libro_id: str):
    """Generate embeddings for chunks that don't have them yet."""
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GCP_API_KEY", ""))
    except Exception as e:
        print(f"GenAI init error: {e}")
        return 0

    chunks = query("""
    MATCH (c:Chunk {libro_id: $lid})
    WHERE c.embedding IS NULL
    RETURN c.id AS id, c.text AS text,
           c.titulo_capitulo AS titulo_capitulo,
           c.titulo_seccion AS titulo_seccion,
           c.tipo_contenido AS tipo_contenido
    ORDER BY c.chunk_index
    """, {"lid": libro_id})

    if not chunks:
        return 0

    total = 0
    embed_batch = 20

    for i in range(0, len(chunks), embed_batch):
        batch = chunks[i:i + embed_batch]

        # Build contextual text for embedding
        texts = []
        for c in batch:
            parts = []
            if c.get("titulo_capitulo"):
                parts.append(f"Capitulo: {c['titulo_capitulo']}")
            if c.get("titulo_seccion"):
                parts.append(f"Seccion: {c['titulo_seccion']}")
            prefix = ". ".join(parts)
            text = f"{prefix}. {c['text']}" if prefix else c["text"]
            texts.append(text[:2000])

        try:
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=texts,
            )
            updates = [{"id": c["id"], "embedding": e.values}
                       for c, e in zip(batch, result.embeddings)]
            write("""
            UNWIND $updates AS u
            MATCH (c:Chunk {id: u.id})
            SET c.embedding = u.embedding
            """, {"updates": updates})
            total += len(batch)
        except Exception as e:
            print(f"Embedding error batch {i}: {e}")
            time.sleep(5)

        if i + embed_batch < len(chunks):
            time.sleep(1)

    return total


def run_ingest(libro_id: str, pdf_path: str, titulo: str, autor: str):
    """Full ingest pipeline. Updates _jobs dict with progress."""
    job = {"status": "running", "progress": [], "result": None}
    _jobs[libro_id] = job

    def report(step, pct, msg):
        job["progress"].append({"step": step, "pct": pct, "msg": msg})

    try:
        # Parse
        report("parse", 0, f"Parseando {titulo}...")
        children, parents = parse_and_chunk(pdf_path, libro_id)
        report("parse", 100, f"Parseado: {len(children)} chunks, {len(parents)} parents")

        # Upload
        report("upload", 0, "Subiendo a Neo4j...")
        upload_to_neo4j(libro_id, children, parents)
        report("upload", 100, f"Subido: {len(children)} chunks a Neo4j")

        # Vectorize
        report("vectorize", 0, "Generando embeddings...")
        n_emb = vectorize_chunks(libro_id)
        report("vectorize", 100, f"Vectorizado: {n_emb} embeddings")

        job["status"] = "completed"
        job["result"] = {
            "libro_id": libro_id,
            "children": len(children),
            "parents": len(parents),
            "embeddings": n_emb,
        }

    except Exception as e:
        job["status"] = "error"
        job["result"] = {"error": str(e)}

    finally:
        # Cleanup temp file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
