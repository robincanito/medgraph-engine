"""Parser v2: PDF → chunks semánticos con estructura, overlap y parent-child.

Módulo importable. Funciones principales:
  - parse_libro_v2(libro_id, pdf_path) → (children, parents)
  - detect_structure(pages, libro_id) → structured_pages
  - generate_chunks_v2(structured_pages, libro_id) → (children, parents)

CLI: python parser_v2.py <libro_id> [--all] [--status]
"""

import json
import os
from datetime import datetime

# ── DESDE EL 6-sep-2026 EL CODIGO VIVE EN pipeline/parseo.py (una sola fuente de verdad).
# Este archivo es un SHIM: conserva el catalogo y el CLI, y re-exporta lo demas para que
# `from parser_v2 import parse_pdf_v2` (ingest.py y scripts viejos) siga funcionando.
from pipeline.parseo import (  # noqa: F401
    MAX_CHARS_ENCABEZADO,
    MAX_PARENT_WORDS,
    MAX_PERDIDA_TOKENS,
    MAX_SIZE,
    MIN_COLS_TABLA,
    MIN_FILAS_TABLA,
    MIN_RATIO_LETRAS_ENCABEZADO,
    MIN_SIZE,
    OVERLAP_SIZE,
    PARENT_WINDOW,
    SOLAPE_BLOQUE_TABLA,
    STRUCTURE_PATTERNS,
    TARGET_SIZE,
    _area,
    _celda_limpia,
    _clean_spaced_text,
    _dentro_de_alguna,
    _find_sentence_boundary,
    _fusionar_continuaciones,
    _parece_encabezado,
    _pierde_contenido,
    _tokens,
    classify_content_type,
    clean_text,
    detect_structure,
    extraer_texto_pagina,
    generate_chunks_v2,
    normalize_for_search,
    parse_pdf_v2,
    render_tabla,
)

LIBROS_DIR = os.path.join(os.path.dirname(__file__), "..", "LIBROS")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "parsed")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")

# --- Configuración de chunking ---

# --- Patrones de estructura por libro ---

def load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog):
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


def parse_libro_v2(libro_id: str, pdf_path: str = None,
                   on_progress: callable = None) -> tuple:
    """Parsea un libro por ID o ruta directa. Función principal importable.

    Args:
        libro_id: ID del libro en el catálogo
        pdf_path: Ruta al PDF (opcional, se busca en catálogo si no se da)
        on_progress: Callback(step, pct, msg) para reportar progreso

    Returns:
        (children, parents) — listas de dicts
    """
    def report(pct, msg):
        if on_progress:
            on_progress("parse", pct, msg)
        print(f"  [{pct}%] {msg}")

    # Resolver ruta del PDF
    if not pdf_path:
        catalog = load_catalog()
        libro = next((l for l in catalog["libros"] if l["id"] == libro_id), None)
        if not libro:
            raise ValueError(f"Libro '{libro_id}' no encontrado en catálogo")
        pdf_path = os.path.join(LIBROS_DIR, libro["archivo"])

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

    report(0, f"Iniciando parseo de {libro_id}")

    # Parsear
    children, parents = parse_pdf_v2(pdf_path, libro_id)

    # Guardar JSONs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    children_path = os.path.join(OUTPUT_DIR, f"{libro_id}_v2_chunks.json")
    with open(children_path, "w", encoding="utf-8") as f:
        json.dump(children, f, ensure_ascii=False, indent=2)

    parents_path = os.path.join(OUTPUT_DIR, f"{libro_id}_v2_parents.json")
    with open(parents_path, "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)

    report(100, f"Guardado: {len(children)} children -> {children_path}")

    # Actualizar catálogo
    catalog = load_catalog()
    libro_entry = next((l for l in catalog["libros"] if l["id"] == libro_id), None)
    if libro_entry:
        libro_entry["chunks_v2"] = len(children)
        libro_entry["parents_v2"] = len(parents)
        libro_entry["fecha_parseo_v2"] = datetime.now().isoformat()
        save_catalog(catalog)

    return children, parents


# --- CLI ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\nparser_v2 — Chunking semántico con estructura, overlap y parent-child")
        print("\nUso:")
        print("  python parser_v2.py status              - Ver estado")
        print("  python parser_v2.py <libro_id>           - Parsear un libro")
        print("  python parser_v2.py all                  - Parsear todos los digitales")
        print("  python parser_v2.py <libro_id> --preview - Preview: 10 chunks sin guardar")
        catalog = load_catalog()
        print("\nLibros disponibles:")
        for lib in catalog["libros"]:
            v2 = f" (v2: {lib.get('chunks_v2', 0)} chunks)" if lib.get('chunks_v2') else ""
            print(f"  {lib['id']}: {lib['titulo']} ({lib['estado']}){v2}")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "status":
        catalog = load_catalog()
        print(f"\n{'='*60}")
        print("  ESTADO DE PARSEO V2")
        print(f"{'='*60}")
        for lib in catalog["libros"]:
            v2_chunks = lib.get("chunks_v2", 0)
            v2_parents = lib.get("parents_v2", 0)
            if v2_chunks:
                icon = "[v2]"
                info = f" → {v2_chunks} children, {v2_parents} parents"
            elif lib["estado"] == "parseado":
                icon = "[v1]"
                info = f" → {lib.get('chunks_generados', 0)} chunks v1"
            else:
                icon = "[ ]"
                info = ""
            print(f"  {icon} {lib['titulo']} ({lib['paginas']} págs, {lib['tipo_pdf']}){info}")

    elif cmd == "all":
        catalog = load_catalog()
        for lib in catalog["libros"]:
            if lib["tipo_pdf"] == "digital" and lib["estado"] == "parseado":
                print(f"\n{'='*60}")
                print(f"  {lib['titulo']}")
                print(f"{'='*60}")
                parse_libro_v2(lib["id"])
            elif lib["tipo_pdf"] == "escaneado":
                print(f"\n  SKIP: {lib['titulo']} (escaneado)")

    elif "--preview" in sys.argv:
        libro_id = cmd
        catalog = load_catalog()
        libro = next((l for l in catalog["libros"] if l["id"] == libro_id), None)
        if not libro:
            print(f"Libro '{libro_id}' no encontrado")
            sys.exit(1)
        pdf_path = os.path.join(LIBROS_DIR, libro["archivo"])
        children, parents = parse_pdf_v2(pdf_path, libro_id)

        print(f"\n{'='*60}")
        print(f"  PREVIEW: {len(children)} children, {len(parents)} parents")
        print(f"{'='*60}")

        # Mostrar stats
        sizes = [c["word_count"] for c in children]
        print(f"  Tamaño promedio: {sum(sizes)/len(sizes):.0f} palabras")
        print(f"  Min: {min(sizes)}, Max: {max(sizes)}")

        caps = set(c["titulo_capitulo"] for c in children if c["titulo_capitulo"])
        secs = set(c["titulo_seccion"] for c in children if c["titulo_seccion"])
        print(f"  Capítulos detectados: {len(caps)}")
        print(f"  Secciones detectadas: {len(secs)}")

        # Mostrar 10 chunks del medio (no prólogos)
        start = len(children) // 3
        for c in children[start:start+10]:
            print(f"\n--- [{c['id']}] págs {c['page_start']}-{c['page_end']} ({c['word_count']} words) ---")
            print(f"  Cap: {c['titulo_capitulo'][:60] if c['titulo_capitulo'] else '(no detectado)'}")
            print(f"  Sec: {c['titulo_seccion'][:60] if c['titulo_seccion'] else '(no detectado)'}")
            print(f"  Tipo: {c['tipo_contenido']}")
            print(f"  Parent: {c['parent_id']}")
            preview_text = c['text'][:200].encode('ascii', 'replace').decode('ascii')
            print(f"  Texto: {preview_text}...")

    else:
        parse_libro_v2(cmd)
