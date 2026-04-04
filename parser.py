"""Parser de PDFs a texto limpio + chunks semánticos."""

import fitz  # PyMuPDF
import json
import os
import re
from datetime import datetime

LIBROS_DIR = os.path.join(os.path.dirname(__file__), "..", "LIBROS")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "parsed")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog):
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


def parse_pdf(pdf_path: str, libro_id: str, chunk_size: int = 800) -> dict:
    """Parsea un PDF a texto limpio con chunks semánticos.

    Args:
        pdf_path: Ruta al PDF
        libro_id: ID del libro en el catálogo
        chunk_size: Tamaño objetivo de cada chunk en tokens (~palabras)

    Returns:
        dict con páginas y chunks
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    print(f"  Parseando {os.path.basename(pdf_path)} ({total_pages} pags)...")

    # Extraer texto por página
    pages = []
    for i in range(total_pages):
        page = doc[i]
        text = page.get_text()
        text = clean_text(text)

        if len(text.strip()) > 20:  # Ignorar páginas vacías/solo imagen
            pages.append({
                "page": i + 1,
                "text": text
            })

        if (i + 1) % 500 == 0:
            print(f"    ... {i + 1}/{total_pages} paginas")

    doc.close()

    # Generar chunks semánticos
    chunks = generate_chunks(pages, libro_id, chunk_size)

    result = {
        "libro_id": libro_id,
        "archivo": os.path.basename(pdf_path),
        "total_paginas": total_pages,
        "paginas_con_texto": len(pages),
        "chunks": len(chunks),
        "fecha_parseo": datetime.now().isoformat(),
        "pages_data": pages,
        "chunks_data": chunks
    }

    print(f"  Resultado: {len(pages)} pags con texto, {len(chunks)} chunks")
    return result


def clean_text(text: str) -> str:
    """Limpia texto extraído de PDF."""
    # Remover saltos de línea excesivos
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remover líneas que solo son números de página
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    # Remover headers/footers repetitivos comunes
    text = re.sub(r'(?i)^.*medicina interna.*edici[oó]n.*$', '', text, flags=re.MULTILINE)
    # Normalizar espacios
    text = re.sub(r'[ \t]+', ' ', text)
    # Trim líneas
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def generate_chunks(pages: list, libro_id: str, target_size: int = 800) -> list:
    """Genera chunks semánticos a partir de páginas.

    Intenta cortar en límites de sección/párrafo, no a mitad de oración.
    """
    chunks = []
    current_chunk = ""
    current_start_page = None

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        if current_start_page is None:
            current_start_page = page_num

        paragraphs = text.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 10:
                continue

            word_count = len(current_chunk.split())
            para_words = len(para.split())

            # Si agregar este párrafo excede el target, guardar chunk actual
            if word_count + para_words > target_size and word_count > 100:
                chunks.append({
                    "id": f"{libro_id}_chunk_{len(chunks):05d}",
                    "libro_id": libro_id,
                    "page_start": current_start_page,
                    "page_end": page_num,
                    "text": current_chunk.strip(),
                    "word_count": word_count
                })
                current_chunk = para + "\n\n"
                current_start_page = page_num
            else:
                current_chunk += para + "\n\n"

    # Último chunk
    if current_chunk.strip() and len(current_chunk.split()) > 50:
        chunks.append({
            "id": f"{libro_id}_chunk_{len(chunks):05d}",
            "libro_id": libro_id,
            "page_start": current_start_page,
            "page_end": pages[-1]["page"] if pages else 0,
            "text": current_chunk.strip(),
            "word_count": len(current_chunk.split())
        })

    return chunks


def parse_libro(libro_id: str):
    """Parsea un libro por su ID del catálogo."""
    catalog = load_catalog()

    libro = None
    for lib in catalog["libros"]:
        if lib["id"] == libro_id:
            libro = lib
            break

    if not libro:
        print(f"Libro '{libro_id}' no encontrado en catalogo.")
        return

    if libro["tipo_pdf"] == "escaneado":
        print(f"'{libro['titulo']}' es un PDF escaneado. Requiere OCR. Saltando.")
        return

    if libro["estado"] == "parseado":
        print(f"'{libro['titulo']}' ya fue parseado. Usa --force para re-parsear.")
        return

    pdf_path = os.path.join(LIBROS_DIR, libro["archivo"])
    if not os.path.exists(pdf_path):
        print(f"Archivo no encontrado: {pdf_path}")
        return

    # Parsear
    result = parse_pdf(pdf_path, libro_id)

    # Guardar resultado (sin pages_data para ahorrar espacio, solo chunks)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Guardar chunks
    chunks_path = os.path.join(OUTPUT_DIR, f"{libro_id}_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(result["chunks_data"], f, ensure_ascii=False, indent=2)

    # Guardar metadata
    meta_path = os.path.join(OUTPUT_DIR, f"{libro_id}_meta.json")
    meta = {k: v for k, v in result.items() if k not in ("pages_data", "chunks_data")}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Actualizar catálogo
    libro["estado"] = "parseado"
    libro["fecha_parseo"] = result["fecha_parseo"]
    libro["chunks_generados"] = result["chunks"]
    save_catalog(catalog)

    print(f"  Guardado en: {chunks_path}")
    print(f"  Catalogo actualizado.")


def parse_all():
    """Parsea todos los libros digitales pendientes."""
    catalog = load_catalog()

    for libro in catalog["libros"]:
        if libro["tipo_pdf"] == "digital" and libro["estado"] == "pendiente":
            print(f"\n{'='*60}")
            print(f"  {libro['titulo']}")
            print(f"{'='*60}")
            parse_libro(libro["id"])
        elif libro["tipo_pdf"] == "escaneado":
            print(f"\n  SKIP: {libro['titulo']} (escaneado, requiere OCR)")
        elif libro["estado"] == "parseado":
            print(f"\n  SKIP: {libro['titulo']} (ya parseado)")


def status():
    """Muestra estado del parseo."""
    catalog = load_catalog()

    print(f"\n{'='*60}")
    print(f"  ESTADO DE PARSEO")
    print(f"{'='*60}")

    for libro in catalog["libros"]:
        status_icon = {
            "pendiente": "[ ]",
            "parseado": "[x]",
            "requiere_ocr": "[!]"
        }.get(libro["estado"], "[?]")

        chunks_info = f" ({libro['chunks_generados']} chunks)" if libro["chunks_generados"] else ""
        print(f"  {status_icon} {libro['titulo']} - {libro['paginas']} pags - {libro['tipo_pdf']}{chunks_info}")

    if catalog.get("libros_faltantes"):
        print(f"\n  Libros no disponibles aun:")
        for lib in catalog["libros_faltantes"]:
            print(f"  [-] {lib}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\nUso:")
        print("  python parser.py status          - Ver estado")
        print("  python parser.py all             - Parsear todos los pendientes")
        print("  python parser.py <libro_id>      - Parsear un libro especifico")
        print("\nLibros disponibles:")
        catalog = load_catalog()
        for lib in catalog["libros"]:
            print(f"  {lib['id']}: {lib['titulo']} ({lib['estado']})")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "status":
        status()
    elif cmd == "all":
        parse_all()
    elif cmd == "--force" and len(sys.argv) > 2:
        libro_id = sys.argv[2]
        catalog = load_catalog()
        for lib in catalog["libros"]:
            if lib["id"] == libro_id:
                lib["estado"] = "pendiente"
        save_catalog(catalog)
        parse_libro(libro_id)
    else:
        parse_libro(cmd)
