"""Parser v2: PDF → chunks semánticos con estructura, overlap y parent-child.

Módulo importable. Funciones principales:
  - parse_libro_v2(libro_id, pdf_path) → (children, parents)
  - detect_structure(pages, libro_id) → structured_pages
  - generate_chunks_v2(structured_pages, libro_id) → (children, parents)

CLI: python parser_v2.py <libro_id> [--all] [--status]
"""

import fitz  # PyMuPDF
import json
import os
import re
import unicodedata
from datetime import datetime

LIBROS_DIR = os.path.join(os.path.dirname(__file__), "..", "LIBROS")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "parsed")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")

# --- Configuración de chunking ---
TARGET_SIZE = 280       # palabras objetivo por child chunk
MIN_SIZE = 150          # mínimo aceptable
MAX_SIZE = 380          # máximo antes de forzar corte
OVERLAP_SIZE = 60       # palabras de overlap entre chunks
PARENT_WINDOW = 3       # cantidad de children por parent chunk
MAX_PARENT_WORDS = 1200 # máximo de palabras por parent

# --- Patrones de estructura por libro ---

STRUCTURE_PATTERNS = {
    "farreras-2020": {
        "capitulo": [
            r'^SECCIÓN\s+[IVXLCDM]+\b',
            r'^Capítulo\s+\d+',
            r'^CAPÍTULO\s+\d+',
        ],
        "seccion": [
            r'^\d+\.\d+[\s\.]+[A-ZÁÉÍÓÚÑ]',    # 23.4 Glaucoma...
            r'^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,60}$', # MODELOS DE REGRESIÓN (línea sola en mayúsculas)
        ],
    },
    "garcia-feijoo-2012": {
        "capitulo": [
            r'PA\s*R\s*T\s*E\s+\d+',             # PA R T E 1 : BÁSICO
            r'^PARTE\s+\d+',
        ],
        "seccion": [
            r'^\d+\s*\|\s*.+',                    # 1 | Embriología. Desarrollo...
            r'^\d+\.\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{3,}',  # 1. Embriología... (requiere palabra real, no "3. Mixto.")
        ],
    },
    "diamante-orl": {
        "capitulo": [
            r'^SECCIÓN\s+[IVXLCDM]+',
            r'^Sección\s+[IVXLCDM]+',
            r'^CAPÍTULO\s+\d+',
        ],
        "seccion": [
            r'^\d+\.\s+[A-ZÁÉÍÓÚÑ]',
            r'^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,50}$',
        ],
    },
    "_default": {
        "capitulo": [
            r'^SECCIÓN\s+[IVXLCDM]+',
            r'^CAPÍTULO\s+\d+',
            r'^Capítulo\s+\d+',
            r'^PARTE\s+\d+',
        ],
        "seccion": [
            r'^\d+\.\d+[\s\.]+[A-ZÁÉÍÓÚÑ]',
            r'^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,60}$',
        ],
    },
}


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog):
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


def normalize_for_search(text: str) -> str:
    """Quita acentos y pasa a minúsculas para full-text search."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def clean_text(text: str) -> str:
    """Limpia texto extraído de PDF."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^.*medicina interna.*edici[oó]n.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^.*booksmedicos\.org.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^.*© Elsevier\. Fotocopiar sin autorización es un delito\..*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^.*© \d{4}.*Elsevier.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def classify_content_type(text: str) -> str:
    """Clasifica el tipo de contenido de un chunk."""
    lines = text.strip().split('\n')
    if not lines:
        return "body"

    # Tablas: muchas líneas con | o tabulaciones
    tab_lines = sum(1 for l in lines if '|' in l or '\t' in l)
    if tab_lines > len(lines) * 0.3 and tab_lines >= 3:
        return "tabla"

    # Listas: muchas líneas empezando con -, *, •, números
    list_lines = sum(1 for l in lines if re.match(r'^\s*[-*•●■]\s', l) or re.match(r'^\s*\d+[\.\)]\s', l))
    if list_lines > len(lines) * 0.3 and list_lines >= 3:
        return "lista"

    # Definiciones: empieza con patrones típicos
    first_100 = text[:200].lower()
    if any(p in first_100 for p in ['concepto', 'definición', 'se define como', 'se denomina', 'es la']):
        return "definicion"

    return "body"


def _clean_spaced_text(text: str) -> str:
    """Reconstruye texto con letras espaciadas del PDF.

    "PA R T E 3 : A M E T R O P Í A S" → "PARTE 3 : AMETROPÍAS"
    "B Á S I C O" → "BÁSICO"
    """
    # Buscar secuencias donde hay letras sueltas separadas por un espacio
    # Patrón: al menos 2 pares de "letra espacio" seguidos de una letra final
    letter = r'[A-ZÁÉÍÓÚÑa-záéíóúñ]'
    pattern = rf'({letter} ){{2,}}{letter}'

    def collapse_spaced(match):
        return match.group(0).replace(' ', '')

    result = re.sub(pattern, collapse_spaced, text)
    return result


def detect_structure(pages: list, libro_id: str) -> list:
    """Detecta títulos de capítulo y sección en las páginas.

    Args:
        pages: Lista de {page, text}
        libro_id: ID del libro para seleccionar patrones

    Returns:
        Lista de {page, text, titulo_capitulo, titulo_seccion}
    """
    patterns = STRUCTURE_PATTERNS.get(libro_id, STRUCTURE_PATTERNS["_default"])
    cap_patterns = [re.compile(p, re.MULTILINE) for p in patterns["capitulo"]]
    sec_patterns = [re.compile(p, re.MULTILINE) for p in patterns["seccion"]]

    current_capitulo = ""
    current_seccion = ""
    structured = []

    for page_data in pages:
        text = page_data["text"]
        lines = text.split('\n')

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 3:
                continue

            # Detectar capítulo
            for pat in cap_patterns:
                if pat.match(line_stripped):
                    # Limpiar: reconstruir texto con espacios intercalados
                    # "PA R T E 3 : A M E T R O P Í A S" → "PARTE 3: AMETROPÍAS"
                    clean_cap = _clean_spaced_text(line_stripped)
                    clean_cap = re.sub(r'\s{2,}', ' ', clean_cap).strip()
                    current_capitulo = clean_cap[:120]
                    current_seccion = ""
                    break

            # Detectar sección (solo si no fue detectado como capítulo)
            is_capitulo = any(pat.match(line_stripped) for pat in cap_patterns)
            if not is_capitulo:
                for pat in sec_patterns:
                    if pat.match(line_stripped):
                        # Solo aceptar como sección si es línea corta (título, no contenido)
                        # y no contiene punto seguido de más texto (indica oración, no título)
                        if len(line_stripped) < 80 and line_stripped.count('.') <= 2:
                            current_seccion = line_stripped[:100]
                            break

        structured.append({
            "page": page_data["page"],
            "text": text,
            "titulo_capitulo": current_capitulo,
            "titulo_seccion": current_seccion,
        })

    return structured


def _find_sentence_boundary(words: list, target_idx: int) -> int:
    """Busca el final de oración más cercano al target_idx.

    Retorna el índice del último word que termina una oración,
    buscando en un rango de ±30 palabras del target.
    """
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

    return best + 1  # Retorna posición después del punto


def generate_chunks_v2(structured_pages: list, libro_id: str,
                       target_size: int = TARGET_SIZE,
                       overlap: int = OVERLAP_SIZE) -> tuple:
    """Genera child chunks y parent chunks a partir de páginas estructuradas.

    Args:
        structured_pages: Lista de {page, text, titulo_capitulo, titulo_seccion}
        libro_id: ID del libro
        target_size: Palabras objetivo por chunk (default 280)
        overlap: Palabras de overlap (default 60)

    Returns:
        (child_chunks, parent_chunks)
    """
    # Construir un buffer continuo con metadata por página
    all_words = []       # Lista plana de palabras
    word_meta = []       # Metadata por palabra: (page, capitulo, seccion)

    for sp in structured_pages:
        page = sp["page"]
        cap = sp["titulo_capitulo"]
        sec = sp["titulo_seccion"]
        words = sp["text"].split()

        for w in words:
            all_words.append(w)
            word_meta.append((page, cap, sec))

    if not all_words:
        return [], []

    # Generar child chunks con overlap
    children = []
    pos = 0
    chunk_index = 0

    while pos < len(all_words):
        # Determinar fin del chunk
        end_target = pos + target_size

        if end_target >= len(all_words):
            # Último chunk: tomar todo lo que queda
            end = len(all_words)
        else:
            # Buscar límite de oración cerca del target
            end = _find_sentence_boundary(all_words, end_target)

            # Si el chunk es demasiado grande, forzar corte
            if end - pos > MAX_SIZE:
                end = _find_sentence_boundary(all_words, pos + MAX_SIZE)
                if end - pos > MAX_SIZE + 50:
                    end = pos + MAX_SIZE  # Corte duro como último recurso

        # Chunk demasiado pequeño al final: merge con anterior
        if end - pos < MIN_SIZE and children:
            prev = children[-1]
            prev["text"] = prev["text"] + " " + " ".join(all_words[pos:end])
            prev["word_count"] = len(prev["text"].split())
            prev["page_end"] = word_meta[end - 1][0]
            break

        chunk_text = " ".join(all_words[pos:end])
        page_start = word_meta[pos][0]
        page_end = word_meta[end - 1][0]

        # Metadata: tomar la del inicio del chunk (más representativa)
        capitulo = word_meta[pos][1]
        seccion = word_meta[pos][2]

        # Si hay cambio de capítulo/sección dentro del chunk, usar el más nuevo
        for i in range(pos, min(end, pos + 50)):
            if word_meta[i][1] and word_meta[i][1] != capitulo:
                capitulo = word_meta[i][1]
            if word_meta[i][2] and word_meta[i][2] != seccion:
                seccion = word_meta[i][2]

        tipo = classify_content_type(chunk_text)

        children.append({
            "id": f"{libro_id}_v2_{chunk_index:05d}",
            "libro_id": libro_id,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text,
            "word_count": end - pos,
            "titulo_capitulo": capitulo,
            "titulo_seccion": seccion,
            "tipo_contenido": tipo,
            "parent_id": None,  # Se asigna después
            "chunk_index": chunk_index,
            "version": 2,
        })

        chunk_index += 1

        # Avanzar con overlap
        next_pos = end - overlap
        if next_pos <= pos:
            next_pos = end  # Evitar loop infinito
        pos = next_pos

    # Generar parent chunks (ventana de PARENT_WINDOW children)
    parents = []
    parent_idx = 0
    i = 0

    while i < len(children):
        window_end = min(i + PARENT_WINDOW, len(children))
        window = children[i:window_end]

        # Concatenar textos de los children (sin overlap duplicado)
        parent_text_parts = []
        for j, child in enumerate(window):
            if j == 0:
                parent_text_parts.append(child["text"])
            else:
                # Quitar overlap del inicio de este child (ya está en el anterior)
                child_words = child["text"].split()
                # El overlap son las últimas ~OVERLAP_SIZE palabras del child anterior
                skip = min(overlap, len(child_words) // 3)  # No skipear más de 1/3
                parent_text_parts.append(" ".join(child_words[skip:]))

        parent_text = " ".join(parent_text_parts)
        parent_words = len(parent_text.split())

        # Si el parent es muy grande, solo tomar lo necesario
        if parent_words > MAX_PARENT_WORDS:
            parent_text = " ".join(parent_text.split()[:MAX_PARENT_WORDS])
            parent_words = MAX_PARENT_WORDS

        parent_id = f"{libro_id}_v2_parent_{parent_idx:05d}"

        parents.append({
            "id": parent_id,
            "libro_id": libro_id,
            "page_start": window[0]["page_start"],
            "page_end": window[-1]["page_end"],
            "text": parent_text,
            "word_count": parent_words,
            "titulo_capitulo": window[0]["titulo_capitulo"],
            "titulo_seccion": window[0]["titulo_seccion"],
            "child_ids": [c["id"] for c in window],
            "version": 2,
        })

        # Asignar parent_id a los children
        for child in window:
            child["parent_id"] = parent_id

        parent_idx += 1
        i += PARENT_WINDOW

    return children, parents


def parse_pdf_v2(pdf_path: str, libro_id: str) -> tuple:
    """Parsea un PDF completo a chunks v2.

    Args:
        pdf_path: Ruta al archivo PDF
        libro_id: ID del libro

    Returns:
        (children, parents) — listas de dicts
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    print(f"  Parseando {os.path.basename(pdf_path)} ({total_pages} págs)...")

    # Extraer texto por página
    pages = []
    for i in range(total_pages):
        page = doc[i]
        text = page.get_text()
        text = clean_text(text)

        if len(text.strip()) > 20:
            pages.append({"page": i + 1, "text": text})

        if (i + 1) % 500 == 0:
            print(f"    ... {i + 1}/{total_pages} páginas")

    doc.close()
    print(f"  {len(pages)} páginas con texto extraído")

    # Detectar estructura
    structured = detect_structure(pages, libro_id)

    # Generar chunks
    children, parents = generate_chunks_v2(structured, libro_id)
    print(f"  Resultado: {len(children)} children, {len(parents)} parents")

    return children, parents


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
        print(f"  ESTADO DE PARSEO V2")
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
