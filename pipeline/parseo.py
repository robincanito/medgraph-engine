"""Parseo y chunking canonicos de MedGraph: PDF -> paginas -> estructura -> chunks.

UNA SOLA FUENTE DE VERDAD (Fase 2, tanda 1, 6-sep-2026). Hasta hoy este codigo vivia
TRES veces: `parser_v2.py` (CLI, la version nueva con reconstruccion de tablas del
9-ago), `api/services/ingest.py` (14 copias MAS VIEJAS y un chunker propio) y
`medgraph-engine/parser_v2.py` (OSS, 209 lineas atras). El bug de embeddings de
agosto vivio exactamente en una de esas copias. Doctrina de la casa: "un numero
repetido en dos archivos siempre diverge en silencio".

Quien lo importa:
  - CLI:  `parser_v2.py` y `migrate_chunks.py` son shims que re-exportan de aca.
  - API:  `api/services/ingest.py::parse_and_chunk` = parse_pdf_v2 + normalize_chunks.
  - OSS:  `medgraph-engine/pipeline/parseo.py` es una copia identica (se replica a mano
          antes de publicar; un test compara los dos archivos).

El codigo de las funciones es el de parser_v2.py del 6-sep, movido verbatim por AST.
`fitz` (PyMuPDF) se importa PEREZOSO dentro de las funciones que lo usan: importarlo al
cargar el modulo le costaba ~10 s a cada arranque en frio de Cloud Run (1-sep-2026).
"""
import os
import re
import unicodedata
from collections import Counter

# Los parametros de chunkeo y los patrones de estructura viven en pipeline/estrategia.py:
# son del DOMINIO, no del codigo. Se re-exportan con los nombres de siempre para no romper a
# quien los importe (parser_v2 y los tests los usan).
from pipeline.estrategia import (  # noqa: E402, F401  (re-export deliberado)
    MAX_PARENT_WORDS,
    MAX_SIZE,
    MIN_SIZE,
    OVERLAP_SIZE,
    PARENT_WINDOW,
    POR_DEFECTO,
    TARGET_SIZE,
    Estrategia,
)
from pipeline.estrategia import PATRONES_MEDICINA as STRUCTURE_PATTERNS  # noqa: E402, F401

MIN_FILAS_TABLA = 2
MIN_COLS_TABLA = 2
SOLAPE_BLOQUE_TABLA = 0.5          # fraccion del bloque que debe caer dentro de la tabla
MAX_CHARS_ENCABEZADO = 60          # mas largo que esto no es un nombre de columna
MIN_RATIO_LETRAS_ENCABEZADO = 0.5  # ver _parece_encabezado
MAX_PERDIDA_TOKENS = 0.02          # ver _pierde_contenido


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


def _area(r) -> float:
    return max(0.0, r.x1 - r.x0) * max(0.0, r.y1 - r.y0)


def _dentro_de_alguna(rect, cajas: list) -> bool:
    """El bloque cae mayormente dentro de alguna tabla ya reconstruida?"""
    import fitz  # PyMuPDF: perezoso, le costaba ~10 s al arranque en frio de Cloud Run
    area = _area(rect)
    if area <= 0:
        return False
    for caja in cajas:
        inter = fitz.Rect(rect)
        inter.intersect(caja)
        if _area(inter) / area > SOLAPE_BLOQUE_TABLA:
            return True
    return False


def _celda_limpia(v) -> str:
    if not v:
        return ""
    return re.sub(r'\s+', ' ', str(v)).strip()


def _fusionar_continuaciones(filas: list) -> list:
    """Une con la fila anterior las filas que son continuacion de una celda.

    Una celda multilinea se parte en varias filas, con None en las columnas que
    no siguen:
        ['VCM', '80-94 u3', 'Macrocitosis']
        [None,  None,       'Normocitosis']   <- continuacion de la de arriba
    Sin esto, 'Normocitosis' quedaria como una fila propia sin su parametro.
    """
    salida = []
    for fila in filas:
        celdas = [_celda_limpia(c) for c in fila]
        if salida and not celdas[0] and any(celdas):
            for i, c in enumerate(celdas):
                if c and i < len(salida[-1]):
                    salida[-1][i] = (salida[-1][i] + " " + c).strip()
        else:
            salida.append(celdas)
    return salida


def _parece_encabezado(fila: list) -> bool:
    """La fila sirve como encabezado, o es una fila de datos mas?

    Un encabezado es texto: nombra la columna. Una fila de datos trae valores,
    unidades y simbolos. Se mide la proporcion de letras de cada celda, que es
    lo que mejor separa un caso del otro:
        'VALOR NORMAL' -> 100% letras   sirve
        '0 puntos'     ->  86% letras   sirve (el digito es parte del nombre)
        '80-94 um3'    ->  14% letras   NO sirve, es un valor
    Basta una celda que no pase para descartar la fila entera: etiquetar mal las
    columnas es peor que no etiquetarlas.
    """
    llenas = [c for c in fila if c]
    if len(llenas) < MIN_COLS_TABLA:
        return False
    for celda in llenas:
        if len(celda) > MAX_CHARS_ENCABEZADO:
            return False
        cuerpo = re.sub(r'\s', '', celda)
        if not cuerpo:
            return False
        letras = sum(1 for c in cuerpo if c.isalpha())
        if letras / len(cuerpo) < MIN_RATIO_LETRAS_ENCABEZADO:
            return False
    return True


def render_tabla(tabla) -> str:
    """Tabla detectada -> una fila autocontenida por linea. '' si no sirve.

    NO se usa tabla.header: cuando decide que el encabezado esta fuera de la
    tabla agarra el texto de arriba, y ahi puede levantar un pedazo suelto de la
    linea anterior. En el hemograma de SIAJ convirtio 'Serie Roja:' en 'oja:' y
    etiqueto las tres columnas con eso. El unico encabezado confiable es la
    primera fila de la propia grilla, y solo si parece un encabezado.
    """
    try:
        filas = _fusionar_continuaciones(tabla.extract())
    except Exception:
        return ""
    if len(filas) < MIN_FILAS_TABLA:
        return ""

    # Una tabla partida entre dos paginas deja la continuacion sin encabezado.
    # En ese caso se emiten las filas sin nombre de columna: se conserva el corte
    # entre celdas, que ya es mas de lo que daba get_text() aplanando todo.
    encabezados = None
    if _parece_encabezado(filas[0]):
        encabezados = filas[0]
        filas = filas[1:]
    if len(filas) < MIN_FILAS_TABLA:
        return ""

    lineas = []
    for fila in filas:
        partes = []
        for i, celda in enumerate(fila):
            if not celda:
                continue
            col = encabezados[i] if encabezados and i < len(encabezados) else ""
            partes.append(f"{celda} ({col})" if col else celda)
        if not partes:
            continue
        linea = "; ".join(partes)
        lineas.append(linea if linea.endswith(('.', ':', '?', '!')) else linea + ".")

    return "\n".join(lineas)


def _tokens(texto: str) -> Counter:
    return Counter(re.findall(r'\w{2,}', texto.lower()))


def _pierde_contenido(viejo: str, nuevo: str) -> bool:
    """La reconstruccion se comio texto que get_text() si traia?

    find_tables() a veces marca como tabla una region que en realidad es prosa
    maquetada en columnas. Al excluir los bloques de esa region y reemplazarlos
    por las pocas filas que la grilla logra extraer, la pagina pierde parrafos
    enteros. Visto en el Manual de Pediatria: una pagina de 3.375 caracteres
    quedaba en 193.

    Perder texto es peor que aplanarlo, asi que ante cualquier perdida se vuelve
    a get_text(). El cambio solo puede agregar estructura, nunca sacar contenido.
    """
    faltan = _tokens(viejo) - _tokens(nuevo)
    total = sum(_tokens(viejo).values())
    return total > 0 and sum(faltan.values()) / total > MAX_PERDIDA_TOKENS


def extraer_texto_pagina(page) -> str:
    """Texto de la pagina con las tablas reconstruidas en su lugar de lectura."""
    import fitz  # PyMuPDF: perezoso, le costaba ~10 s al arranque en frio de Cloud Run
    plano = page.get_text()
    try:
        detectadas = list(page.find_tables().tables)
    except Exception:
        return plano

    if not detectadas:
        return plano

    reconstruidas = []
    for t in detectadas:
        texto = render_tabla(t)
        if texto:
            reconstruidas.append((fitz.Rect(t.bbox), texto))

    if not reconstruidas:
        return plano

    # La prosa se toma por bloques para poder descartar los que son la tabla — si
    # no, el contenido quedaria duplicado: una vez aplanado y otra reconstruido.
    cajas = [caja for caja, _ in reconstruidas]
    piezas = []
    for b in page.get_text("blocks"):
        rect = fitz.Rect(b[:4])
        if _dentro_de_alguna(rect, cajas):
            continue
        piezas.append((rect.y0, rect.x0, b[4]))

    piezas.extend((caja.y0, caja.x0, texto) for caja, texto in reconstruidas)
    piezas.sort(key=lambda p: (round(p[0], 1), p[1]))
    armado = "\n".join(p[2] for p in piezas)

    return plano if _pierde_contenido(plano, armado) else armado


def classify_content_type(text: str) -> str:
    """Clasifica el tipo de contenido de un chunk.

    OJO: generate_chunks_v2 rearma el chunk con " ".join(palabras), asi que el
    texto que llega aca NO tiene saltos de linea. Todo criterio que cuente lineas
    ve una sola linea y no se dispara nunca. Por eso las tablas se reconocen por
    la firma que deja render_tabla, no por estructura de lineas.
    """
    lines = text.strip().split('\n')
    if not lines:
        return "body"

    # Tablas: filas emitidas por render_tabla -> "valor (COLUMNA); valor (COLUMNA)."
    if len(re.findall(r'\([^()]{2,40}\);', text)) >= 3:
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


def detect_structure(pages: list, libro_id: str, estrategia: Estrategia = POR_DEFECTO) -> list:
    """Detecta títulos de capítulo y sección en las páginas.

    Args:
        pages: Lista de {page, text}
        libro_id: ID del libro para seleccionar patrones

    Returns:
        Lista de {page, text, titulo_capitulo, titulo_seccion}
    """
    # Los patrones son del DOMINIO (pipeline/estrategia.py), no de este archivo.
    patterns = estrategia.patrones_de(libro_id)
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
                       target_size: int | None = None,
                       overlap: int | None = None,
                       estrategia: Estrategia = POR_DEFECTO) -> tuple:
    """Genera child chunks y parent chunks a partir de páginas estructuradas.

    Args:
        structured_pages: Lista de {page, text, titulo_capitulo, titulo_seccion}
        libro_id: ID del libro
        target_size: Palabras objetivo por chunk. None = lo que diga la estrategia.
        overlap: Palabras de solape. None = lo que diga la estrategia.
        estrategia: Estrategia del dominio (tamaños). Default: la de medicina de siempre.

    Returns:
        (child_chunks, parent_chunks)

    `target_size` y `overlap` siguen aceptandose sueltos porque hay llamadores viejos que los
    pasan; cuando vienen, ganan sobre la estrategia (override puntual de una corrida).
    """
    target_size = estrategia.target_size if target_size is None else target_size
    overlap = estrategia.overlap if overlap is None else overlap
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
            if end - pos > estrategia.max_size:
                end = _find_sentence_boundary(all_words, pos + estrategia.max_size)
                if end - pos > estrategia.max_size + 50:
                    end = pos + estrategia.max_size  # Corte duro como último recurso

        # Chunk demasiado pequeño al final: merge con anterior
        if end - pos < estrategia.min_size and children:
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

    # Generar parent chunks (ventana de parent_window children)
    parents = []
    parent_idx = 0
    i = 0

    while i < len(children):
        window_end = min(i + estrategia.parent_window, len(children))
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
        if parent_words > estrategia.max_parent_words:
            parent_text = " ".join(parent_text.split()[:estrategia.max_parent_words])
            parent_words = estrategia.max_parent_words

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
        i += estrategia.parent_window

    return children, parents


def parse_pdf_v2(pdf_path: str, libro_id: str, estrategia: Estrategia = POR_DEFECTO) -> tuple:
    """Parsea un PDF completo a chunks v2.

    Args:
        pdf_path: Ruta al archivo PDF
        libro_id: ID del libro
        estrategia: Como parsear y chunkear este material (pipeline/estrategia.py).
            El default es medicina, o sea el comportamiento historico.

    Returns:
        (children, parents) — listas de dicts
    """
    import fitz  # PyMuPDF: perezoso, le costaba ~10 s al arranque en frio de Cloud Run
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    print(f"  Parseando {os.path.basename(pdf_path)} ({total_pages} págs)...")

    # Extraer texto por página
    pages = []
    for i in range(total_pages):
        page = doc[i]
        text = extraer_texto_pagina(page)
        text = clean_text(text)

        if len(text.strip()) > 20:
            pages.append({"page": i + 1, "text": text})

        if (i + 1) % 500 == 0:
            print(f"    ... {i + 1}/{total_pages} páginas")

    doc.close()
    print(f"  {len(pages)} páginas con texto extraído")

    # Detectar estructura
    structured = detect_structure(pages, libro_id, estrategia)

    # Generar chunks
    children, parents = generate_chunks_v2(structured, libro_id, estrategia=estrategia)
    print(f"  Resultado: {len(children)} children, {len(parents)} parents")

    return children, parents


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
