"""Búsqueda por keywords sobre los chunks parseados de los libros."""

import json
import os
import re
import sys
from collections import defaultdict

PARSED_DIR = os.path.join(os.path.dirname(__file__), "parsed")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")

# Cache de chunks cargados en memoria
_chunks_cache = {}


def load_chunks(libro_id: str = None) -> list:
    """Carga chunks de uno o todos los libros parseados."""
    if libro_id and libro_id in _chunks_cache:
        return _chunks_cache[libro_id]

    catalog = load_catalog()
    chunks = []

    for libro in catalog["libros"]:
        if libro["estado"] != "parseado":
            continue
        if libro_id and libro["id"] != libro_id:
            continue

        chunks_path = os.path.join(PARSED_DIR, f"{libro['id']}_chunks.json")
        if not os.path.exists(chunks_path):
            continue

        with open(chunks_path, "r", encoding="utf-8") as f:
            libro_chunks = json.load(f)
            for chunk in libro_chunks:
                chunk["libro_titulo"] = libro["titulo"]
            chunks.extend(libro_chunks)
            _chunks_cache[libro["id"]] = libro_chunks

    return chunks


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def search(query: str, libro_id: str = None, top_k: int = 5, context: int = 0) -> list:
    """Búsqueda por keywords sobre chunks.

    Scoring: cuenta ocurrencias de cada término de búsqueda en el chunk,
    ponderado por si aparece en las primeras líneas (probable título/sección).

    Args:
        query: Texto a buscar
        libro_id: Filtrar por libro específico (None = todos)
        top_k: Cantidad de resultados
        context: Líneas de contexto adicional a mostrar

    Returns:
        Lista de resultados ordenados por relevancia
    """
    chunks = load_chunks(libro_id)
    terms = normalize(query).split()

    if not terms:
        return []

    results = []

    for chunk in chunks:
        text_normalized = normalize(chunk["text"])
        text_lower = text_normalized.lower()

        # Calcular score
        score = 0
        matched_terms = set()

        for term in terms:
            term_lower = term.lower()
            # Contar ocurrencias
            count = text_lower.count(term_lower)
            if count > 0:
                matched_terms.add(term)
                score += count

                # Bonus si aparece en las primeras 200 chars (probablemente título)
                if term_lower in text_lower[:200]:
                    score += 3

        # Solo incluir si matchea al menos la mitad de los términos
        if len(matched_terms) >= max(1, len(terms) // 2):
            # Bonus por matchear más términos distintos
            score += len(matched_terms) * 2

            results.append({
                "chunk_id": chunk["id"],
                "libro_id": chunk["libro_id"],
                "libro_titulo": chunk.get("libro_titulo", chunk["libro_id"]),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "score": score,
                "matched_terms": list(matched_terms),
                "preview": get_preview(chunk["text"], terms),
                "full_text": chunk["text"] if context > 0 else None,
                "word_count": chunk["word_count"]
            })

    # Ordenar por score descendente
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def normalize(text: str) -> str:
    """Normaliza texto para búsqueda."""
    # Remover acentos comunes
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ü': 'u', 'ñ': 'n',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'Ü': 'U', 'Ñ': 'N'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def safe_print(text: str):
    """Print safe para Windows cp1252."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def get_preview(text: str, terms: list, max_len: int = 300) -> str:
    """Extrae un preview del chunk centrado en los términos encontrados."""
    text_lower = normalize(text).lower()
    terms_lower = [normalize(t).lower() for t in terms]

    # Buscar la primera ocurrencia de cualquier término
    best_pos = len(text)
    for term in terms_lower:
        pos = text_lower.find(term)
        if pos != -1 and pos < best_pos:
            best_pos = pos

    # Extraer ventana alrededor de la primera ocurrencia
    start = max(0, best_pos - 80)
    end = min(len(text), start + max_len)

    preview = text[start:end].strip()
    if start > 0:
        preview = "..." + preview
    if end < len(text):
        preview = preview + "..."

    return preview


def print_results(results: list, verbose: bool = False):
    """Imprime resultados de búsqueda."""
    if not results:
        print("\n  Sin resultados.")
        return

    print(f"\n  {len(results)} resultado(s):\n")

    for i, r in enumerate(results):
        safe_print(f"  [{i+1}] {r['libro_titulo']} - pags {r['page_start']}-{r['page_end']} (score: {r['score']})")
        safe_print(f"      Terminos: {', '.join(r['matched_terms'])}")
        safe_print(f"      {r['preview']}")

        if verbose and r.get("full_text"):
            safe_print(f"\n      --- TEXTO COMPLETO ---")
            for line in r["full_text"].split("\n"):
                safe_print(f"      {line}")
            safe_print(f"      --- FIN ---")

        print()


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("\nUso:")
        print('  python search.py "otitis media tratamiento"')
        print('  python search.py "ECG normal" --libro farreras-2020')
        print('  python search.py "otalgia" --top 10')
        print('  python search.py "ergometria indicaciones" --verbose')
        sys.exit(0)

    # Parsear argumentos
    query_parts = []
    libro_id = None
    top_k = 5
    verbose = False

    i = 0
    while i < len(args):
        if args[i] == "--libro" and i + 1 < len(args):
            libro_id = args[i + 1]
            i += 2
        elif args[i] == "--top" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif args[i] == "--verbose":
            verbose = True
            i += 1
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts)

    if not query:
        print("Error: query vacia")
        sys.exit(1)

    print(f"\n  Buscando: \"{query}\"", end="")
    if libro_id:
        print(f" (en {libro_id})", end="")
    print()

    results = search(query, libro_id=libro_id, top_k=top_k, context=1 if verbose else 0)
    print_results(results, verbose=verbose)
