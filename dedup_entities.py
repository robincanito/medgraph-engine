"""Deduplicación de entidades en Neo4j - Fase 1: Normalización de acentos.

Encuentra entidades duplicadas que difieren solo por acentos/tildes
y las mergea en un nodo canónico, transfiriendo todas las relaciones.

v2: Retry con backoff, batch Cypher por label, checkpoint para retomar.

CLI:
  python dedup_entities.py              # dry-run (solo reporta)
  python dedup_entities.py --execute    # ejecuta merge
  python dedup_entities.py --label Patologia  # solo un label
"""

import sys
import os
import json
import time
import unicodedata

# Fix Windows encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from db import run_query, run_write

ENTITY_LABELS = [
    "Patologia", "EstructuraAnatomica", "Procedimiento", "Farmaco",
    "GrupoFarmacologico", "Agente", "Signo", "Sintoma",
    "MetodoDx", "Hallazgo", "Parametro",
]

CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), "dedup_checkpoint.json")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubles each retry


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def has_accents(s: str) -> bool:
    return s != strip_accents(s)


def retry_query(func, *args, **kwargs):
    """Execute a DB function with retry + exponential backoff."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            delay = RETRY_DELAY * (2 ** attempt)
            print(f"      Retry {attempt+1}/{MAX_RETRIES} in {delay}s: {str(e)[:60]}")
            time.sleep(delay)


def pick_canonical(nodes: list) -> dict:
    return max(nodes, key=lambda n: (
        has_accents(n["nombre"]),
        n.get("freq") or 0,
        len(n.get("sinonimos") or []),
    ))


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed_labels": [], "stats": {}}


def save_checkpoint(data: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def fetch_entities(label: str) -> list:
    return retry_query(run_query, f"""
        MATCH (n:{label})
        RETURN elementId(n) AS eid, n.nombre AS nombre,
               n.freq AS freq, n.sinonimos AS sinonimos
    """)


def find_accent_groups(entities: list) -> dict:
    groups = {}
    for ent in entities:
        key = strip_accents(ent["nombre"].lower().strip())
        groups.setdefault(key, []).append(ent)
    return {k: v for k, v in groups.items() if len(v) > 1}


def merge_group_batch(canon: dict, duplicates: list, label: str) -> dict:
    """Mergea un grupo usando queries batch con retry."""
    canon_eid = canon["eid"]
    total = {"menciona": 0, "rels_out": 0, "rels_in": 0, "deleted": 0}

    all_sinonimos = set(canon.get("sinonimos") or [])
    total_freq = canon.get("freq") or 0
    dup_eids = [d["eid"] for d in duplicates]

    for dup in duplicates:
        all_sinonimos.add(dup["nombre"])
        for s in (dup.get("sinonimos") or []):
            all_sinonimos.add(s)
        total_freq += dup.get("freq") or 0

    # 1. Batch: transferir MENCIONA de todos los dups al canon
    menciona = retry_query(run_query, """
        UNWIND $dup_eids AS deid
        MATCH (c:Chunk)-[:MENCIONA]->(dup) WHERE elementId(dup) = deid
        RETURN DISTINCT elementId(c) AS chunk_eid
    """, {"dup_eids": dup_eids})

    if menciona:
        retry_query(run_write, """
            UNWIND $chunks AS ceid
            MATCH (c) WHERE elementId(c) = ceid
            MATCH (canon) WHERE elementId(canon) = $canon_eid
            MERGE (c)-[:MENCIONA]->(canon)
        """, {"chunks": [m["chunk_eid"] for m in menciona], "canon_eid": canon_eid})
        total["menciona"] = len(menciona)

    # 2. Batch: transferir rels outgoing de todos los dups
    rels_out = retry_query(run_query, """
        UNWIND $dup_eids AS deid
        MATCH (dup)-[r]->(target)
        WHERE elementId(dup) = deid
          AND elementId(target) <> $canon_eid
          AND NOT elementId(target) IN $dup_eids
          AND type(r) <> 'MENCIONA'
        RETURN DISTINCT type(r) AS rtype, elementId(target) AS target_eid
    """, {"dup_eids": dup_eids, "canon_eid": canon_eid})

    if rels_out:
        by_type = {}
        for r in rels_out:
            by_type.setdefault(r["rtype"], []).append(r["target_eid"])
        for rtype, targets in by_type.items():
            retry_query(run_write, f"""
                UNWIND $targets AS teid
                MATCH (canon) WHERE elementId(canon) = $canon_eid
                MATCH (t) WHERE elementId(t) = teid
                MERGE (canon)-[:{rtype}]->(t)
            """, {"targets": targets, "canon_eid": canon_eid})
        total["rels_out"] = len(rels_out)

    # 3. Batch: transferir rels incoming de todos los dups
    rels_in = retry_query(run_query, """
        UNWIND $dup_eids AS deid
        MATCH (source)-[r]->(dup)
        WHERE elementId(dup) = deid
          AND elementId(source) <> $canon_eid
          AND NOT elementId(source) IN $dup_eids
          AND type(r) <> 'MENCIONA'
        RETURN DISTINCT type(r) AS rtype, elementId(source) AS source_eid
    """, {"dup_eids": dup_eids, "canon_eid": canon_eid})

    if rels_in:
        by_type = {}
        for r in rels_in:
            by_type.setdefault(r["rtype"], []).append(r["source_eid"])
        for rtype, sources in by_type.items():
            retry_query(run_write, f"""
                UNWIND $sources AS seid
                MATCH (s) WHERE elementId(s) = seid
                MATCH (canon) WHERE elementId(canon) = $canon_eid
                MERGE (s)-[:{rtype}]->(canon)
            """, {"sources": sources, "canon_eid": canon_eid})
        total["rels_in"] = len(rels_in)

    # 4. Batch: DETACH DELETE todos los dups de una vez
    retry_query(run_write, """
        UNWIND $dup_eids AS deid
        MATCH (n) WHERE elementId(n) = deid
        DETACH DELETE n
    """, {"dup_eids": dup_eids})
    total["deleted"] = len(duplicates)

    # 5. Actualizar canónico
    all_sinonimos.discard(canon["nombre"])
    retry_query(run_write, """
        MATCH (n) WHERE elementId(n) = $eid
        SET n.sinonimos = $sins, n.freq = $freq
    """, {"eid": canon_eid, "sins": sorted(list(all_sinonimos)), "freq": total_freq})

    return total


def dedup_label(label: str, execute: bool = False) -> dict:
    entities = fetch_entities(label)
    groups = find_accent_groups(entities)

    if not groups:
        return {"groups": 0, "deleted": 0, "menciona": 0, "rels": 0}

    total = {"groups": len(groups), "deleted": 0, "menciona": 0, "rels": 0}
    processed = 0

    for key, nodes in groups.items():
        canon = pick_canonical(nodes)
        dups = [n for n in nodes if n["eid"] != canon["eid"]]

        if not execute:
            dup_names = [d["nombre"] for d in dups]
            print(f"    {canon['nombre']}  <-  {dup_names}")
            total["deleted"] += len(dups)
            continue

        try:
            stats = merge_group_batch(canon, dups, label)
            total["deleted"] += stats["deleted"]
            total["menciona"] += stats["menciona"]
            total["rels"] += stats["rels_out"] + stats["rels_in"]
            processed += 1

            if processed % 50 == 0:
                print(f"    [{processed}/{len(groups)}] {total['deleted']} eliminados")

        except Exception as e:
            print(f"    ERROR en grupo '{key}': {str(e)[:80]}")
            continue

    return total


def main():
    execute = "--execute" in sys.argv
    target_label = None
    if "--label" in sys.argv:
        idx = sys.argv.index("--label")
        target_label = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    labels = [target_label] if target_label else ENTITY_LABELS
    mode = "EXECUTE" if execute else "DRY-RUN"

    # Load checkpoint
    checkpoint = load_checkpoint()
    if execute and not target_label:
        completed = checkpoint.get("completed_labels", [])
        remaining = [l for l in labels if l not in completed]
        if completed:
            print(f"  Retomando desde checkpoint. Ya completados: {completed}")
            labels = remaining

    print(f"\n{'='*60}")
    print(f"  DEDUP FASE 1 v2: Normalización de acentos [{mode}]")
    print(f"  Labels a procesar: {labels}")
    print(f"{'='*60}\n")

    grand_total = {"groups": 0, "deleted": 0, "menciona": 0, "rels": 0}

    for label in labels:
        print(f"--- {label} ---")
        t0 = time.time()

        try:
            stats = dedup_label(label, execute=execute)
        except Exception as e:
            print(f"  FATAL en {label}: {str(e)[:100]}")
            print(f"  Guardando checkpoint y saliendo...")
            save_checkpoint(checkpoint)
            sys.exit(1)

        elapsed = time.time() - t0

        for k in grand_total:
            grand_total[k] += stats[k]

        if stats["groups"] == 0:
            print("  Sin duplicados por acentos")
        else:
            print(f"  {stats['groups']} grupos, {stats['deleted']} nodos eliminados" +
                  (f", {stats['menciona']} MENCIONA, {stats['rels']} rels ({elapsed:.1f}s)" if execute else ""))

        # Save checkpoint per label
        if execute:
            checkpoint["completed_labels"] = checkpoint.get("completed_labels", []) + [label]
            checkpoint["stats"] = checkpoint.get("stats", {})
            checkpoint["stats"][label] = stats
            save_checkpoint(checkpoint)
            print(f"  Checkpoint guardado.")

        print()

    # Resumen
    print(f"{'='*60}")
    print(f"  RESUMEN {'(ejecutado)' if execute else '(dry-run)'}")
    print(f"{'='*60}")
    print(f"  Grupos duplicados: {grand_total['groups']}")
    print(f"  Nodos {'eliminados' if execute else 'a eliminar'}: {grand_total['deleted']}")
    if execute:
        print(f"  MENCIONA transferidas: {grand_total['menciona']}")
        print(f"  Relaciones transferidas: {grand_total['rels']}")

        # Limpiar checkpoint al terminar
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("  Checkpoint limpiado (todo completado).")

    print()


if __name__ == "__main__":
    main()
