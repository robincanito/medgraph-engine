"""Deduplicacion de entidades por ACENTOS Y CAJA. Informe primero, escritura despues.

QUE HACE. Junta las entidades cuyo nombre es el mismo una vez sacadas las tildes y bajada la caja
—«clítoris»/«clitoris», «tórax»/«torax»— y las funde en un nodo canonico, transfiriendole las
aristas. Es la regla MAS ANGOSTA de las dos que el repo tiene: `eval/_fusion.clave_fusion` (la de
`eval/fusionar_entidades.py`) ademas pliega la puntuacion, el orden de las palabras, el plural y las
siglas, y encuentra el doble. **Todo grupo de acentos esta contenido en un grupo de la otra**, asi
que correr este primero no cambia el estado final: es un paso mas chico y mas facil de leer.

NUNCA SE CORRIO SOBRE ESTE CORPUS. Medido el 22-sep-2026 sobre los 154.342 nodos de entidad del
grafo: **7.362 grupos** y **7.467 nodos** que desaparecerian, todos por acentos.

LOS TRES DEFECTOS QUE TENIA, y por que habia que arreglarlos ANTES de la primera corrida (R3,
22-sep-2026). Ninguno se veia porque el guion nunca se ejecuto:

 1. **Fundia entre labels distintos.** Agrupaba dentro de `MATCH (n:{label})`, label por label, asi
    que un nodo con DOS labels caia en dos pasadas y se podia fundir con uno de un solo label. La
    regla de la casa —«nunca entre labels distintos»— no estaba garantizada por construccion. Hoy
    el grafo no tiene ni un nodo de entidad con mas de un label (medido), o sea que el defecto era
    latente; la clave del grupo ahora LLEVA el conjunto de labels y no puede volver.
 2. **Elegia otro canonico que la fusion.** `pick_canonical` ordenaba por (tiene acentos, `freq`,
    cantidad de sinonimos) y `freq` **se sobreescribe en cada carga de libro** (`cypher_entidades`
    hace `SET e.freq = ent.freq`): en el grafo viejo es la frecuencia del ULTIMO libro que cargo la
    entidad, no del corpus. La vara buena es el `degree`, que si acumula y es lo que decide quien
    gana en la recuperacion. Con dos varas distintas, correr este guion y despues la fusion dejaba
    como superviviente un nodo distinto del que la fusion sola habria elegido.
 3. **Perdia la procedencia de la arista** (la deuda que R1 dejo anotada,
    `docs/DISENO-procedencia-22sep.md` §5). El `MERGE (canon)-[:TIPO]->(t)` no copiaba NINGUNA
    propiedad: el libro, el fragmento, el perfil y la evidencia del duplicado se iban con el nodo.
    Hoy el grafo tiene cero aristas con procedencia, asi que todavia no hay nada que perder; la
    primera re-extraccion la escribe, y entonces si. El Cypher ahora es el de
    `pipeline/fusion.py`, UNA sola copia compartida con `eval/fusionar_entidades.py`.

CLI
  py -3.13 dedup_entities.py                      # INFORME (dry-run): no escribe nada
  py -3.13 dedup_entities.py --label Patologia    # un solo label
  py -3.13 dedup_entities.py --aplicar            # ESCRIBE. Snapshot de la VM antes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

try:
    import bitacora  # noqa: E402
except ModuleNotFoundError:            # el espejo OSS (medgraph-engine) no trae bitacora.py
    bitacora = None
from db import DATABASE, get_driver, run_query  # noqa: E402
from pipeline import fusion  # noqa: E402
from pipeline.normalizacion import para_busqueda  # noqa: E402
from pipeline.perfiles import taxonomia  # noqa: E402

log = logging.getLogger(__name__)

# LOS LABELS SALEN DEL PERFIL (19-sep-2026, `docs/DISENO-labels-del-perfil-19sep.md`). Estaban
# escritos a mano y eran ONCE de los doce tipos de `medicina@4`: faltaba `MoleculaBiologica`, o
# sea que el tipo nuevo de v3 iba a acumular duplicados por acentos que este script jamas
# hubiera mirado.
ENTITY_LABELS = list(taxonomia().labels)

CHECKPOINT_FILE = RAIZ / "dedup_checkpoint.json"
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos, se duplica en cada reintento


def clave_por_acentos(nombre: str) -> str:
    """LA REGLA: sin tildes y en minusculas, y nada mas.

    `normalizacion.para_busqueda` desde el 22-sep-2026 (antes eran dos lineas propias de NFKD, la
    cuarta copia del mismo plegado). NO pliega puntuacion, ni orden, ni plural, ni siglas: eso es
    `eval/_fusion.clave_fusion`, y mezclarlas convertiria a este guion en el otro sin decirlo.
    """
    return para_busqueda(nombre or "").strip()


def reintentar(func, *args, **kwargs):
    """Ejecuta con reintento y backoff: la VM del grafo corta conexiones cuando se despierta."""
    for intento in range(MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - se reintenta cualquier fallo de red/sesion
            if intento == MAX_RETRIES:
                raise
            espera = RETRY_DELAY * (2 ** intento)
            log.warning(f"  reintento {intento + 1}/{MAX_RETRIES} en {espera}s: {str(e)[:60]}")
            time.sleep(espera)


# ══════════════════════════════════════════════════════════════════════════════════════
# LECTURA: los grupos
# ══════════════════════════════════════════════════════════════════════════════════════

CAMPOS = """elementId(n) AS eid, coalesce(n.nombre, n.name) AS nombre, labels(n) AS labels,
       coalesce(n.freq, 0) AS freq, COUNT { (n)--() } AS degree"""


def censo(consultar, labels: list[str]) -> list[dict]:
    """Las entidades de esos labels, con su conjunto COMPLETO de labels, freq y degree."""
    return reintentar(consultar, f"""
        MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels)
        RETURN {CAMPOS}""", {"labels": labels})


def canonico_de(miembros: list[dict]) -> tuple[dict, list[dict]]:
    """El superviviente y los que se funden. **MISMO CRITERIO QUE `eval/fusionar_entidades`**:
    mayor `degree`, y si empatan `freq`, largo de la etiqueta y `eid`.

    Que las dos herramientas elijan el mismo canonico es lo que hace que correr esta primero sea un
    subconjunto de la otra y no una decision distinta tomada dos veces.
    """
    orden = sorted(miembros, key=lambda m: (-(m.get("degree") or 0), -(m.get("freq") or 0),
                                            -len(m.get("nombre") or ""), m["eid"]))
    return orden[0], orden[1:]


def grupos_de(entidades: list[dict]) -> list[dict]:
    """Los grupos: dos o mas entidades con LOS MISMOS labels y la misma clave por acentos.

    La clave lleva el conjunto de labels, asi que «nunca entre labels distintos» se cumple por
    construccion y no por como se recorra el grafo.
    """
    por_clave: dict = {}
    for n in entidades:
        clave = (tuple(sorted(n.get("labels") or [])), clave_por_acentos(n.get("nombre")))
        if clave[1]:
            por_clave.setdefault(clave, []).append(n)
    grupos = []
    for (labels, clave), miembros in por_clave.items():
        if len(miembros) < 2:
            continue
        canonico, duplicados = canonico_de(miembros)
        grupos.append({"clave": clave, "labels": list(labels),
                       "canonico": canonico, "duplicados": duplicados})
    return sorted(grupos, key=lambda g: (-(g["canonico"].get("degree") or 0), g["clave"]))


def tipos_del_grupo(consultar, grupo: dict) -> list[str]:
    """Los tipos de relacion que tocan a los duplicados. Salen del grafo y no de una lista escrita
    a mano: una relacion de un perfil viejo que ya no se extrae tambien hay que re-apuntarla."""
    filas = reintentar(consultar, """
        MATCH (d)-[r]-(x) WHERE elementId(d) IN $dups
        RETURN DISTINCT type(r) AS tipo""",
        {"dups": [d["eid"] for d in grupo["duplicados"]]})
    return sorted(f["tipo"] for f in filas)


def dudoso(grupo: dict) -> str | None:
    """Un grupo DUDOSO no se descarta: se marca para que una persona lo lea.

    Con la regla por acentos los sospechosos son pocos y de dos clases: los que ademas de tildes
    difieren en la CAJA (que el extractor no produce hoy: si aparece uno, algo cambio aguas arriba)
    y los de mas de tres grafias, que suelen ser la misma palabra escrita de todas las formas
    posibles pero conviene mirar.
    """
    etiquetas = [grupo["canonico"]["nombre"]] + [d["nombre"] for d in grupo["duplicados"]]
    if len({e.lower() for e in etiquetas}) < len({e for e in etiquetas}):
        return "difieren tambien por MAYUSCULAS: el extractor normaliza al crear, asi que esto no lo produjo el"
    if len(etiquetas) > 3:
        return f"{len(etiquetas)} grafias del mismo nombre: mirar que sean la misma palabra"
    return None


# ══════════════════════════════════════════════════════════════════════════════════════
# ESCRITURA
# ══════════════════════════════════════════════════════════════════════════════════════

def aplicar(grupos: list[dict], consultar, transaccion, *, avisar=None) -> dict:
    """Funde cada grupo en UNA transaccion, con el Cypher compartido de `pipeline/fusion.py`.

    Idempotente: corrido dos veces, la segunda no encuentra los duplicados y no hace nada.
    Un grupo que falla se registra y NO corta la corrida: con 7.362 grupos, abortar en el 5.000
    dejaria el grafo a medio deduplicar y sin manera de saber donde.
    """
    total = {"grupos": 0, "nodos": 0, "fallados": 0}
    for k, g in enumerate(grupos, 1):
        try:
            transaccion(fusion.sentencias_de_grupo(
                g["canonico"]["eid"], [d["eid"] for d in g["duplicados"]],
                tipos_del_grupo(consultar, g)))
        except Exception as e:  # noqa: BLE001 - un grupo roto no puede frenar los otros 7.361
            log.error(f"  grupo '{g['clave']}' FALLO: {type(e).__name__}: {str(e)[:100]}")
            total["fallados"] += 1
            continue
        total["grupos"] += 1
        total["nodos"] += len(g["duplicados"])
        if avisar:
            avisar(k, g)
    return total


# ══════════════════════════════════════════════════════════════════════════════════════
# EL PARTE
# ══════════════════════════════════════════════════════════════════════════════════════

def parte(grupos: list[dict], aplicado: bool, cuantas: int) -> str:
    nodos = sum(len(g["duplicados"]) for g in grupos)
    por_label: dict = {}
    for g in grupos:
        e = por_label.setdefault("/".join(g["labels"]), {"grupos": 0, "nodos": 0})
        e["grupos"] += 1
        e["nodos"] += len(g["duplicados"])
    dudosos = [(g, d) for g in grupos for d in [dudoso(g)] if d]
    L = [f"# Dedup por acentos — {'APLICADO' if aplicado else 'informe (dry-run)'} "
         f"({date.today().isoformat()})", "",
         "Entidades cuyo nombre es el mismo sin tildes y en minusculas. Es la regla MAS ANGOSTA de "
         "las dos: todo grupo de aca esta contenido en un grupo de `eval/_fusion.clave_fusion`, que "
         "ademas pliega puntuacion, orden, plural y siglas. El canonico es el de mayor `degree`, el "
         "MISMO criterio que la fusion — con dos varas distintas, correr las dos dejaba un "
         "superviviente que ninguna de las dos habria elegido sola.", "",
         "| | |", "|---|---|",
         f"| entidades leidas | {cuantas} |",
         f"| grupos que se funden | **{len(grupos)}** |",
         f"| nodos que desaparecen | **{nodos}** |",
         f"| grupos dudosos (revision a ojo) | {len(dudosos)} |", "",
         "## Por label", "", "| label | grupos | nodos |", "|---|---|---|"]
    L += [f"| {k} | {v['grupos']} | {v['nodos']} |"
          for k, v in sorted(por_label.items(), key=lambda kv: -kv[1]["grupos"])]
    L += ["", "## Los 40 grupos mas grandes (por degree del canonico)", "",
          "| label | canonico (degree) | se funden |", "|---|---|---|"]
    for g in grupos[:40]:
        L.append(f"| {'/'.join(g['labels'])} | **{g['canonico']['nombre']}** "
                 f"({g['canonico']['degree']}) | "
                 + "; ".join(f"{d['nombre']} ({d['degree']})" for d in g["duplicados"]) + " |")
    L += ["", "## Grupos dudosos", ""]
    L += ([f"- **{g['canonico']['nombre']}** ← "
           f"{', '.join(d['nombre'] for d in g['duplicados'])}: {razon}"
           for g, razon in dudosos[:60]]
          or ["Ninguno que la revision automatica marque."])
    if len(dudosos) > 60:
        L += ["", f"(y {len(dudosos) - 60} mas en el JSON del plan)"]
    L += ["", "## Como se aplica", "",
          "```", "# NEO4J_URI del Secret Manager. SNAPSHOT DE LA VM ANTES: borra nodos.",
          "py -3.13 dedup_entities.py            # este informe, sin escribir",
          "py -3.13 dedup_entities.py --aplicar  # escribe", "```", ""]
    return "\n".join(L) + "\n"


def cargar_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {"claves_hechas": []}


def guardar_checkpoint(datos: dict) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if bitacora:
        bitacora.configurar()
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aplicar", "--execute", action="store_true", dest="aplicar",
                    help="ESCRIBE EN EL GRAFO. Sin esto, informe.")
    ap.add_argument("--label", default=None, help="un solo label")
    ap.add_argument("--parte", default=None,
                    help=f"default: docs/E-dedup-acentos-{date.today().isoformat()}.md")
    args = ap.parse_args()

    labels = [args.label] if args.label else ENTITY_LABELS
    entidades = censo(run_query, labels)
    grupos = grupos_de(entidades)
    nodos = sum(len(g["duplicados"]) for g in grupos)
    log.info(f"{len(entidades)} entidades leidas · {len(grupos)} grupos por acentos · "
             f"{nodos} nodos que desaparecen")

    salida = RAIZ / "eval" / "resultados" / f"dedup-acentos-{date.today().isoformat()}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({"fecha": date.today().isoformat(), "aplicado": args.aplicar,
                                  "entidades": len(entidades), "grupos": grupos},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    md = Path(args.parte) if args.parte else RAIZ / "docs" / "E-dedup-acentos-22sep.md"
    md.write_text(parte(grupos, args.aplicar, len(entidades)), encoding="utf-8")
    log.info(f"plan: {salida.relative_to(RAIZ)} · parte: {md.relative_to(RAIZ)}")

    if not args.aplicar:
        log.info("DRY-RUN: no se escribio NADA en el grafo. Con --aplicar se funde.")
        return

    hechas = set(cargar_checkpoint().get("claves_hechas", []))
    pendientes = [g for g in grupos if g["clave"] not in hechas]
    if hechas:
        log.info(f"retomando desde checkpoint: {len(hechas)} grupos ya hechos")
    with get_driver() as driver:
        transaccion = fusion.transaccion_con(driver, DATABASE)

        def avisar(k, g):
            hechas.add(g["clave"])
            if k % 100 == 0:
                guardar_checkpoint({"claves_hechas": sorted(hechas)})
                log.info(f"  [{k}/{len(pendientes)}] {len(hechas)} grupos fundidos")

        total = aplicar(pendientes, run_query, transaccion, avisar=avisar)
    guardar_checkpoint({"claves_hechas": sorted(hechas)})
    log.info(f"aplicado: {total['grupos']} grupos, {total['nodos']} nodos borrados, "
             f"{total['fallados']} fallados")
    if not total["fallados"]:
        os.remove(CHECKPOINT_FILE)
        log.info("checkpoint limpiado (todo completado)")


if __name__ == "__main__":
    main()
