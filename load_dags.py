"""Carga DAGs desde YAML a Neo4j como relaciones PATHWAY y CLINICAL.

Los nodos referenciados deben existir en el grafo. Si no existen,
se crean como nodos genéricos con el tipo indicado.

Uso:
  python load_dags.py                     # Cargar todos los YAML de dags/
  python load_dags.py dags/otitis.yaml    # Cargar uno específico
  python load_dags.py --list              # Listar DAGs cargados
  python load_dags.py --delete nombre     # Borrar un DAG
"""

import os
import sys
import yaml
from db import run_write, run_query

DAGS_DIR = os.path.join(os.path.dirname(__file__), "dags")

# Mapa tipo_nodo -> label Neo4j
TIPO_TO_LABEL = {
    "patologia": "Patologia",
    "sintoma": "Sintoma",
    "signo": "Signo",
    "procedimiento": "Procedimiento",
    "farmaco": "Farmaco",
    "agente": "Agente",
    "hallazgo": "Hallazgo",
    "metodo_dx": "MetodoDx",
    "parametro": "Parametro",
    "estructura_anatomica": "EstructuraAnatomica",
    "grupo_farmacologico": "GrupoFarmacologico",
}


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_node_exists(nombre: str, tipo_nodo: str):
    """Si el nodo no existe en el grafo, lo crea."""
    label = TIPO_TO_LABEL.get(tipo_nodo, "Hallazgo")

    existing = run_query(f"""
    MATCH (n:{label})
    WHERE toLower(n.nombre) = toLower($nombre)
    RETURN n.nombre AS nombre LIMIT 1
    """, {"nombre": nombre})

    if not existing:
        # Buscar en cualquier label
        any_match = run_query("""
        MATCH (n)
        WHERE toLower(n.nombre) = toLower($nombre)
        AND NOT n:Chunk AND NOT n:ParentChunk
        RETURN n.nombre AS nombre, labels(n)[0] AS label LIMIT 1
        """, {"nombre": nombre})

        if not any_match:
            # Crear nodo nuevo
            run_write(f"""
            CREATE (n:{label} {{nombre: $nombre, tipo: $tipo, fuente: 'dag'}})
            """, {"nombre": nombre, "tipo": tipo_nodo})
            return "created"
        return "found_other_label"
    return "exists"


def load_dag(dag: dict) -> dict:
    """Carga un DAG a Neo4j."""
    nombre_dag = dag["nombre"]
    tipo_dag = dag["tipo"]  # pathway o clinical
    rel_type = "PATHWAY" if tipo_dag == "pathway" else "CLINICAL"
    pasos = dag["pasos"]

    print(f"  Cargando DAG: {nombre_dag} ({tipo_dag}, {len(pasos)} pasos)")

    # Borrar DAG existente con mismo nombre
    run_write(f"""
    MATCH ()-[r:{rel_type} {{nombre_dag: $nombre}}]->()
    DELETE r
    """, {"nombre": nombre_dag})

    # Asegurar que todos los nodos existen
    for paso in pasos:
        status = ensure_node_exists(paso["nodo"], paso["tipo_nodo"])
        if status == "created":
            print(f"    Nodo creado: {paso['nodo']} ({paso['tipo_nodo']})")

    # Crear relaciones entre pasos consecutivos
    created = 0
    for i in range(len(pasos) - 1):
        current = pasos[i]
        next_paso = pasos[i + 1]

        # Si tienen el mismo orden, son bifurcaciones del paso anterior
        # Conectar desde el último paso con orden menor
        if current["orden"] == next_paso["orden"]:
            continue

        current_label = TIPO_TO_LABEL.get(current["tipo_nodo"], "Hallazgo")
        next_label = TIPO_TO_LABEL.get(next_paso["tipo_nodo"], "Hallazgo")

        props = {
            "nombre_dag": nombre_dag,
            "orden": next_paso["orden"],
            "tipo_paso": next_paso.get("tipo_paso", ""),
            "condicion": next_paso.get("condicion", ""),
            "nota": next_paso.get("nota", ""),
        }

        try:
            run_write(f"""
            MATCH (a:{current_label}) WHERE toLower(a.nombre) = toLower($from_name)
            MATCH (b:{next_label}) WHERE toLower(b.nombre) = toLower($to_name)
            CREATE (a)-[:{rel_type} {{
                nombre_dag: $props.nombre_dag,
                orden: $props.orden,
                tipo_paso: $props.tipo_paso,
                condicion: $props.condicion,
                nota: $props.nota
            }}]->(b)
            """, {"from_name": current["nodo"], "to_name": next_paso["nodo"], "props": props})
            created += 1
        except Exception as e:
            print(f"    Error: {current['nodo']} -> {next_paso['nodo']}: {str(e)[:60]}")

    # Para bifurcaciones (mismo orden), conectar desde el paso anterior
    orders = sorted(set(p["orden"] for p in pasos))
    for idx, order in enumerate(orders):
        pasos_at_order = [p for p in pasos if p["orden"] == order]
        if len(pasos_at_order) > 1:
            # Encontrar el último paso del orden anterior
            prev_order = orders[idx - 1] if idx > 0 else None
            if prev_order is not None:
                prev_pasos = [p for p in pasos if p["orden"] == prev_order]
                source = prev_pasos[-1]  # Último del orden anterior
                source_label = TIPO_TO_LABEL.get(source["tipo_nodo"], "Hallazgo")

                for branch in pasos_at_order:
                    branch_label = TIPO_TO_LABEL.get(branch["tipo_nodo"], "Hallazgo")
                    props = {
                        "nombre_dag": nombre_dag,
                        "orden": branch["orden"],
                        "tipo_paso": branch.get("tipo_paso", "decision"),
                        "condicion": branch.get("condicion", ""),
                        "nota": branch.get("nota", ""),
                    }
                    try:
                        run_write(f"""
                        MATCH (a:{source_label}) WHERE toLower(a.nombre) = toLower($from_name)
                        MATCH (b:{branch_label}) WHERE toLower(b.nombre) = toLower($to_name)
                        CREATE (a)-[:{rel_type} {{
                            nombre_dag: $props.nombre_dag,
                            orden: $props.orden,
                            tipo_paso: $props.tipo_paso,
                            condicion: $props.condicion,
                            nota: $props.nota
                        }}]->(b)
                        """, {"from_name": source["nodo"], "to_name": branch["nodo"], "props": props})
                        created += 1
                    except Exception as e:
                        print(f"    Error bifurcacion: {source['nodo']} -> {branch['nodo']}: {str(e)[:60]}")

    print(f"  Resultado: {created} relaciones {rel_type}")
    return {"dag": nombre_dag, "relations": created}


def list_dags():
    """Lista DAGs cargados en Neo4j."""
    for rel_type in ["PATHWAY", "CLINICAL"]:
        dags = run_query(f"""
        MATCH ()-[r:{rel_type}]->()
        RETURN DISTINCT r.nombre_dag AS nombre, count(r) AS relaciones
        ORDER BY nombre
        """)
        if dags:
            print(f"\n  {rel_type}:")
            for d in dags:
                print(f"    {d['nombre']}: {d['relaciones']} relaciones")


def delete_dag(nombre: str):
    """Borra un DAG por nombre."""
    for rel_type in ["PATHWAY", "CLINICAL"]:
        run_write(f"""
        MATCH ()-[r:{rel_type} {{nombre_dag: $nombre}}]->()
        DELETE r
        """, {"nombre": nombre})
    print(f"  DAG '{nombre}' borrado")


# CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Cargar todos los YAML
        if not os.path.exists(DAGS_DIR):
            print("No existe directorio dags/")
            sys.exit(1)

        yamls = [f for f in os.listdir(DAGS_DIR) if f.endswith((".yaml", ".yml"))]
        if not yamls:
            print("No hay archivos YAML en dags/")
            sys.exit(1)

        print(f"Cargando {len(yamls)} DAGs...")
        for fname in sorted(yamls):
            dag = load_yaml(os.path.join(DAGS_DIR, fname))
            load_dag(dag)

        print("\nDAGs cargados:")
        list_dags()

    elif sys.argv[1] == "--list":
        list_dags()

    elif sys.argv[1] == "--delete" and len(sys.argv) > 2:
        delete_dag(sys.argv[2])

    else:
        path = sys.argv[1]
        if os.path.exists(path):
            dag = load_yaml(path)
            load_dag(dag)
        else:
            print(f"Archivo no encontrado: {path}")
