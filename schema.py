"""Define y crea el schema de MedGraph en Neo4j."""

from db import run_write, run_query

CONSTRAINTS = [
    # Unicidad por nombre para cada tipo de nodo
    "CREATE CONSTRAINT patologia_nombre IF NOT EXISTS FOR (n:Patologia) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT sintoma_nombre IF NOT EXISTS FOR (n:Sintoma) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT signo_nombre IF NOT EXISTS FOR (n:Signo) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT metodo_dx_nombre IF NOT EXISTS FOR (n:MetodoDx) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT tratamiento_nombre IF NOT EXISTS FOR (n:Tratamiento) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT farmaco_nombre IF NOT EXISTS FOR (n:Farmaco) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT procedimiento_nombre IF NOT EXISTS FOR (n:Procedimiento) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT parametro_nombre IF NOT EXISTS FOR (n:Parametro) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT up_id IF NOT EXISTS FOR (n:UP) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT tema_nombre IF NOT EXISTS FOR (n:Tema) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT fuente_id IF NOT EXISTS FOR (n:Fuente) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT especialidad_nombre IF NOT EXISTS FOR (n:Especialidad) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT agente_nombre IF NOT EXISTS FOR (n:Agente) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT grupo_farmacologico_nombre IF NOT EXISTS FOR (n:GrupoFarmacologico) REQUIRE n.nombre IS UNIQUE",
    "CREATE CONSTRAINT grupo_etario_nombre IF NOT EXISTS FOR (n:GrupoEtario) REQUIRE n.nombre IS UNIQUE",
    # v2: chunks y parent chunks
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT parent_chunk_id IF NOT EXISTS FOR (p:ParentChunk) REQUIRE p.id IS UNIQUE",
]

INDEXES = [
    # Índices full-text para búsqueda por texto
    """CREATE FULLTEXT INDEX busqueda_patologias IF NOT EXISTS
       FOR (n:Patologia) ON EACH [n.nombre, n.definicion]""",
    """CREATE FULLTEXT INDEX busqueda_sintomas IF NOT EXISTS
       FOR (n:Sintoma) ON EACH [n.nombre, n.descripcion]""",
    """CREATE FULLTEXT INDEX busqueda_temas IF NOT EXISTS
       FOR (n:Tema) ON EACH [n.nombre, n.descripcion]""",
    # Full-text index sobre chunks v2 (normalizado, multi-campo)
    """CREATE FULLTEXT INDEX busqueda_chunks_v2 IF NOT EXISTS
       FOR (n:Chunk) ON EACH [n.text_busqueda, n.titulo_seccion_busqueda,
                               n.titulo_capitulo_busqueda, n.keywords]""",
]

# Índices a borrar durante migración v2
INDEXES_TO_DROP = [
    "DROP INDEX busqueda_chunks IF EXISTS",
]


def create_schema(drop_old: bool = False):
    """Crea constraints e índices en Neo4j.

    Args:
        drop_old: Si True, borra índices viejos (para migración v2)
    """
    if drop_old:
        print("Borrando índices viejos...")
        for idx in INDEXES_TO_DROP:
            try:
                run_write(idx)
                print(f"  DROPPED: {idx}")
            except Exception as e:
                print(f"  SKIP: {e}")

    print("Creando constraints...")
    for c in CONSTRAINTS:
        try:
            run_write(c)
            # Extraer label del constraint
            if "FOR (n:" in c:
                label = c.split("FOR (n:")[1].split(")")[0]
            elif "FOR (c:" in c:
                label = c.split("FOR (c:")[1].split(")")[0]
            elif "FOR (p:" in c:
                label = c.split("FOR (p:")[1].split(")")[0]
            else:
                label = "?"
            print(f"  OK: {label}")
        except Exception as e:
            print(f"  SKIP: {e}")

    print("\nCreando índices full-text...")
    for idx in INDEXES:
        try:
            run_write(idx)
            name = idx.split("INDEX ")[1].split(" IF")[0]
            print(f"  OK: {name}")
        except Exception as e:
            print(f"  SKIP: {e}")

    print("\nSchema creado.")


def show_schema():
    """Muestra el schema actual."""
    constraints = run_query("SHOW CONSTRAINTS")
    indexes = run_query("SHOW INDEXES")

    print(f"\n=== CONSTRAINTS ({len(constraints)}) ===")
    for c in constraints:
        print(f"  {c.get('name', '?')}: {c.get('labelsOrTypes', '?')}")

    print(f"\n=== INDEXES ({len(indexes)}) ===")
    for i in indexes:
        print(f"  {i.get('name', '?')}: {i.get('labelsOrTypes', '?')} ({i.get('type', '?')})")


if __name__ == "__main__":
    create_schema()
    show_schema()
