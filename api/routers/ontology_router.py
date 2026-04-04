"""Endpoint de ontología — traversal jerárquico ATC + SNOMED."""

from fastapi import APIRouter
from services.graph import query as read

router = APIRouter(tags=["ontology"])


@router.get("/topic/{tema}/ontology")
async def get_ontology(tema: str):
    """Clasificación ontológica: jerarquía ATC de fármacos, SNOMED de patologías, queries cruzadas."""

    resultado = {
        "tema": tema,
        "farmacos_atc": [],
        "patologias_snomed": [],
        "anatomia_snomed": [],
        "procedimientos_snomed": [],
        "query_cruzada": [],
    }

    # 1. Fármacos → ATC hierarchy
    farmacos = read("""
    MATCH (f:Farmaco)-[:ES_UN*1..5]->(cat:CategoriaATC)
    WHERE toLower(f.nombre) CONTAINS toLower($tema)
    WITH f.nombre AS farmaco, collect(DISTINCT {codigo: cat.codigo, nombre: cat.nombre, nivel: cat.nivel}) AS jerarquia
    RETURN farmaco, jerarquia
    ORDER BY farmaco
    LIMIT 20
    """, {"tema": tema})

    for r in farmacos:
        jerarquia_sorted = sorted(r["jerarquia"], key=lambda x: x.get("nivel", 0))
        resultado["farmacos_atc"].append({
            "nombre": r["farmaco"],
            "jerarquia": jerarquia_sorted
        })

    # 2. Patologías → SNOMED hierarchy
    patologias = read("""
    MATCH (p:Patologia)-[:ES_UN*1..3]->(cat:CategoriaSNOMED)
    WHERE toLower(p.nombre) CONTAINS toLower($tema)
    WITH p.nombre AS patologia, collect(DISTINCT {codigo: cat.codigo, nombre: cat.nombre, nivel: cat.nivel, sistema: cat.sistema}) AS jerarquia
    RETURN patologia, jerarquia
    ORDER BY patologia
    LIMIT 20
    """, {"tema": tema})

    for r in patologias:
        jerarquia_sorted = sorted(r["jerarquia"], key=lambda x: x.get("nivel", 0))
        resultado["patologias_snomed"].append({
            "nombre": r["patologia"],
            "jerarquia": jerarquia_sorted
        })

    # 3. Anatomía → SNOMED
    anatomia = read("""
    MATCH (e:EstructuraAnatomica)-[:ES_UN*1..3]->(cat:CategoriaSNOMED)
    WHERE toLower(e.nombre) CONTAINS toLower($tema)
    WITH e.nombre AS estructura, collect(DISTINCT {codigo: cat.codigo, nombre: cat.nombre, sistema: cat.sistema}) AS jerarquia
    RETURN estructura, jerarquia
    ORDER BY estructura
    LIMIT 20
    """, {"tema": tema})

    for r in anatomia:
        resultado["anatomia_snomed"].append({
            "nombre": r["estructura"],
            "jerarquia": r["jerarquia"]
        })

    # 4. Procedimientos → SNOMED
    procedimientos = read("""
    MATCH (p:Procedimiento)-[:ES_UN*1..3]->(cat:CategoriaSNOMED)
    WHERE toLower(p.nombre) CONTAINS toLower($tema)
    WITH p.nombre AS procedimiento, collect(DISTINCT {codigo: cat.codigo, nombre: cat.nombre, sistema: cat.sistema}) AS jerarquia
    RETURN procedimiento, jerarquia
    ORDER BY procedimiento
    LIMIT 20
    """, {"tema": tema})

    for r in procedimientos:
        resultado["procedimientos_snomed"].append({
            "nombre": r["procedimiento"],
            "jerarquia": r["jerarquia"]
        })

    # 5. Query cruzada: fármacos que tratan patologías del mismo sistema
    if resultado["patologias_snomed"]:
        sistemas = set()
        for p in resultado["patologias_snomed"]:
            for j in p["jerarquia"]:
                if j.get("sistema"):
                    sistemas.add(j["sistema"])

        for sistema in list(sistemas)[:3]:
            cruzada = read("""
            MATCH (p:Patologia)-[:ES_UN*1..3]->(sno:CategoriaSNOMED {sistema: $sistema})
            WHERE toLower(p.nombre) CONTAINS toLower($tema)
            MATCH (p)-[:SE_TRATA_CON]->(f:Farmaco)
            OPTIONAL MATCH (f)-[:ES_UN*1..5]->(atc:CategoriaATC)
            RETURN DISTINCT f.nombre AS farmaco, p.nombre AS patologia,
                   collect(DISTINCT atc.nombre)[0] AS clase_atc,
                   $sistema AS sistema
            LIMIT 20
            """, {"tema": tema, "sistema": sistema})

            for r in cruzada:
                resultado["query_cruzada"].append({
                    "farmaco": r["farmaco"],
                    "patologia": r["patologia"],
                    "clase_atc": r.get("clase_atc"),
                    "sistema_snomed": r["sistema"]
                })

    resultado["tiene_atc"] = len(resultado["farmacos_atc"]) > 0
    resultado["tiene_snomed"] = len(resultado["patologias_snomed"]) > 0 or len(resultado["anatomia_snomed"]) > 0
    resultado["tiene_cruzada"] = len(resultado["query_cruzada"]) > 0

    return resultado
