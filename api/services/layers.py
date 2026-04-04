"""Layer executors — cada capa busca en una dimensión del conocimiento.
Se ejecutan en paralelo via asyncio.gather.
"""

from services import graph, vector


def execute_ontology(analysis: dict) -> dict:
    """Traversal ontológico ATC + SNOMED. Bidireccional: sube y baja por ES_UN."""
    result = {"farmacos_atc": [], "entidades_snomed": [], "query_cruzada": [], "categorias_encontradas": []}

    for ent in analysis.get("entidades_detectadas", []):
        buscar_en = ent.get("buscar_en", "")
        texto = ent.get("texto", "")

        if buscar_en == "CategoriaATC":
            # Buscar categoría → bajar a fármacos (HACIA ABAJO)
            farmacos = graph.query("""
            MATCH (cat:CategoriaATC)
            WHERE toLower(cat.nombre) CONTAINS toLower($term)
            OPTIONAL MATCH (f:Farmaco)-[:ES_UN*1..5]->(cat)
            RETURN cat.nombre AS categoria, cat.codigo AS codigo, cat.nivel AS nivel,
                   collect(DISTINCT f.nombre)[0..20] AS farmacos
            """, {"term": texto})
            for r in farmacos:
                result["categorias_encontradas"].append({
                    "nombre": r["categoria"], "codigo": r["codigo"],
                    "nivel": r["nivel"], "tipo": "ATC",
                    "miembros": [f for f in (r["farmacos"] or []) if f]
                })

        elif buscar_en == "Farmaco":
            # Buscar fármaco → subir a categorías (HACIA ARRIBA)
            cats = graph.query("""
            MATCH (f:Farmaco)-[:ES_UN*1..5]->(cat:CategoriaATC)
            WHERE toLower(f.nombre) CONTAINS toLower($term)
            RETURN f.nombre AS farmaco,
                   collect(DISTINCT {codigo: cat.codigo, nombre: cat.nombre, nivel: cat.nivel}) AS jerarquia
            LIMIT 5
            """, {"term": texto})
            for r in cats:
                result["farmacos_atc"].append({
                    "nombre": r["farmaco"],
                    "jerarquia": sorted(r["jerarquia"], key=lambda x: x.get("nivel", 0))
                })

        elif buscar_en == "CategoriaSNOMED":
            # Buscar sistema/categoría → bajar a entidades
            entidades = graph.query("""
            MATCH (cat:CategoriaSNOMED)
            WHERE toLower(cat.nombre) CONTAINS toLower($term)
            OPTIONAL MATCH (e)-[:ES_UN*1..3]->(cat)
            WHERE e:Patologia OR e:EstructuraAnatomica OR e:Procedimiento
            RETURN cat.nombre AS categoria, cat.codigo AS codigo, cat.sistema AS sistema,
                   collect(DISTINCT {nombre: e.nombre, tipo: labels(e)[0]})[0..20] AS miembros
            """, {"term": texto})
            for r in entidades:
                result["categorias_encontradas"].append({
                    "nombre": r["categoria"], "codigo": r["codigo"],
                    "sistema": r["sistema"], "tipo": "SNOMED",
                    "miembros": [m for m in (r["miembros"] or []) if m.get("nombre")]
                })

        elif buscar_en in ("Patologia", "EstructuraAnatomica", "Procedimiento", "Signo", "Sintoma", "MetodoDx", "Hallazgo", "Agente", "Parametro"):
            # Buscar entidad → subir a categoría SNOMED
            cats = graph.query(f"""
            MATCH (e:{buscar_en})-[:ES_UN*1..3]->(cat:CategoriaSNOMED)
            WHERE toLower(e.nombre) CONTAINS toLower($term)
            RETURN e.nombre AS entidad, labels(e)[0] AS tipo,
                   collect(DISTINCT {{nombre: cat.nombre, sistema: cat.sistema, nivel: cat.nivel}}) AS jerarquia
            LIMIT 5
            """, {"term": texto})
            for r in cats:
                result["entidades_snomed"].append({
                    "nombre": r["entidad"], "tipo": r["tipo"],
                    "jerarquia": sorted(r["jerarquia"], key=lambda x: x.get("nivel", 0))
                })

    # Query cruzada: si hay ATC + SNOMED, cruzar
    atc_cats = [c for c in result["categorias_encontradas"] if c["tipo"] == "ATC"]
    snomed_cats = [c for c in result["categorias_encontradas"] if c["tipo"] == "SNOMED"]

    if atc_cats and snomed_cats:
        for atc in atc_cats[:2]:
            for sno in snomed_cats[:2]:
                cruzada = graph.query("""
                MATCH (f:Farmaco)-[:ES_UN*1..5]->(atc:CategoriaATC {codigo: $atc_code})
                MATCH (p:Patologia)-[:SE_TRATA_CON]->(f)
                MATCH (p)-[:ES_UN*1..3]->(sno:CategoriaSNOMED {codigo: $sno_code})
                RETURN DISTINCT f.nombre AS farmaco, p.nombre AS patologia
                LIMIT 15
                """, {"atc_code": atc["codigo"], "sno_code": sno["codigo"]})
                for r in cruzada:
                    result["query_cruzada"].append({
                        "farmaco": r["farmaco"], "patologia": r["patologia"],
                        "clase_atc": atc["nombre"], "sistema_snomed": sno.get("sistema", "")
                    })

    return result


def execute_graph(analysis: dict) -> dict:
    """Busca relaciones directas en el grafo para entidades detectadas."""
    result = {"patologias": [], "procedimientos": [], "relaciones": []}

    for ent in analysis.get("entidades_detectadas", []):
        buscar_en = ent.get("buscar_en", "")
        texto = ent.get("texto", "")

        if buscar_en == "Patologia":
            patos = graph.get_pathology(texto)
            if patos:
                result["patologias"].extend(patos[:5])

        elif buscar_en == "Procedimiento":
            procs = graph.get_procedure(texto)
            if procs:
                result["procedimientos"].extend(procs[:3])

        # Buscar relaciones genéricas para cualquier entidad
        if buscar_en in ("Patologia", "Farmaco", "EstructuraAnatomica", "Procedimiento",
                         "Signo", "Sintoma", "MetodoDx", "Agente"):
            try:
                rels = graph.query(f"""
                MATCH (e:{buscar_en})-[r]-(related)
                WHERE toLower(e.nombre) CONTAINS toLower($term)
                AND NOT related:Chunk AND NOT related:ParentChunk
                RETURN e.nombre AS desde, type(r) AS relacion, related.nombre AS hasta,
                       labels(related)[0] AS tipo_hasta
                LIMIT 15
                """, {"term": texto})
                for r in rels:
                    result["relaciones"].append({
                        "desde": r["desde"], "relacion": r["relacion"],
                        "hasta": r["hasta"], "tipo_hasta": r["tipo_hasta"]
                    })
            except Exception:
                pass

    return result


def execute_bibliography(analysis: dict, top_k: int = 20) -> dict:
    """Búsqueda híbrida multi-query: cada sub-query busca un aspecto del tema."""
    import logging

    sub_queries = analysis.get("sub_queries", [analysis["original"]])
    if not sub_queries:
        sub_queries = [analysis["original"]]

    # Limitar a 5 sub-queries max para no explotar latencia
    sub_queries = sub_queries[:5]

    all_results = {}  # id -> chunk (dedup)
    total_keyword = 0
    total_semantic = 0

    for sq in sub_queries:
        try:
            # Cada sub-query busca un pool amplio para maximizar recall
            per_query_k = max(20, top_k)
            result = vector.search_hybrid(sq, top_k=per_query_k)

            total_keyword += result.get("keyword_count", 0)
            total_semantic += result.get("semantic_count", 0)

            for chunk in result.get("results", []):
                cid = chunk.get("id", "")
                if cid not in all_results:
                    all_results[cid] = chunk
                else:
                    # Si ya existe, sumar score (boost por aparecer en múltiples sub-queries)
                    all_results[cid]["rrf_score"] = all_results[cid].get("rrf_score", 0) + chunk.get("rrf_score", 0)

        except Exception as e:
            logging.error(f"Bibliography sub-query failed '{sq}': {type(e).__name__}: {str(e)[:80]}")

    # Ordenar por score acumulado y tomar top_k
    sorted_results = sorted(all_results.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
    final = sorted_results[:top_k]

    return {
        "keyword_count": total_keyword,
        "semantic_count": total_semantic,
        "sub_queries_ejecutadas": len(sub_queries),
        "chunks_unicos_encontrados": len(all_results),
        "intencion": analysis.get("intencion", "general"),
        "query_expandida": analysis.get("expandida", ""),
        "results": final,
    }


def execute_activities(analysis: dict) -> dict:
    """Busca actividades académicas (TPs, seminarios, talleres).

    Siempre busca usando TODAS las entidades detectadas (no solo las
    clasificadas como actividad_academica), porque el analyzer raramente
    clasifica entidades como actividades.
    """
    result = {"actividades": [], "material": []}
    existing_ids = set()

    def _add_activities(acts):
        for act in (acts or [])[:5]:
            aid = act.get("id")
            if aid and aid not in existing_ids:
                existing_ids.add(aid)
                result["actividades"].append(act)
                try:
                    mat = graph.get_activity_material(aid)
                    if mat:
                        result["material"].extend(mat)
                except Exception:
                    pass

    # 1. Buscar por cada entidad detectada (todas, no solo actividades)
    for ent in analysis.get("entidades_detectadas", []):
        texto = ent.get("texto", "")
        if texto and len(texto) >= 3:
            _add_activities(graph.get_activity(texto))

    # 2. Buscar por sub_queries
    for sq in analysis.get("sub_queries", []):
        _add_activities(graph.get_activity(sq))

    # 3. Buscar por palabras individuales del query original (fallback)
    if not result["actividades"]:
        pregunta = analysis.get("original", "")
        words = [w for w in pregunta.lower().split() if len(w) >= 4]
        for word in words[:6]:
            _add_activities(graph.get_activity(word))
            if result["actividades"]:
                break

    return result


def execute_dags(analysis: dict) -> dict:
    """Busca flujos clínicos (PATHWAY + CLINICAL) para entidades detectadas."""
    result = {"pathways": [], "clinical": []}

    search_terms = set()
    for ent in analysis.get("entidades_detectadas", []):
        search_terms.add(ent.get("texto", ""))

    for term in search_terms:
        if not term:
            continue

        # Pathways
        try:
            pathways = graph.query("""
            MATCH (a)-[r:PATHWAY]->(b)
            WHERE toLower(a.nombre) CONTAINS toLower($term)
               OR toLower(b.nombre) CONTAINS toLower($term)
               OR r.nombre_dag CONTAINS toLower($term)
            RETURN DISTINCT r.nombre_dag AS dag, a.nombre AS desde,
                   b.nombre AS hasta, r.orden AS orden, r.nota AS nota
            ORDER BY r.nombre_dag, r.orden
            """, {"term": term})

            dags = {}
            for p in pathways:
                dag_name = p["dag"]
                if dag_name not in dags:
                    dags[dag_name] = {"nombre": dag_name, "pasos": []}
                dags[dag_name]["pasos"].append({
                    "orden": p["orden"], "desde": p["desde"],
                    "hasta": p["hasta"], "nota": p.get("nota")
                })
            result["pathways"].extend(dags.values())
        except Exception:
            pass

        # Clinical
        try:
            steps = graph.query("""
            MATCH (a)-[r:CLINICAL]->(b)
            WHERE toLower(a.nombre) CONTAINS toLower($term)
               OR toLower(b.nombre) CONTAINS toLower($term)
               OR r.nombre_dag CONTAINS toLower($term)
            RETURN DISTINCT r.nombre_dag AS dag, a.nombre AS desde,
                   b.nombre AS hasta, r.orden AS orden,
                   r.tipo_paso AS tipo_paso, r.condicion AS condicion, r.nota AS nota
            ORDER BY r.nombre_dag, r.orden, r.condicion
            """, {"term": term})

            dags = {}
            for s in steps:
                dag_name = s["dag"]
                if dag_name not in dags:
                    dags[dag_name] = {"nombre": dag_name, "pasos": []}
                dags[dag_name]["pasos"].append({
                    "orden": s["orden"], "desde": s["desde"],
                    "hasta": s["hasta"], "tipo_paso": s.get("tipo_paso"),
                    "condicion": s.get("condicion"), "nota": s.get("nota")
                })
            result["clinical"].extend(dags.values())
        except Exception:
            pass

    return result
