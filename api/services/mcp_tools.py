"""Las siete tools de `mcp/v1` sobre lo que ingesto el pipeline (17-sep-2026).

QUE ES ESTE ARCHIVO. La capa de dominio del MCP remoto: `routers/mcp_remote.py` solo declara firmas,
modelos de salida y descripciones; lo que consulta el grafo esta aca. Contrato: `mcp/v1` de
nomos-contracts (`mcp-v1.md` + `schemas/mcp-tools.schema.json`, copia vendorizada en
`tests/contracts/`).

ES LA CUARTA IMPLEMENTACION DEL CONTRATO, y es la que lo prueba: las otras tres las escribio quien
escribio el contrato. Si un tercero puede servir su propio grafo como Nomos Graph clonando este
repo, el contrato es un contrato; si hiciera falta conocimiento privado, era un acuerdo entre dos.
Es el mismo argumento que `services/admin_v1.py` hace para `admin/v1`.

NO DUPLICA RETRIEVAL. Las primitivas son las que este repo ya tenia: `services/vector.py` (la
busqueda hibrida con filtros, garantia, agrupacion y procedencia), `services/analyzer.py` +
`services/layers.py` (el router de `POST /query`), `services/admin_v1.py` (el catalogo y las stats)
y `services/graph.py` (Cypher predefinido, nunca una consulta abierta).

LO QUE ESTA INSTANCIA NO PUBLICA, dicho donde se ve:
  · **`evidence`**: es una CAPACIDAD OPCIONAL del contrato (§3.8) y exige una fuente EXTERNA. Este
    repo no trae ningun adaptador: la instancia privada le pregunta a PubMed, y eso es de su dominio
    y de su cuota. Siete tools cumplen el contrato entero.
  · **una URL por unidad**: no hay ruta de lectura de un chunk (`capabilities.units` es false), asi
    que `url` apunta a la FUENTE en `admin/v1`, que es la direccion que esta instancia si puede
    servir. Un link que da 404 es peor que un link menos preciso; la posicion exacta viaja en
    `locator` y en `metadata`.

EL IDIOMA. Los comentarios van en castellano, como el resto de `api/` y de `pipeline/`; el TEXTO de
las descripciones de las tools va en INGLES, como el README y todo lo que mira quien clona el repo:
esas descripciones son la superficie del producto y las lee un modelo que puede estar hablando
cualquier idioma. El contrato fija el largo minimo de una descripcion, no su idioma (§7).
"""
import logging

from services import admin_v1, telemetria, vector
from services.autorizacion import AdminError
from services.graph import query as cypher
from services.settings import get_settings

log = logging.getLogger(__name__)

#: Tope de resultados de una tool: el mismo que `/search/*` (`vector.TOPE_TOP_K`). Tenerlo escrito
#: dos veces seria la forma de que un dia el REST deje pasar 50 y el MCP 20 sin que nadie lo decida.
MAX_RESULTS = vector.TOPE_TOP_K
DEFAULT_RESULTS = 10
SNIPPET_CHARS = 320
#: Cuanto texto devuelve `fetch`. Es defensivo: una tool no deberia devolver un documento entero, y
#: varios clientes truncan la respuesta sin avisar. `metadata.truncated` lo dice cuando pasa.
MAX_TEXT_CHARS = 12000
#: Cuanto texto de cada unidad viaja en un `deep_dive`: son decenas de unidades en una respuesta.
DEEP_DIVE_MAX_CHARS = 1100

#: `deep_dive` agrupa por padre POR DEFECTO, al reves que `search` (contrato §3.6): un documento se
#: escribe con pasajes enteros y no con cinco recortes de la misma pagina. La ventana del padre ya
#: esta en el grafo (`:ParentChunk`, la escribe `pipeline/carga.py`) y no cuesta nada.
AGRUPACION_DEEP_DIVE = "padre"

# ---------------------------------------------------------------------------
#  Avisos: lo que lee el MODELO cuando el corpus no se pudo consultar
# ---------------------------------------------------------------------------
# En ingles a proposito: los lee un modelo, no una persona. Los dos dicen lo mismo con distinto
# alcance, y existen por la regla 4 del contrato: una lista vacia por CAIDA del retrieval se lee
# igual que "el corpus no tiene material", y el modelo contesta de su propio conocimiento sin
# decirlo. En una herramienta cuya promesa es "cita fuente y pagina", eso es un fallo de CORRECCION.
AVISO_TOTAL = (
    "WARNING: the corpus could not be queried (retrieval failure), so this empty result does NOT "
    "mean the corpus lacks material on this topic. Do not answer as if the sources had been "
    "consulted: say the corpus could not be reached, and retry in a couple of minutes.")
AVISO_PARCIAL = (
    "NOTE: one of the retrieval paths failed, so these results are less complete than usual. Treat "
    "them as partial.")


def _aviso(degradado: str | None) -> dict:
    """`{degraded, notice}` a partir de lo que reporta `vector.search_hybrid`.

    Un aviso que aparece siempre deja de leerse: sin degradacion, `degraded` es False y `notice` es
    None.
    """
    if degradado == "total":
        return {"degraded": True, "notice": AVISO_TOTAL}
    if degradado == "parcial":
        return {"degraded": True, "notice": AVISO_PARCIAL}
    return {"degraded": False, "notice": None}


def _base() -> str:
    return get_settings().public_base_url.rstrip("/")


def _url_de_fuente(libro_id: str | None) -> str:
    """La direccion de la FUENTE en `admin/v1`, que es lo que esta instancia puede servir.

    NO ES LA URL DE LA UNIDAD, y el contrato lo admite ("una `url` citable si existe, si no la ruta
    que la instancia tenga"): `capabilities.units` es false en este repo --no hay endpoint que
    muestre un chunk-- y publicar una URL inventada por unidad daria 404 en el primer click. La
    posicion exacta del pasaje viaja en `locator` y en `metadata` (pagina, capitulo, seccion).
    """
    return f"{_base()}/admin/v1/sources/{libro_id or ''}"


def _locator(fila: dict) -> str | None:
    """La posicion citable de un pasaje: "p. 604", "pp. 604-605". Sin paginas, None."""
    p0, p1 = fila.get("pag_inicio"), fila.get("pag_fin")
    if not p0:
        return None
    return f"p. {p0}" if (not p1 or p1 == p0) else f"pp. {p0}-{p1}"


def _titulo(fila: dict) -> str:
    """Titulo legible: fuente — ubicacion — pagina. Es el orden con que se cita."""
    partes = [fila.get("libro_titulo") or fila.get("libro") or "Source"]
    ubicacion = fila.get("seccion") or fila.get("capitulo")
    if ubicacion:
        partes.append(str(ubicacion).strip())
    loc = _locator(fila)
    if loc:
        partes.append(loc)
    return " — ".join(partes)


# ---------------------------------------------------------------------------
#  search / fetch
# ---------------------------------------------------------------------------
def _resultado(c: dict) -> dict:
    texto = (c.get("texto") or "").strip()
    return {
        "id": c.get("id"),
        "title": _titulo(c),
        "url": _url_de_fuente(c.get("libro")),
        "snippet": texto[:SNIPPET_CHARS] + ("…" if len(texto) > SNIPPET_CHARS else ""),
        "source_id": c.get("libro"),
        "locator": _locator(c),
        "page_start": c.get("pag_inicio"),
        "page_end": c.get("pag_fin"),
        # `null` = NO MEDIDA, que el contrato distingue de "ok" (§3.1). La escribe el parseo por
        # unidad; un corpus cargado con una version anterior la trae vacia, y eso no es "esta bien".
        "calidad": c.get("calidad"),
        "agrupado_por": c.get("agrupado_por"),
        "unidades": c.get("unidades") or [],
    }


def search(query: str, limit: int = DEFAULT_RESULTS, filtros: dict | None = None,
           garantizar: dict | None = None, agrupar: str | None = None) -> dict:
    """Descubrir unidades del corpus. Contrato `mcp/v1` §3.1.

    Reusa la busqueda hibrida de `services/vector.py` (BM25 + KNN + RRF + sub-queries), la misma que
    sirve `POST /search/hybrid`. Devuelve ids estables, resolubles con `fetch`.

    Una consulta vacia o sin coincidencias devuelve lista vacia y `degraded: False`, NO un error. Si
    el retrieval se cayo, la lista vacia viaja con `degraded: True` y el `notice` que lo explica: es
    lo unico que impide que el modelo lea la caida como una ausencia.
    """
    q = (query or "").strip()
    limit = max(1, min(int(limit or DEFAULT_RESULTS), MAX_RESULTS))
    if not q:
        return {"results": [], "degraded": False, "notice": None, "procedencia": {}}
    try:
        r = vector.search_hybrid(q, top_k=limit, filtros=filtros, garantizar=garantizar,
                                 agrupar=agrupar)
    except Exception as e:
        # DEGRADA, NO REVIENTA. `search_hybrid` LANZA cuando todas sus rutas cayeron y al menos una
        # era del grafo (asi `/search/hybrid` puede contestar 503). Una tool de MCP no tiene codigo
        # de estado: si la excepcion escapara, el cliente veria un error sin explicacion; y si se
        # tragara en silencio, veria una lista vacia. Sale como aviso.
        log.error("mcp search fallo: %s: %s", type(e).__name__, str(e)[:120])
        telemetria.registrar("mcp_degradado", tool="search", capa="busqueda_hibrida",
                             error=type(e).__name__)
        return {"results": [], "procedencia": {}, **_aviso("total")}
    return {"results": [_resultado(c) for c in r.get("results", []) if c.get("id")],
            "procedencia": r.get("procedencia") or {},
            **_aviso(r.get("degradado"))}


#: `fetch` abre UNA unidad por su id. El engine no tiene ruta de lectura de unidades
#: (`capabilities.units` es false), asi que el Cypher vive aca --predefinido, como todo lo de
#: `services/graph.py`: en un grafo expuesto a un chat no entra una consulta abierta--.
#: `OPTIONAL MATCH` sobre `:Book` porque una fuente puede existir SOLO como `libro_id` en los chunks
#: (quien llama a `carga.cargar_libro` y no a `carga.registrar_fuente`): ahi el titulo y el autor
#: vienen en null y el pasaje se abre igual.
Q_UNIDAD = """
MATCH (c:Chunk {id: $id})
OPTIONAL MATCH (b:Book {id: c.libro_id})
RETURN c.id AS id, c.libro_id AS libro, c.text AS texto, c.word_count AS palabras,
       c.page_start AS pag_inicio, c.page_end AS pag_fin, c.chunk_index AS indice,
       c.titulo_capitulo AS capitulo, c.titulo_seccion AS seccion,
       c.tipo_contenido AS tipo, c.parent_id AS parent_id, c.calidad AS calidad,
       c.embedding IS NOT NULL AS tiene_embedding,
       b.title AS libro_titulo, b.author AS autor, b.edition AS edicion
LIMIT 1
"""


def fetch(id: str) -> dict:
    """Abrir UNA unidad por su id. Contrato §3.2.

    Un id que no existe levanta `ValueError` -> el transporte lo traduce a un error tipado de tool,
    que es lo que el contrato pide: NO un texto vacio, que el modelo leeria como "esa fuente no dice
    nada".
    """
    uid = (id or "").strip()
    if not uid:
        raise ValueError("The 'id' parameter is required.")
    filas = cypher(Q_UNIDAD, {"id": uid})
    if not filas:
        raise ValueError(f"No unit with id '{uid}' in this graph. Ids come from a previous search "
                         "call: they are not constructed by hand.")
    c = filas[0]
    texto = c.get("texto") or ""
    return {
        "id": c["id"],
        "title": _titulo(c),
        "text": texto[:MAX_TEXT_CHARS],
        "url": _url_de_fuente(c.get("libro")),
        # METADATA = PROCEDENCIA, y nada de operacion: ni tiempos, ni el Cypher, ni identificadores
        # internos (regla 5 del contrato).
        "metadata": {
            "source_id": c.get("libro"),
            "source_title": c.get("libro_titulo"),
            "authors": [c["autor"]] if c.get("autor") else [],
            "edition": c.get("edicion"),
            "chapter": c.get("capitulo"),
            "section": c.get("seccion"),
            "locator": _locator(c),
            "page_start": c.get("pag_inicio"),
            "page_end": c.get("pag_fin"),
            "index": c.get("indice"),
            "type": c.get("tipo"),
            "word_count": c.get("palabras"),
            "calidad": c.get("calidad"),
            "embedded": bool(c.get("tiene_embedding")),
            "truncated": len(texto) > MAX_TEXT_CHARS,
        },
    }


# ---------------------------------------------------------------------------
#  list_sources
# ---------------------------------------------------------------------------
def _fuente_publicada(f: dict) -> dict:
    """AdminSource -> el item del catalogo que ve el modelo.

    SE PROYECTA Y NO SE REENVIA ENTERO: `AdminSource` lleva `lineage`, `active_job_id` y timestamps
    de operacion, que son de la consola y no del modelo. Queda el minimo del contrato (§3.3): el id
    que va en `filtros`, un titulo legible, `colecciones` y los numeros de la fuente.
    """
    campos = f.get("fields") or {}
    stats = f.get("stats") or {}
    return {
        "id": f["id"],
        "titulo": f.get("title") or f["id"],
        "autor": campos.get("autor"),
        "edicion": campos.get("edicion"),
        # ESTE REPO NO GUARDA COLECCIONES en `:Book` (`carga.registrar_fuente` escribe los cuatro
        # campos que publica `SOURCE_FIELDS` y nada mas), asi que la lista viaja vacia en vez de
        # inventar una. El contrato pide la clave porque es el valor que se pasa a `filtros`; el dia
        # que una instancia las escriba, salen solas.
        "colecciones": campos.get("colecciones") or [],
        "unidades": stats.get("units"),
        "unidades_vectorizadas": stats.get("units_embedded"),
        "paginas": stats.get("pages"),
        "estado": f.get("status"),
        "detalle": f.get("status_detail"),
    }


def list_sources(autor: str = "", q: str = "") -> dict:
    """El catalogo: que fuentes hay y de que tratan. Contrato §3.3.

    CERO ARGUMENTOS OBLIGATORIOS y los dos ESCALARES: un catalogo que exige un filtro no es un
    catalogo, y un argumento que acepta un objeto o una lista es la puerta por la que una tool pide
    "el contexto de la conversacion". Los filtros son los campos que ESTA instancia declara
    filtrables en `descriptor.source_fields`.
    """
    try:
        filtros = {"autor": autor} if autor else {}
        listado = admin_v1.listar_fuentes(q=q or None, filtros=filtros, limit=admin_v1.LIMITE_MAX)
        fuentes = [_fuente_publicada(f) for f in listado["items"]]
        stats = admin_v1.stats()
    except AdminError as e:
        # Un filtro que no existe es un error del cliente y se dice como tal, con los nombres
        # correctos adentro: ignorarlo devolveria el catalogo entero como si el filtro se hubiera
        # aplicado.
        raise ValueError(e.detail) from e
    except Exception as e:
        log.error("mcp list_sources fallo: %s", type(e).__name__)
        telemetria.registrar("mcp_degradado", tool="list_sources", capa="catalogo",
                             error=type(e).__name__)
        return {"stats": {}, "sources": [], "count": 0, "degraded": True, "notice": AVISO_TOTAL}
    return {"stats": stats, "sources": fuentes, "count": len(fuentes),
            "degraded": False, "notice": None}


# ---------------------------------------------------------------------------
#  grafo de entidades
# ---------------------------------------------------------------------------
#: La proyeccion de un nodo del grafo de conceptos. `coalesce(n.nombre, n.name)` porque el extractor
#: escribe `nombre` y algunos cargadores escriben `name`.
CAMPOS_NODO = """elementId(n) AS id, coalesce(n.nombre, n.name) AS label,
       labels(n)[0] AS type, coalesce(n.freq, 0) AS freq,
       COUNT { (n)--() } AS degree"""

#: `:Book` SE EXCLUYE ademas de los chunks: una fuente no es un concepto del grafo, y devolverla por
#: `search_entities` mandaria a `expand_concept` a expandir un libro entero. Las fuentes son
#: `list_sources`, que ademas devuelve el id que aceptan los filtros.
Q_ENTIDADES = f"""
CALL db.index.fulltext.queryNodes('entity_search', $q)
YIELD node AS n, score
WHERE NOT (n:Chunk OR n:ParentChunk OR n:Book)
RETURN {CAMPOS_NODO}, score
ORDER BY score DESC LIMIT $limit
"""

#: El fallback cuando el indice `entity_search` no existe --que es el caso de un grafo recien
#: cargado: ningun script de este repo lo crea--. CONTAINS sobre el nombre es mas pobre y no rompe.
Q_ENTIDADES_SIN_INDICE = f"""
MATCH (n)
WHERE NOT (n:Chunk OR n:ParentChunk OR n:Book)
  AND toLower(coalesce(n.nombre, n.name, '')) CONTAINS toLower($q)
RETURN {CAMPOS_NODO}, 0.0 AS score
ORDER BY freq DESC LIMIT $limit
"""

Q_VECINOS = """
MATCH (n)-[r]-(m)
WHERE elementId(n) = $id AND NOT (m:Chunk OR m:ParentChunk)
  AND ($relaciones IS NULL OR type(r) IN $relaciones)
WITH r, m ORDER BY coalesce(m.freq, 0) DESC
LIMIT $limit
RETURN elementId(r) AS rel_id, type(r) AS rel_type,
       elementId(startNode(r)) AS source, elementId(endNode(r)) AS target,
       elementId(m) AS m_id, coalesce(m.nombre, m.name) AS m_label,
       labels(m)[0] AS m_type, coalesce(m.freq, 0) AS m_freq,
       COUNT { (m)--() } AS m_degree
"""


def search_entities(query: str, limit: int = 20) -> dict:
    """Entidades tipadas del grafo por nombre. Contrato §3.4.

    Devuelve los ids que consume `expand_concept`. Los ids NO se construyen: salen de aca.
    """
    q = (query or "").strip()
    limit = max(1, min(int(limit or 20), 50))
    if not q:
        return {"query": "", "entities": [], "count": 0, "degraded": False, "notice": None}
    try:
        filas = cypher(Q_ENTIDADES, {"q": q, "limit": limit})
    except Exception:
        # El indice fulltext puede no existir: no es una caida del grafo, es un grafo sin ese
        # indice. Se cae al CONTAINS y, si ESE falla, ahi si no se pudo consultar.
        try:
            filas = cypher(Q_ENTIDADES_SIN_INDICE, {"q": q, "limit": limit})
        except Exception as e:
            log.error("mcp search_entities fallo: %s", type(e).__name__)
            telemetria.registrar("mcp_degradado", tool="search_entities", capa="grafo",
                                 error=type(e).__name__)
            return {"query": q, "entities": [], "count": 0, "degraded": True,
                    "notice": AVISO_TOTAL}
    return {"query": q, "entities": filas, "count": len(filas),
            "degraded": False, "notice": None}


def expand_concept(entity_id: str, relation_types: list | None = None, limit: int = 20) -> dict:
    """Los vecinos de UNA entidad, UN salto. Contrato §3.5.

    UN SALTO Y NO SE PIDE POR PARAMETRO: dos saltos sobre un grafo de cientos de miles de entidades
    es una respuesta que ningun cliente puede leer. `relation_types` filtra por TIPO DE RELACION
    --los del perfil activo, `descriptor.relation_types`-- y no por label del vecino: son dos cosas
    distintas, y pasar una por la otra devuelve cero vecinos sin decir por que.
    """
    eid = (entity_id or "").strip()
    if not eid:
        raise ValueError("The 'entity_id' parameter is required.")
    limit = max(1, min(int(limit or 20), 200))
    relaciones = [str(t).strip() for t in (relation_types or []) if str(t).strip()] or None

    centro = cypher(f"MATCH (n) WHERE elementId(n) = $id RETURN {CAMPOS_NODO}", {"id": eid})
    if not centro:
        raise ValueError(f"Entity not found: {eid}. Use search_entities to get valid ids.")
    filas = cypher(Q_VECINOS, {"id": eid, "limit": limit, "relaciones": relaciones})

    nodos = {centro[0]["id"]: centro[0]}
    aristas, vistas = [], set()
    for r in filas:
        if r["m_id"] not in nodos:
            nodos[r["m_id"]] = {"id": r["m_id"], "label": r["m_label"], "type": r["m_type"],
                                "freq": r["m_freq"], "degree": r["m_degree"]}
        if r["rel_id"] not in vistas:
            vistas.add(r["rel_id"])
            aristas.append({"id": r["rel_id"], "source": r["source"], "target": r["target"],
                            "type": r["rel_type"]})
    return {"center": eid, "nodes": list(nodos.values()), "edges": aristas,
            "count": {"nodes": len(nodos), "edges": len(aristas)}}


# ---------------------------------------------------------------------------
#  deep_dive
# ---------------------------------------------------------------------------
#: LAS FACETAS POR DEFECTO, POR IDIOMA DEL PERFIL. Son los ejes con que se lee un documento de
#: cualquier dominio --que es lo que este repo puede prometer--: la instancia privada usa la
#: estructura de un capitulo de su disciplina, y eso no se puede escribir una vez para todos.
#:
#: POR QUE POR IDIOMA Y NO POR DOMINIO: los terminos de cada faceta entran al retrieval, asi que
#: tienen que estar en el idioma del CORPUS o no encuentran nada. El perfil ya lo declara
#: (`profiles/<dominio>.yaml`, clave `language`). Si un perfil declara un idioma que no esta en esta
#: tabla, se usa el NOMBRE de la faceta como consulta: peor, pero honesto y en el idioma correcto.
#: Quien tenga un vocabulario mejor para su dominio lo pasa en `facets`, que acepta ejes libres.
FACETAS_POR_IDIOMA = {
    "es": [
        ("definicion", "definicion concepto que es"),
        ("contexto", "antecedentes contexto historia origen"),
        ("mecanismo", "mecanismo funcionamiento proceso causas"),
        ("clasificacion", "clasificacion tipos formas variantes"),
        ("procedimiento", "procedimiento pasos metodo protocolo"),
        ("criterios", "criterios requisitos condiciones indicaciones"),
        ("riesgos", "riesgos limitaciones complicaciones advertencias"),
        ("evidencia", "evidencia datos resultados estudios"),
    ],
    "en": [
        ("definition", "definition concept what is"),
        ("context", "background context history origin"),
        ("mechanism", "mechanism how it works process causes"),
        ("classification", "classification types forms variants"),
        ("procedure", "procedure steps method protocol"),
        ("criteria", "criteria requirements conditions indications"),
        ("risks", "risks limitations complications warnings"),
        ("evidence", "evidence data results studies"),
    ],
}


def facetas_por_defecto() -> list:
    """Las facetas de esta instancia, en el idioma que declara su perfil activo."""
    idioma = str(admin_v1.perfil().get("language") or "").strip().lower()[:2]
    tabla = FACETAS_POR_IDIOMA.get(idioma)
    if tabla:
        return list(tabla)
    # Sin vocabulario para ese idioma: el nombre de la faceta ES la consulta. Se usan los nombres
    # neutros en ingles porque son los unicos que este repo tiene escritos, y se dice en la
    # descripcion de la tool para que nadie lo lea como una promesa de cobertura.
    return [(n, n) for n, _ in FACETAS_POR_IDIOMA["en"]]


def _facetas(facets: list | None) -> list:
    """Las facetas a correr: las por defecto, un subconjunto, o las libres que pida el cliente."""
    por_defecto = facetas_por_defecto()
    if not facets:
        return por_defecto
    pedidas = {str(f).strip().lower() for f in facets if str(f).strip()}
    if not pedidas:
        return por_defecto
    conocidas = [(n, t) for n, t in por_defecto if n in pedidas]
    libres = [(f, f) for f in sorted(pedidas) if f not in {n for n, _ in por_defecto}]
    return conocidas + libres or por_defecto


def _unidad_de_faceta(c: dict) -> dict:
    texto = (c.get("texto") or "").strip()
    return {
        "id": c.get("id"),
        "source": c.get("libro"),
        "title": _titulo(c),
        "url": _url_de_fuente(c.get("libro")),
        "locator": _locator(c),
        "page_start": c.get("pag_inicio"),
        "page_end": c.get("pag_fin"),
        "section": c.get("seccion") or c.get("capitulo"),
        "text": texto[:DEEP_DIVE_MAX_CHARS],
        "truncated": len(texto) > DEEP_DIVE_MAX_CHARS,
        "calidad": c.get("calidad"),
        "agrupado_por": c.get("agrupado_por"),
        "unidades": c.get("unidades") or [],
    }


def deep_dive(topic: str, facets: list | None = None, per_facet: int = 6,
              include_graph: bool = True, filtros: dict | None = None,
              garantizar: dict | None = None,
              agrupar: str | None = AGRUPACION_DEEP_DIVE) -> dict:
    """TODO lo que el corpus tiene sobre un tema, agrupado por faceta. Contrato §3.6.

    Corre una busqueda hibrida por faceta EN PARALELO y deduplica entre facetas: cada unidad aparece
    UNA vez, bajo la faceta que la trajo. Que se caiga UNA faceta es un hueco; que se caigan TODAS es
    no haber consultado el corpus, y se dicen distinto.
    """
    from concurrent.futures import ThreadPoolExecutor

    tema = (topic or "").strip()
    if not tema:
        return {"topic": "", "facets": {}, "total_passages": 0, "sources_used": [],
                "graph_relations": [], "degraded": False, "notice": None, "procedencia": {}}
    seleccion = _facetas(facets)
    per_facet = max(2, min(int(per_facet or 6), 12))
    caidas: list = []
    procedencias: list = []

    def buscar(par):
        nombre, terminos = par
        try:
            r = vector.search_hybrid(f"{tema} {terminos}", top_k=per_facet, filtros=filtros,
                                     garantizar=garantizar, agrupar=agrupar)
        except Exception as e:
            log.error("deep_dive faceta '%s' fallo: %s", nombre, type(e).__name__)
            telemetria.registrar("mcp_degradado", tool="deep_dive", capa=f"faceta:{nombre}",
                                 error=type(e).__name__)
            caidas.append(nombre)
            return nombre, []
        if r.get("degradado") == "total":
            # "SIN RESULTADOS" Y "NO SE PUDO BUSCAR" NO SON LO MISMO. Sin esto, la faceta se cuenta
            # como vacia, el informe sale sin una sola cita y sin marcar `degraded`.
            telemetria.registrar("mcp_degradado", tool="deep_dive", capa=f"faceta:{nombre}",
                                 error="retrieval_caido")
            caidas.append(nombre)
            return nombre, []
        # `.append` sobre una lista desde varios hilos es atomico en CPython: no hace falta lock
        # para juntar unos dicts chicos.
        procedencias.append(r.get("procedencia") or {})
        return nombre, r.get("results", [])

    with ThreadPoolExecutor(max_workers=min(4, len(seleccion))) as pool:
        crudos = list(pool.map(buscar, seleccion))

    vistos: set = set()
    salida: dict = {}
    total = 0
    for nombre, resultados in crudos:
        items = []
        for c in resultados:
            cid = c.get("id")
            if not cid or cid in vistos:
                continue
            vistos.add(cid)
            items.append(_unidad_de_faceta(c))
        if items:
            salida[nombre] = items
            total += len(items)

    resultado = {
        "topic": tema,
        "facets": salida,
        "total_passages": total,
        "sources_used": sorted({i["source"] for f in salida.values() for i in f if i.get("source")}),
        "graph_relations": [],
        "procedencia": _procedencia_del_informe(salida, procedencias),
        "degraded": False,
        "notice": None,
    }
    if caidas:
        resultado["degraded"] = True
        resultado["notice"] = (AVISO_TOTAL if len(caidas) == len(seleccion) else
                               f"NOTE: {len(caidas)} of {len(seleccion)} facets could not be "
                               f"retrieved ({', '.join(caidas)}), so this summary has gaps that are "
                               "NOT absences in the corpus.")
    if include_graph:
        resultado["graph_relations"] = _relaciones_del_tema(tema)
    return resultado


def _procedencia_del_informe(facetas: dict, procedencias: list) -> dict:
    """Junta la `procedencia` de las facetas en una del informe entero.

    `top_k` se recuenta sobre lo que sobrevivio al dedup global --que es lo que el redactor tiene
    delante-- y no sumando los de cada faceta, que contarian de mas. `candidatos_unicos` es la SUMA
    de los pools por faceta y NO un conteo global: son N busquedas distintas y deduplicarlas
    costaria guardar miles de ids para un numero informativo. Dicho aca para que nadie lo lea como
    "unidades distintas del corpus".
    """
    top: dict = {}
    for items in facetas.values():
        for i in items:
            if i.get("source"):
                top[i["source"]] = top.get(i["source"], 0) + 1
    pool: dict = {}
    garantizados = unicos = 0
    for p in procedencias:
        for libro, n in (p.get("pool") or {}).items():
            pool[libro] = pool.get(libro, 0) + n
        garantizados += p.get("garantizados") or 0
        unicos += p.get("candidatos_unicos") or 0
    return {"top_k": dict(sorted(top.items(), key=lambda kv: (-kv[1], kv[0]))),
            "pool": dict(sorted(pool.items(), key=lambda kv: (-kv[1], kv[0]))),
            "garantizados": garantizados, "candidatos_unicos": unicos}


def _etiqueta(subgrafo: dict, node_id: str) -> str:
    for n in subgrafo.get("nodes", []):
        if n.get("id") == node_id:
            return n.get("label") or node_id
    return node_id


def _relaciones_del_tema(tema: str) -> list:
    """Relaciones GUARDADAS del grafo alrededor del tema. Nunca lanza.

    Son evidencia del grafo y se informan aparte de las unidades del corpus: una relacion extraida
    por un LLM no es una cita de la fuente.
    """
    try:
        entidades = search_entities(tema, limit=3).get("entities") or []
    except Exception as e:
        log.warning("deep_dive grafo fallo: %s", type(e).__name__)
        return []
    relaciones = []
    for e in entidades[:2]:
        try:
            sub = expand_concept(e["id"], limit=12)
        except Exception:
            continue
        relaciones += [{"from": _etiqueta(sub, x["source"]), "relation": x["type"],
                        "to": _etiqueta(sub, x["target"])} for x in sub.get("edges", [])]
    return relaciones[:25]


# ---------------------------------------------------------------------------
#  unified_query
# ---------------------------------------------------------------------------
def unified_query(question: str, top_k: int = 8, filtros: dict | None = None,
                  garantizar: dict | None = None, agrupar: str | None = None) -> dict:
    """El router completo de la instancia sobre una pregunta. Contrato §3.7.

    Es el mismo camino que `POST /query`: `services/analyzer.py` clasifica la pregunta, la expande y
    la descompone en sub-consultas, y `services/layers.execute_bibliography` las corre todas contra
    el corpus y fusiona. `layers_activated` dice QUE capas eligio el analizador, que es lo unico que
    permite distinguir una respuesta pobre de una capa que no corrio.

    EL ANALIZADOR NECESITA UN PROVEEDOR (`GCP_API_KEY`) y puede no estar: `analyze_query` ya tiene
    su fallback determinista, y si ni eso contesta se busca la pregunta TAL CUAL y se dice en
    `notice`. Nunca un error ni una lista vacia muda.
    """
    from services import layers
    from services.analyzer import analyze_query

    q = (question or "").strip()
    top_k = max(1, min(int(top_k or 8), MAX_RESULTS))
    if not q:
        return {"question": "", "intent": None, "layers_activated": [], "sub_queries": [],
                "bibliography": {"count": 0, "results": []}, "degraded": False, "notice": None,
                "procedencia": {}}

    aviso_router = None
    try:
        analisis = analyze_query(q)
    except Exception as e:
        log.error("mcp unified_query: el analizador no contesto: %s", type(e).__name__)
        telemetria.registrar("mcp_degradado", tool="unified_query", capa="analyzer",
                             error=type(e).__name__)
        analisis = {"original": q, "sub_queries": [q], "intencion": None, "expandida": q,
                    "capas": ["BIBLIOGRAPHY"]}
        aviso_router = ("NOTE: the query router did not answer, so the question was searched as "
                        "written: it was not classified, expanded or decomposed. The retrieved "
                        "material is real; the routing is not.")

    salida = {
        "question": q,
        "intent": analisis.get("intencion"),
        "layers_activated": analisis.get("capas", []),
        "sub_queries": analisis.get("sub_queries", []),
    }
    try:
        biblio = layers.execute_bibliography(analisis, top_k, filtros, garantizar, agrupar)
    except Exception as e:
        log.error("capa bibliografia fallo: %s", type(e).__name__)
        telemetria.registrar("mcp_degradado", tool="unified_query", capa="bibliografia",
                             error=type(e).__name__)
        # Sin corpus, esta respuesta NO tiene respaldo documental: decirlo es la diferencia entre
        # una respuesta citada y una que lo parece.
        return {**salida, "bibliography": {"count": 0, "results": []}, "procedencia": {},
                **_aviso("total")}

    salida["bibliography"] = {
        "count": len(biblio.get("results", [])),
        "results": [_resultado(c) for c in biblio.get("results", []) if c.get("id")],
    }
    salida["procedencia"] = biblio.get("procedencia") or {}
    salida.update(_aviso(biblio.get("degradado")))
    if aviso_router:
        # Se SUMA al aviso del retrieval en vez de pisarlo: son dos huecos distintos.
        salida["notice"] = " ".join(x for x in (salida.get("notice"), aviso_router) if x)
    return salida
