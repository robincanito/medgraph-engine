"""admin/v1 del engine: el mismo contrato que MedGraph y LexGraph, sin dominio adentro.

Contrato: `nomos-contracts` (openapi/admin-v1.yaml + schemas admin-descriptor/v1 y
admin-source/v1; copias para tests en tests/contracts/). La consola que administra un grafo
—cualquiera— pide `GET /admin/v1/descriptor` y de ahi saca el vocabulario, los campos, los pasos
del pipeline y las capacidades. No sabe que es un libro, ni una ley, ni un chunk.

ESTA ES LA TERCERA IMPLEMENTACION, y es la que prueba el contrato: MedGraph (medicina) y LexGraph
(derecho) las escribio quien escribio el contrato. Si un tercero puede implementarlo clonando este
repo, el contrato es un contrato; si hiciera falta conocimiento privado, era un acuerdo entre dos.

QUE LA HACE DOMAIN-AGNOSTIC, que es lo unico interesante de este archivo:

  - `domain`, `profile`, `entity_types` y `relation_types` salen del PERFIL ACTIVO
    (`profiles/<dominio>.yaml` via `pipeline/perfiles.py`), no de constantes de medicina. Cambiar
    `PROFILE=generico` en el entorno cambia el descriptor entero sin tocar una linea de codigo.
  - `labels` (libro/chunk, norma/articulo, documento/chunk) sale del perfil si este los declara y
    si no, del vocabulario neutro de abajo. El engine no le pone nombre de dominio a las cosas.
  - la FUENTE es el `:Book` que registra `pipeline/carga.registrar_fuente`, y sus unidades son los
    `:Chunk` con ese `libro_id`. Los dos los escribe el pipeline de este repo: no hay un modelo de
    grafo privado escondido aca.

SOLO LECTURA, y por eso `capabilities` publica todo en false con su nota. El contrato es explicito:
publicar una capacidad sin endpoint es mentirle a la consola. Lo que este repo NO tiene (subida
firmada, jobs, clasificador, revision humana, PATCH, DELETE, reingesta, explorador, wake) se
declara apagado, con `kind: not_built` y el camino para construirlo.
"""
import base64
from datetime import UTC, datetime

from pipeline import perfiles
from services import graph
from services.autorizacion import AdminError
from services.settings import get_settings

LIMITE_MAX = 200
#: Centinela del contrato para pedir "las fuentes con este campo VACIO": un null no viaja como
#: valor de query, y sin esto el grupo `value: null` de las facetas seria decorativo.
SIN_VALOR = "__sin_valor__"

# ---------------------------------------------------------------- vocabulario de la instancia
#: Palabras con que la consola nombra las cosas si el perfil no trae las suyas. Neutras a
#: proposito: este repo no sabe si lo que ingestaste es un tratado, una norma o un manual de
#: mantenimiento. `chunk` no es neutro pero es el nombre REAL de la unidad que produce
#: `pipeline/parseo.py`, y llamarla "fragmento" en la consola y "chunk" en el codigo seria dejar
#: dos nombres para la misma cosa.
LABELS_POR_DEFECTO = {
    "source_singular": "documento",
    "source_plural": "documentos",
    "unit_singular": "chunk",
    "unit_plural": "chunks",
}

#: name del contrato -> propiedad del `:Book`. Es la UNICA tabla que sabe como se llama cada cosa
#: en el grafo: filtros y orden pasan por aca, asi que ningun nombre de propiedad llega al Cypher
#: sin estar declarado en el descriptor.
#:
#: SON CUATRO, Y NO QUINCE, POR HONESTIDAD: son los campos que este repo escribe o documenta
#: (`carga.registrar_fuente(props)` y el `catalog.json` del que salen: id, titulo, autor, edicion,
#: paginas). Publicar `area`, `idioma` o `materia` porque la instancia privada los tiene daria una
#: tabla de columnas vacias y un filtro que solo devuelve "sin valor". Una instancia que guarde mas
#: metadata agrega su campo aca y en SOURCE_FIELDS, y la consola lo muestra sin tocar el front.
CAMPO_A_PROP = {
    "titulo": "title",
    "autor": "author",
    "edicion": "edition",
    "paginas": "pages",
}

#: TODOS `editable: false`, y no es un olvido: `capabilities.edit` es false porque no hay PATCH.
#: Un campo editable sin endpoint que lo guarde es un formulario que pierde lo que la persona
#: escribio.
SOURCE_FIELDS = [
    {"name": "titulo", "label": "Title", "type": "string", "editable": False, "required": True,
     "description": "Human-readable title. Written by carga.registrar_fuente; falls back to the id."},
    {"name": "autor", "label": "Author", "type": "string", "editable": False, "filterable": True,
     "description": "Free text, as it appears in your catalog.json entry."},
    {"name": "edicion", "label": "Edition", "type": "string", "editable": False},
    {"name": "paginas", "label": "Pages", "type": "integer", "editable": False,
     "description": "Pages of the original document; derived from the chunks if the source node does not carry it."},
]

#: EL ORDEN ES EL DE ESTE PIPELINE, Y NO EL DE LA STATE MACHINE. `pipeline/embeddings.py` embebe
#: los chunks QUE YA ESTAN EN EL GRAFO (lee por `libro_id`, escribe el vector en el nodo), asi que
#: aca se carga y despues se vectoriza. En la instancia privada es al reves porque su ingesta por
#: API vectoriza el artefacto antes de cargarlo. Los ids siguen siendo estados de job/v1 —el
#: contrato lo exige y hay un test que lo fija—: lo que cambia es el recorrido, que es justo lo que
#: `pipeline_steps` existe para declarar.
#:
#: `uploaded` y `classified` NO ESTAN: este repo no tiene subida ni clasificador (ver capabilities).
#: `pending_review` tampoco: no hay gate humano.
PIPELINE_STEPS = [
    {"id": "parsed", "label": "Parsed",
     "description": "Text per page with structure detected by the domain profile (pipeline/parseo.py)."},
    {"id": "chunked", "label": "Chunked",
     "description": "Overlapping chunks with provenance, sized by the profile's chunk block."},
    {"id": "loaded", "label": "Loaded",
     "description": "Idempotent MERGE into the graph plus orphan deletion (pipeline/carga.py)."},
    {"id": "vectorized", "label": "Vectorized",
     "description": "Embeddings under one policy, over the chunks already in the graph (pipeline/embeddings.py)."},
    {"id": "extracted", "label": "Entities",
     "description": "Entity extraction with the active profile (extract_entities.py). Optional: the graph is usable without it.",
     "optional": True},
]

#: TODO EN FALSE, y cada false lleva su nota. El engine es un pipeline con una API de consulta y
#: de administracion de SOLO LECTURA: lo que administra de verdad (subir, clasificar, revisar,
#: editar, borrar, reingestar) vive en la instancia privada, que es el servicio.
CAPABILITIES = {
    "upload": False, "classify": False, "review": False, "edit": False, "delete": False,
    "reingest": False, "explore": False, "wake": False, "facets": False, "units": False,
    "vectorize": False,
    # `mcp` ES LA UNICA EN TRUE, y no contradice lo de arriba: no es una capacidad de
    # ADMINISTRACION. Lo que habilita no es un boton de la consola sino que una persona conecte este
    # grafo desde su cliente de chat --`/mcp`, contrato `mcp/v1`, siete tools de solo lectura--. Esta
    # en `capabilities` igual porque la consola tiene que poder MOSTRARLA sin saber de dominio: lee
    # `capabilities` con `!!`, asi que un endpoint publicado bajo una capacidad que nadie declaro es
    # una puerta que la pantalla nunca muestra. Publicarla OBLIGA a llevar `auth.mcp` (el schema lo
    # exige con un if/then en las dos direcciones), y por eso no lleva nota: no esta apagada.
    "mcp": True,
}

#: POR QUE ESTA APAGADA Y COMO SE PRENDE. La consola ESCONDE el boton de lo que esta en false, asi
#: que sin esto quien administra ve una pantalla a la que le falta algo y no puede distinguir "no
#: tengo permiso" de "todavia no lo construyeron" de "aca nunca va a existir". `kind` es lo que
#: enruta (el castellano/ingles es para la persona), y `how_to_unblock` es null SOLO en
#: `domain_decision`: no es que falte escribirlo, es que no hay camino.
#:
#: EN ESTE REPO LAS ONCE SON `not_built`: nada esta apagado por decision de dominio ni por falta de
#: configuracion. El codigo no existe, y cada nota dice que habria que escribir. El texto es
#: tambien el `detail` del 403 capability_disabled que devolveria el endpoint si existiera: el muro
#: y el panel no pueden divergir porque no hay dos textos.
CAPABILITY_NOTES = {
    "upload": {
        "kind": "not_built",
        "reason": "This instance has no upload endpoint. Documents are ingested by calling the pipeline from your own script or CLI, with the file already on the machine that runs it.",
        "how_to_unblock": "Build POST /admin/v1/uploads: a signed-URL ticket against your object storage plus a confirm step, and publish accepted_formats — the descriptor schema requires it as soon as this turns true.",
    },
    "classify": {
        "kind": "not_built",
        "reason": "There is no document classifier served over the API: which profile and which source kind a document belongs to is decided by whoever runs the ingest.",
        "how_to_unblock": "Build POST /admin/v1/uploads/{job_id}/classify over your active profile. It depends on uploads existing first.",
    },
    "review": {
        "kind": "not_built",
        "reason": "There is no human gate before loading: pipeline/carga.py is an idempotent MERGE run by your own code, and nobody approves anything in between.",
        "how_to_unblock": "Build the jobs endpoints and leave jobs parked in pending_review, then POST /admin/v1/jobs/{job_id}/approve to resolve them.",
    },
    "edit": {
        "kind": "not_built",
        "reason": "PATCH does not exist, so every field is published editable:false. The source metadata is whatever your ingest passed to carga.registrar_fuente.",
        "how_to_unblock": "Build PATCH /admin/v1/sources/{id} writing only the fields declared editable in this descriptor, and flip them to editable:true in the same commit.",
    },
    "delete": {
        "kind": "not_built",
        "reason": "There is no delete endpoint. The code to do it exists and is tested (pipeline/carga.borrar_libro, which also returns the contract's DeletePreview), but nothing exposes it over HTTP.",
        "how_to_unblock": "Build DELETE /admin/v1/sources/{id} with ?dry_run, calling carga.borrar_libro and returning what it gives you. Decide first who is allowed to call it: this instance authenticates with a single API key and has no roles.",
    },
    "reingest": {
        "kind": "not_built",
        "reason": "Re-ingesting means running the pipeline again, and there is no pipeline runner behind this API: there are no jobs and no stored original file to re-read.",
        "how_to_unblock": "Build uploads and jobs first; the load itself is already idempotent by id, so a re-ingest never makes a source disappear halfway.",
    },
    "explore": {
        "kind": "not_built",
        "reason": "The graph explorer routes (/graph/search, /graph/node/{id}/neighbors) are not implemented here.",
        "how_to_unblock": "Build them over services/graph.py with predefined Cypher — never a free-form query endpoint, which is a hole in any graph exposed to a browser.",
    },
    "wake": {
        "kind": "not_built",
        "reason": "This service assumes the graph is always reachable: it never reports `sleeping` and never answers 503 warming_up.",
        "how_to_unblock": "Only relevant if you run the graph on on-demand infrastructure. Then: detect the sleeping state, start it on a real query, and answer 503 with Retry-After while it boots.",
    },
    "facets": {
        "kind": "not_built",
        "reason": "GET /admin/v1/sources/facets is not implemented, so a filterable field can only be filtered by typing the exact value.",
        "how_to_unblock": "Compute, per filterable field, its distinct values with counts, applying every filter EXCEPT that field's own (standard faceting). The listing already filters in memory, so the same helper serves both.",
    },
    "units": {
        "kind": "not_built",
        "reason": "The unit viewer (GET /admin/v1/sources/{id}/units) is not implemented: there is no screen here to open a single chunk and see its text, its neighbours and the state of its embedding.",
        "how_to_unblock": "Build the two read-only routes of admin-unit/v1 and publish unit_types, which the schema requires as soon as this turns true: the type vocabulary is parseo's tipo_contenido, and retrieval_excluded is parseo.NO_CONTENIDO.",
    },
    "vectorize": {
        "kind": "not_built",
        "reason": "POST /admin/v1/sources/{id}/vectorize is not implemented. A source whose provider was down long enough to exhaust the retries and the second pass shows up here as `parcial`, and the only fix is to call pipeline/embeddings.vectorizar_libro yourself.",
        "how_to_unblock": "Expose that call as the endpoint: it already embeds only what has no vector and reports embedded / still missing / billable calls, which is exactly the contract's VectorizeResult.",
    },
}

#: La misma API key del middleware, y nada mas. Sin Clerk, sin JWT, sin roles: el contrato
#: pide declarar el rol que habilita las escrituras, y en esta instancia ese "rol" es tener la
#: clave. Como ademas no hay una sola escritura (ver CAPABILITIES), no hay nada que separe.
AUTH = {"schemes": ["api_key"], "admin_role": "api_key", "issuers": []}

FILTRABLES = {f["name"] for f in SOURCE_FIELDS if f.get("filterable")}
ORDENABLES = {"title", "updated_at", "stats.units", "stats.units_embedded"} | {
    f"fields.{n}" for n in FILTRABLES}
ESTADOS = ("ok", "parcial", "degradado", "en_proceso", "fallido")


# ---------------------------------------------------------------- perfil y descriptor
def perfil() -> dict:
    """El perfil activo. El cacheo lo hace `pipeline.perfiles`; el dominio, el entorno."""
    return perfiles.cargar(get_settings().profile)


def labels() -> dict:
    """Vocabulario de la interfaz: el del perfil si lo declara, el neutro si no."""
    declarados = perfil().get("labels") or {}
    return {**LABELS_POR_DEFECTO, **{k: v for k, v in declarados.items() if isinstance(v, str)}}


def source_kinds() -> list:
    """UNA clase de fuente: este pipeline trata a todos los documentos igual.

    Las instancias que tienen mas de una lo hacen porque su recorrido DIFIERE (una norma no se
    chunkea ni se vectoriza). Aca no difiere, y clases que no cambian nada serian decorado.
    """
    voc = labels()
    return [{
        "id": "documento",
        "label": voc["source_singular"].capitalize(),
        "unit_label": voc["unit_plural"],
        "description": "A document ingested through the pipeline: parsed, chunked, loaded and embedded.",
    }]


KIND_POR_DEFECTO = "documento"


def _ahora() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def descriptor() -> dict:
    """Lo que publica esta instancia para ser administrada. Todo lo de dominio viene del perfil."""
    # DIFERIDO A PROPOSITO: este modulo es `services/` y `mcp_remote` es un router; al reves seria un
    # ciclo. Es el mismo patron que usa `services/mcp_tools` con `routers/`.
    from routers import mcp_remote

    p = perfil()
    s = get_settings()
    return {
        "contract_version": "admin/v1",
        "domain": p["profile"],
        "profile": f"{p['profile']}@{p['version']}",
        "instance": {
            "name": s.instance_name,
            "display_name": s.instance_display_name,
            "api_base": s.public_base_url,
            "graph_target": (p.get("graph") or {}).get("target") or s.instance_name,
            "environment": s.environment_declarable,
            "version": s.instance_version,
        },
        "labels": labels(),
        "source_kinds": source_kinds(),
        "source_fields": SOURCE_FIELDS,
        "pipeline_steps": PIPELINE_STEPS,
        "entity_types": [{"id": e["id"], "label": e["label"], "description": e.get("desc", "")}
                         for e in p.get("entities", [])],
        "relation_types": [r["id"] for r in p.get("relations", [])],
        # SIN `accepted_formats` y SIN `unit_types`: el schema los exige cuando `upload` y `units`
        # son true, y aca las dos estan apagadas. Publicar formatos que ninguna puerta acepta seria
        # el mismo error que publicar la capacidad.
        "capabilities": CAPABILITIES,
        "capability_notes": CAPABILITY_NOTES,
        # `auth.mcp` DECLARA LA PUERTA DEL MCP (contrato `mcp/v1` §4): donde esta, que version del
        # contrato de tools implementa y que tools publica HOY. `tools` sale del REGISTRO del
        # servidor MCP y no de una lista retipeada: una vidriera que ofrece una tool que el servidor
        # no registro es un boton que falla contra el backend.
        #
        # **SIN `oauth`, Y ES LO CORRECTO**: ese bloque describe el estado de un servidor de
        # autorizacion (DCR/CIMD leidos de su metadata publica) y esta instancia no tiene ninguno --
        # autentica con una clave compartida, ver `AUTH`--. El contrato lo dice explicito: "ausente =
        # la instancia autentica el MCP de otra manera (una clave), y entonces no hay nada de OAuth
        # que contar". Publicarlo en `null` NO es una opcion: el schema lo declara `type: object`, asi
        # que un `null` ahi hace fallar la validacion del descriptor (comprobado contra la copia
        # vendorizada, 17-sep-2026).
        "auth": {**AUTH,
                 "mcp": {"url": s.mcp_resource_url,
                         "contract": mcp_remote.CONTRATO_MCP,
                         "tools": mcp_remote.nombres_de_tools()}},
    }


# ---------------------------------------------------------------- estado
def health() -> dict:
    """Estado de la API y del grafo. Nunca `sleeping`: esta instancia no despierta a nadie."""
    estado = graph.db_state()
    return {
        "status": "ok" if estado == "connected" else "degraded",
        "graph": estado,
        "wake_eta_s": None,
        "version": get_settings().instance_version,
        # Versionar el corpus es de dominios que lo necesitan (LexGraph versiona el normativo).
        "corpus_version": None,
        "checked_at": _ahora(),
    }


# Contar todos los nodos y relaciones de un grafo con corpus tarda decenas de segundos, y la
# consola pide stats al abrir. Se sirven de cache y `computed_at` dice de cuando son: el contrato
# lo permite explicitamente.
STATS_TTL_S = 600
_stats_cache: dict = {"ts": 0.0, "data": None}


def reiniciar_cache() -> None:
    """Vacia el cache de stats. Para los tests y para quien escriba en el grafo."""
    _stats_cache["ts"], _stats_cache["data"] = 0.0, None


def stats() -> dict:
    import time
    if _stats_cache["data"] and time.time() - _stats_cache["ts"] < STATS_TTL_S:
        return _stats_cache["data"]
    datos = _stats_sin_cache()
    _stats_cache["ts"], _stats_cache["data"] = time.time(), datos
    return datos


Q_UNIDADES = "MATCH (c:Chunk) RETURN count(c) AS units, count(c.embedding) AS units_embedded"
# Fuentes = los libro_id que tienen chunks, MAS los :Book que se quedaron sin ninguno. La union
# es la misma de `_todas_las_fuentes`: los dos conteos tienen que contar lo mismo.
Q_FUENTES_CON_CHUNKS = """
MATCH (c:Chunk) WHERE c.libro_id IS NOT NULL
RETURN count(DISTINCT c.libro_id) AS n
"""
Q_FUENTES_SIN_CHUNKS = """
MATCH (b:Book) WHERE NOT EXISTS { MATCH (c:Chunk {libro_id: b.id}) }
RETURN count(b) AS n
"""


def _stats_sin_cache() -> dict:
    base = graph.get_stats()
    unidades = graph.query(Q_UNIDADES)[0]
    con = graph.query(Q_FUENTES_CON_CHUNKS)[0]["n"]
    sin = graph.query(Q_FUENTES_SIN_CHUNKS)[0]["n"]
    return {
        "nodes_total": base["total_nodos"],
        "relationships_total": base["total_relaciones"],
        "nodes_by_label": base["nodos"],
        "relationships_by_type": base["relaciones"],
        "sources_total": con + sin,
        "units_total": unidades["units"],
        "units_embedded": unidades["units_embedded"],
        "computed_at": _ahora(),
    }


# ---------------------------------------------------------------- fuentes
# DOS CONSULTAS Y NO UNA, por una razon que se paga en produccion: en este repo `:Book` lo escribe
# `carga.registrar_fuente`, que es un paso APARTE de `carga.cargar_libro`. Quien cargue chunks sin
# registrar la fuente —y el README documenta las dos llamadas por separado— tendria una consola
# vacia sobre un grafo lleno. Asi que la lista es la union de los dos lados: los `:Book` (con sus
# chunks, si los tiene) y los `libro_id` que existen en los chunks sin `:Book` que los declare.
# El segundo grupo sale con `status_detail` diciendo que le falta registrarse, que es accionable.
#
# Los conteos son VIVOS (se cuentan los chunks) y no las propiedades `chunk_count`/`embedded_count`
# que deja `registrar_fuente`: esas envejecen en cuanto alguien vectoriza, y un panel que muestra
# un numero viejo es peor que uno que tarda un poco mas.
Q_FUENTES = """
MATCH (b:Book)
OPTIONAL MATCH (c:Chunk {libro_id: b.id})
RETURN b.id AS id, b.title AS title, b.author AS author, b.edition AS edition,
       b.pages AS pages, b.ingested_at AS ingested_at, b.ingested_by AS ingested_by,
       b.updated_at AS updated_at,
       count(c) AS units, count(c.embedding) AS units_embedded, max(c.page_end) AS ultima_pagina
"""
Q_FUENTES_HUERFANAS = """
MATCH (c:Chunk)
WHERE c.libro_id IS NOT NULL AND NOT EXISTS { MATCH (b:Book {id: c.libro_id}) }
RETURN c.libro_id AS id, count(c) AS units, count(c.embedding) AS units_embedded,
       max(c.page_end) AS ultima_pagina
"""


def estado_fuente(units: int, units_embedded: int) -> str:
    """El semaforo del contrato, con lo que este repo puede saber.

    `degradado` (texto ilegible) y `en_proceso` (hay un job vivo) no se usan: el engine no mide
    calidad de extraccion ni tiene jobs. Publicar un estado que nunca se emite seria ruido.
    """
    if units == 0:
        return "fallido"
    if units_embedded < units:
        return "parcial"
    return "ok"


def _detalle(status: str, units: int, units_embedded: int, registrada: bool) -> str | None:
    if status == "fallido":
        return ("The source node has no chunks: the load never ran, or every chunk was deleted "
                "as an orphan by a later ingest.")
    if status == "parcial":
        return f"{units - units_embedded} of {units} chunks have no embedding yet."
    if not registrada:
        return ("Chunks exist but no source node does: call carga.registrar_fuente to give this "
                "source its metadata.")
    return None


def _fuente(row: dict, registrada: bool = True) -> dict:
    """Fila del Cypher -> AdminSource (contrato admin-source/v1)."""
    units = row.get("units") or 0
    embebidas = row.get("units_embedded") or 0
    status = estado_fuente(units, embebidas)
    paginas = row.get("pages")
    if paginas is None:
        paginas = row.get("ultima_pagina")
    return {
        "id": row["id"],
        "kind": KIND_POR_DEFECTO,
        "title": row.get("title") or row["id"],
        "fields": {
            "titulo": row.get("title"),
            "autor": row.get("author"),
            "edicion": row.get("edition"),
            "paginas": paginas,
        },
        "stats": {
            "units": units,
            "units_embedded": embebidas,
            # El engine no marca la forma del texto con que se embebio ni cuenta entidades por
            # fuente: null es "no aplica", que es distinto de cero.
            "units_canonical": None,
            "entities": None,
            "relations": None,
            "pages": paginas,
        },
        "status": status,
        "status_detail": _detalle(status, units, embebidas, registrada),
        "lineage": {
            "job_id": None,
            "profile": None,
            "ingested_at": row.get("ingested_at"),
            "ingested_by": row.get("ingested_by") if row.get("ingested_by") in
            ("cli", "api", "consola", "legacy") else None,
            "corpus_version": None,
        },
        "active_job_id": None,
        "created_at": row.get("ingested_at"),
        "updated_at": row.get("updated_at") or row.get("ingested_at") or _ahora(),
    }


def _todas_las_fuentes() -> list:
    """Las dos naturalezas, unidas y ordenadas por id. Filtrar y paginar se hace en memoria.

    Es lo que hace LexGraph y por la misma razon: son dos consultas con shapes distintos y el
    numero de FUENTES es chico (decenas o cientos) aunque el de chunks no lo sea. Si algun dia un
    corpus lo pide, esto pasa a Cypher; hoy seria complejidad sin beneficio.
    """
    fuentes = [_fuente(r) for r in graph.query(Q_FUENTES)]
    fuentes += [_fuente(r, registrada=False) for r in graph.query(Q_FUENTES_HUERFANAS)]
    fuentes.sort(key=lambda f: f["id"])
    return fuentes


def _coincide(f: dict, q: str | None, kind: str | None, status: str | None, filtros: dict) -> bool:
    if kind and f["kind"] != kind:
        return False
    if status and f["status"] != status:
        return False
    if q and q.lower() not in f"{f['id']} {f['title'] or ''}".lower():
        return False
    for nombre, valor in filtros.items():
        actual = f["fields"].get(nombre)
        if valor == SIN_VALOR:
            if actual not in (None, ""):
                return False
        elif str(actual) != str(valor):
            return False
    return True


def codificar_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(f"o:{offset}".encode()).decode()


def decodificar_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        crudo = base64.urlsafe_b64decode(cursor.encode()).decode()
        if not crudo.startswith("o:"):
            raise ValueError(crudo)
        return max(0, int(crudo[2:]))
    except Exception as e:
        raise AdminError(422, "validation_error", "The cursor is not valid.") from e


def _clave_orden(clave: str):
    if clave == "stats.units":
        return lambda f: f["stats"]["units"]
    if clave == "stats.units_embedded":
        return lambda f: f["stats"]["units_embedded"]
    if clave.startswith("fields."):
        campo = clave.split(".", 1)[1]
        return lambda f: str(f["fields"].get(campo) or "").lower()
    if clave == "updated_at":
        return lambda f: f["updated_at"]
    return lambda f: (f["title"] or "").lower()


def _ordenar(fuentes: list, sort: str | None) -> list:
    sort = sort or "-stats.units"
    desc = sort.startswith("-")
    clave = sort[1:] if desc else sort
    if clave not in ORDENABLES:
        raise AdminError(422, "validation_error",
                         f"sort not allowed: {sort}. Allowed: {', '.join(sorted(ORDENABLES))}.")
    # El id desempata SIEMPRE: sin eso dos fuentes con los mismos chunks pueden cambiar de lugar
    # entre dos paginas y una de ellas no aparece nunca.
    fuentes.sort(key=lambda f: f["id"])
    fuentes.sort(key=_clave_orden(clave), reverse=desc)
    return fuentes


def _validar_filtros(kind: str | None, status: str | None, filtros: dict) -> None:
    if kind and kind not in {k["id"] for k in source_kinds()}:
        raise AdminError(422, "validation_error", f"unknown kind: {kind}")
    if status and status not in ESTADOS:
        raise AdminError(422, "validation_error", f"unknown status: {status}")
    desconocidos = set(filtros) - FILTRABLES
    if desconocidos:
        raise AdminError(
            422, "validation_error",
            f"not filterable: {', '.join(sorted(desconocidos))}. "
            f"Filterable fields (see descriptor.source_fields): "
            f"{', '.join(sorted(FILTRABLES)) or 'none'}.")


def listar_fuentes(q: str | None = None, kind: str | None = None, status: str | None = None,
                   filtros: dict | None = None, sort: str | None = None,
                   limit: int = 50, cursor: str | None = None) -> dict:
    filtros = filtros or {}
    _validar_filtros(kind, status, filtros)
    limit = max(1, min(int(limit), LIMITE_MAX))
    offset = decodificar_cursor(cursor)

    todas = [f for f in _todas_las_fuentes() if _coincide(f, q, kind, status, filtros)]
    _ordenar(todas, sort)
    pagina = todas[offset:offset + limit]
    siguiente = codificar_cursor(offset + limit) if offset + limit < len(todas) else None
    return {"items": pagina, "total": len(todas), "next_cursor": siguiente}


def obtener_fuente(source_id: str) -> dict:
    for f in _todas_las_fuentes():
        if f["id"] == source_id:
            return f
    raise AdminError(404, "not_found", f"No such source: {source_id}")
