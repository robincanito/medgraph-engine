"""Endpoint MCP remoto (Streamable HTTP): la cuarta implementacion de `mcp/v1`, con clave.

QUE ES. La cara MCP de este servicio: las SIETE tools obligatorias del contrato `mcp/v1` de
nomos-contracts sobre lo que ingesto el pipeline. Un agente escrito contra `search`, `fetch`,
`list_sources`, `search_entities`, `expand_concept`, `deep_dive` y `unified_query` de cualquier grafo
Nomos funciona contra este sin tocar una linea: cambia el host y cambia el corpus.

REEMPLAZA AL `mcp_server.py` DE LA RAIZ, que se retiro el 17-sep-2026. Aquel era un servidor **stdio**
que le hablaba por HTTP a una API corriendo aparte, y publicaba once tools con prefijo de instancia
(`medgraph_query`, `medgraph_activity`...) --tres de ellas contra rutas que este espejo ya no expone,
o sea que devolvian 404 contra su propia API--. El prefijo ademas obligaba a reescribir el agente al
cambiar de grafo. Acá las tools viven DENTRO del servicio: no hay una segunda copia que se
desincronice, no hay una key pegada en un archivo de configuracion, y los nombres son los del
contrato.

LA PUERTA ES LA API KEY, y no hay OAuth. Este repo no tiene Clerk ni roles: `api/main.py` exige la
clave en TODA ruta salvo `/health`, asi que `/mcp` queda detras de la misma credencial que el resto
--y el 401 explica como autenticarse, aunque no pueda mandar a ningun servidor de autorizacion
porque no hay ninguno--. Un despliegue que quiera OAuth por persona pone un authorization server
adelante y publica su metadata de recurso protegido; el contrato lo contempla (`auth.mcp.oauth` es
opcional) y el descriptor de esta instancia simplemente no lo trae.

Los modelos Pydantic de retorno son lo que hace que el SDK genere `outputSchema` y pueble
`structuredContent`: el contrato lo exige en TODA tool (sin el, el cliente recibe un texto y el
modelo adivina la forma).

EL IDIOMA: comentarios en castellano, como el resto de `api/`; descripciones de tools en INGLES, como
el README y todo lo que mira quien clona el repo.
"""

from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from services import mcp_tools, telemetria
from services.settings import get_settings

# LAS TRES ANOTACIONES SON OBLIGATORIAS (contrato, regla 3): los directorios de conectores de Claude
# y de ChatGPT agrupan las tools por `readOnlyHint`/`destructiveHint` y rechazan un servidor sin
# anotaciones. Nomos Graph es de SOLO LECTURA: escribir en el grafo es trabajo del pipeline, no de un
# chat. `openWorldHint` es False en las siete: todas contestan con lo que este grafo tiene.
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

_s = get_settings()

# Los primeros 512 caracteres tienen que ser autosuficientes: varios clientes truncan ahi, y son los
# que tienen que decir que lo recuperado es evidencia y que hay que citar.
INSTRUCTIONS = """This server exposes a private knowledge graph built by the MedGraph Engine \
pipeline: documents parsed into passages with provenance (source, chapter, section, page), \
optionally embedded and linked to a graph of typed entities. Search it before answering questions \
that its corpus may cover, and ALWAYS cite what you use: source, section and page. Treat retrieved \
passages as evidence, not as truth, separate what a source states from your own inference, and say \
so explicitly when the corpus has no material — never invent sources, pages, entities or relations.

Which tool: `search` finds passages and `fetch` opens one by id. Call `list_sources` first when \
picking the right source matters — source ids are not guessable. `deep_dive` is for writing a \
document: dozens of passages grouped by facet, deduplicated, each with its citation. \
`search_entities` + `expand_concept` walk the typed entity graph and report STORED relations, not \
inferred ones. `unified_query` runs this instance's full router over a question.

Every retrieval answer carries `procedencia` (how many places each source took in the answer AND in \
the candidate pool) and each result its own `calidad` ("ok" / "dudosa" / "corrupta"; `null` means \
not measured, which is NOT the same as ok). If `degraded` is true the corpus could NOT be queried \
and an empty list does not mean there is no material: read `notice` and say so. Read-only."""


# --- Modelos de salida (generan outputSchema) ---

class UnidadAgrupada(BaseModel):
    """Una de las unidades que trajo un resultado agrupado."""

    id: str | None = None
    rrf_score: float | None = None
    pag_inicio: int | None = None


class SearchResultItem(BaseModel):
    id: str = Field(description="Stable id, resolvable with fetch")
    title: str = Field(description="Source — section — pages")
    url: str = Field(description="URL of the containing SOURCE in this instance's admin API")
    snippet: str = Field(default="", description="Leading fragment, to decide whether to open it")
    source_id: str | None = Field(default=None, description="Id of the containing source")
    locator: str | None = Field(default=None, description='Citable position: "p. 604"')
    page_start: int | None = None
    page_end: int | None = None
    calidad: str | None = Field(default=None, description='"ok" | "dudosa" | "corrupta" | null')
    agrupado_por: str | None = Field(default=None, description='"padre" | "pagina" | null')
    unidades: list[UnidadAgrupada] = Field(
        default=[], description="The units that brought this group (empty if not grouped)")


class SearchOutput(BaseModel):
    results: list[SearchResultItem]
    # `degraded` + `notice` VIAJAN SIEMPRE (contrato, regla 4): una lista vacia por caida del
    # retrieval se lee igual que "no hay material", y el modelo contesta de su propio conocimiento
    # sin decirlo. `notice` esta redactado para que lo lea un modelo.
    degraded: bool = False
    notice: str | None = None
    procedencia: dict[str, Any] = {}


class FetchOutput(BaseModel):
    id: str
    title: str
    text: str
    url: str
    metadata: dict[str, Any]


class SourceItem(BaseModel):
    id: str
    titulo: str | None = None
    autor: str | None = None
    edicion: str | None = None
    colecciones: list[str] = Field(
        default=[], description="Empty here: this pipeline does not write collections")
    unidades: int | None = None
    unidades_vectorizadas: int | None = None
    paginas: int | None = None
    estado: str | None = None
    detalle: str | None = None


class ListSourcesOutput(BaseModel):
    stats: dict[str, Any]
    sources: list[SourceItem]
    count: int
    degraded: bool = False
    notice: str | None = None


class EntityItem(BaseModel):
    id: str
    label: str | None = None
    type: str | None = None
    freq: int | None = None
    degree: int | None = None


class SearchEntitiesOutput(BaseModel):
    query: str
    entities: list[EntityItem]
    count: int
    degraded: bool = False
    notice: str | None = None


class GraphNode(BaseModel):
    id: str
    label: str | None = None
    type: str | None = None
    freq: int | None = None
    degree: int | None = None


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str


class ExpandOutput(BaseModel):
    center: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    count: dict[str, int]


class DeepDivePassage(BaseModel):
    id: str
    source: str | None = None
    title: str | None = None
    url: str | None = None
    locator: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    text: str
    truncated: bool = False
    calidad: str | None = None
    agrupado_por: str | None = None
    unidades: list[UnidadAgrupada] = []


class DeepDiveOutput(BaseModel):
    topic: str
    facets: dict[str, list[DeepDivePassage]]
    total_passages: int
    sources_used: list[str] = []
    # Relaciones GUARDADAS del grafo de entidades. Van aparte de `facets` a proposito: una relacion
    # extraida por un LLM no es una cita de la fuente.
    graph_relations: list[dict[str, Any]] = []
    degraded: bool = False
    notice: str | None = None
    procedencia: dict[str, Any] = {}


class UnifiedQueryOutput(BaseModel):
    question: str
    intent: str | None = None
    layers_activated: list[str] = []
    sub_queries: list[str] = []
    bibliography: dict[str, Any] = {}
    degraded: bool = False
    notice: str | None = None
    procedencia: dict[str, Any] = {}


# --- Servidor MCP ---

# `stateless_http=True`: un servicio que escala horizontalmente no garantiza sesiones pegajosas --con
# sesiones con estado, un request podria caer en otra instancia que no la conoce--.
#
# `streamable_http_path="/mcp"` (default) + montar el sub-app en la RAIZ. Si en cambio se monta en
# "/mcp" con path interno "/", Starlette responde 307 a "/mcp/", y un redirect en POST es fragil:
# varios clientes descartan el body. Asi `/mcp` responde directo, sin redirect.
#
# Proteccion anti DNS-rebinding del SDK: valida el header `Host` y por defecto solo acepta localhost
# --detras de un dominio propio devuelve "421 Invalid Host header"--. Los hosts salen de settings.
_HOSTS = _s.mcp_allowed_hosts_list

_seguridad_transporte = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=_HOSTS,
    # Los clientes de chat llaman server-to-server (sin `Origin` de browser). Se listan los origenes
    # declarados por si alguno lo manda.
    allowed_origins=sorted({*_s.cors_origins, _s.public_base_url}),
)

mcp_server = FastMCP(
    _s.instance_display_name,
    instructions=INSTRUCTIONS,
    stateless_http=True,
    streamable_http_path="/mcp",
    transport_security=_seguridad_transporte,
)

# LA TELEMETRIA ENVUELVE CADA TOOL y va DEBAJO de `@mcp_server.tool(...)`, o sea mas cerca de la
# funcion: FastMCP tiene que registrar el wrapper mas externo para que la medicion ocurra.
# `functools.wraps` copia `__wrapped__` y `inspect.signature` lo sigue, asi que el schema que el SDK
# publica NO cambia --y el contrato CIERRA los argumentos de cada tool, asi que un decorador que los
# cambiara pondria rojo el test de conformidad--.
#
# NO HAY GUARDA DE IDENTIDAD POR TOOL, y es una diferencia con la instancia privada que conviene
# entender: alla cada tool exige que quien pregunta pertenezca a una organizacion de Clerk, porque
# alla hay personas distintas con un mismo corpus. Acá hay UNA credencial y ninguna identidad que
# distinguir: quien tiene la clave entra a todo, y el middleware de `api/main.py` ya lo resolvio
# antes de que la peticion llegue hasta aca. Si algun dia esta API tuviera identidades, la guarda
# entra como un decorador mas, al lado de este.


@mcp_server.tool(annotations=READ_ONLY, structured_output=True)
@telemetria.medir("search")
def search(query: str, limit: int = 10, filtros: dict[str, Any] | None = None,
           garantizar: dict[str, Any] | None = None,
           agrupar: str | None = None) -> SearchOutput:
    """Search this instance's corpus for relevant passages, with provenance and citable positions.

    Use it before answering questions that the corpus may cover. Returns stable ids and the source
    each passage belongs to; call `fetch` with an id to read the full passage.

    Do NOT use it to modify the graph or to infer facts absent from the sources.

    - query: a single natural-language string, in the language of the corpus.
    - limit: max results, 1-20 (default 10).
    - filtros: restrict WHICH sources may answer. `{"libro_ids": [...], "collections": [...],
      "areas": [...], "tipos": [...], "excluir": [...]}` — all optional, all lists, combined with
      AND. The real values come from `list_sources`: do not guess ids. An unknown key is rejected.
    - garantizar: reserve places for sources you name. `{"collections": [...] | "libro_ids": [...],
      "cupo": N}` (default cupo: half the limit). A reserved place stays EMPTY when that source has
      nothing on the topic, and `procedencia.garantizados = 0` is an answer, not a failure.
    - agrupar: "padre" collapses hits of the same parent passage into ONE result carrying the whole
      parent window plus the child units that brought it; "pagina" collapses by page. Default: no
      grouping, one passage per place.

    Every answer carries `procedencia` (places per source in the answer AND in the candidate pool)
    and every result its own `calidad`. An empty query or no matches returns an empty list, not an
    error; if `degraded` is true the corpus could not be queried and the empty list is not an
    absence.
    """
    return SearchOutput(**mcp_tools.search(query, limit=limit, filtros=filtros,
                                           garantizar=garantizar, agrupar=agrupar))


@mcp_server.tool(annotations=READ_ONLY, structured_output=True)
@telemetria.medir("fetch")
def fetch(id: str) -> FetchOutput:
    """Retrieve the full text of a passage returned by `search`.

    - id: exactly one of the ids from a previous `search` result. Do not construct or guess ids.

    Returns the text plus provenance metadata: source id and title, author and edition when the
    source was registered, chapter, section, page range, content type, whether the passage has an
    embedding, and `truncated` when the text was cut. An unknown id returns a typed error, not an
    empty document.
    """
    return FetchOutput(**mcp_tools.fetch(id))


@mcp_server.tool(annotations=READ_ONLY, structured_output=True)
@telemetria.medir("list_sources")
def list_sources(autor: str = "", q: str = "") -> ListSourcesOutput:
    """List which sources this instance has ingested, and how much of each one is indexed.

    Use it BEFORE searching when picking the right source matters, and whenever you need the ids
    that `filtros` and `garantizar` take: source ids are specific to this deployment and cannot be
    guessed.

    - autor: filter by author, as it was written when the source was registered.
    - q: free text over the id and the title.

    Each source reports how many `unidades` (passages) it has, how many are vectorised, its pages
    and its `estado`: "ok", "parcial" (part of it has no embedding yet, so semantic search will miss
    those passages) or "fallido" (the source node has no passages at all). `colecciones` is empty
    here: this pipeline does not write collections.
    """
    return ListSourcesOutput(**mcp_tools.list_sources(autor=autor, q=q))


@mcp_server.tool(annotations=READ_ONLY, structured_output=True)
@telemetria.medir("search_entities")
def search_entities(query: str, limit: int = 20) -> SearchEntitiesOutput:
    """Find typed entities of the knowledge graph by name, to walk their relations afterwards.

    The entity vocabulary is the one declared by this instance's active domain profile (see
    `descriptor.entity_types`), and the entities themselves were extracted from the corpus. Use this
    when the question is about how concepts relate, rather than about a passage of text.

    - query: entity name or partial name.
    - limit: 1-50 (default 20).

    Returns the ids that `expand_concept` consumes. Ids are not constructed: they come from here.
    """
    return SearchEntitiesOutput(**mcp_tools.search_entities(query, limit=limit))


@mcp_server.tool(annotations=READ_ONLY, structured_output=True)
@telemetria.medir("expand_concept")
def expand_concept(entity_id: str, relation_types: list[str] | None = None,
                   limit: int = 20) -> ExpandOutput:
    """Retrieve the graph neighbours of one entity — one hop, and only one.

    Relation types are the ones declared by this instance's domain profile (see
    `descriptor.relation_types`) and were extracted from the corpus.

    - entity_id: an id from `search_entities`. Do not guess ids.
    - relation_types: optional whitelist of RELATION types (not of node labels).
    - limit: 1-200 neighbours (default 20).

    Report these as stored graph evidence, separately from your own inference: they are what was
    saved, not a conclusion. One hop and not two on purpose: two hops over a graph of this size is
    an answer no client can read.
    """
    return ExpandOutput(**mcp_tools.expand_concept(entity_id, relation_types, limit))


@mcp_server.tool(annotations=READ_ONLY, structured_output=True)
@telemetria.medir("deep_dive")
def deep_dive(topic: str, facets: list[str] | None = None, per_facet: int = 6,
              include_graph: bool = True, filtros: dict[str, Any] | None = None,
              garantizar: dict[str, Any] | None = None,
              agrupar: str | None = mcp_tools.AGRUPACION_DEEP_DIVE) -> DeepDiveOutput:
    """Exhaustively retrieve everything this corpus has on a topic, grouped by facet.

    USE THIS — not `search` — when asked for a study document, a full review, a complete summary or
    anything chapter-sized. `search` returns 10-20 passages: enough to answer a question, NOT enough
    to write a document without filling gaps from general knowledge. This returns dozens of passages
    already organised by facet, each with its source, its page and its citation.

    - topic: the subject.
    - facets: optional subset. The defaults are generic axes for reading a document (definition,
      context, mechanism, classification, procedure, criteria, risks, evidence), written in the
      language declared by the active profile — they are deliberately domain-neutral, so if your
      domain has better axes, pass them here: free-form facets are accepted and searched as written.
    - per_facet: passages per facet, 2-12 (default 6).
    - include_graph: also return stored relations from the entity graph around the topic.
    - filtros / garantizar: same as in `search` — scope the document to some sources, or reserve
      places for the ones the user named.
    - agrupar: **"padre" by default here**, unlike `search`: a document is written from whole
      passages, not from five overlapping cuts of the same page. Pass null for raw units.

    Passages are deduplicated across facets: each appears once, under the facet that brought it.
    Write each section from the passages of ITS facet and cite the source and page. If a facet came
    back empty, say so — do not fill it from general knowledge.
    """
    return DeepDiveOutput(**mcp_tools.deep_dive(topic, facets=facets, per_facet=per_facet,
                                                include_graph=include_graph, filtros=filtros,
                                                garantizar=garantizar, agrupar=agrupar))


@mcp_server.tool(annotations=READ_ONLY, structured_output=True)
@telemetria.medir("unified_query")
def unified_query(question: str, top_k: int = 8, filtros: dict[str, Any] | None = None,
                  garantizar: dict[str, Any] | None = None,
                  agrupar: str | None = None) -> UnifiedQueryOutput:
    """Run this instance's full query router over one question.

    An LLM analyser classifies the question, expands it and decomposes it into sub-queries; each one
    searches the corpus and the results are fused and deduplicated. `layers_activated` says which
    layers the router chose, which is the only way to tell a poor answer from a layer that never
    ran.

    Prefer this over plain `search` for broad or reasoning-heavy questions; for a single factual
    lookup, `search` is cheaper and faster.

    - question: the question in natural language.
    - top_k: passages to return (default 8, capped at 20).
    - filtros / garantizar / agrupar: same as in `search`; they reach the corpus layer. The answer
      carries `procedencia` and every passage its `calidad`.

    The analyser needs a model provider configured in this deployment. If it cannot be reached, the
    question is searched exactly as written and `notice` says so: the retrieved material is real,
    the routing is not.
    """
    return UnifiedQueryOutput(**mcp_tools.unified_query(question, top_k=top_k, filtros=filtros,
                                                        garantizar=garantizar, agrupar=agrupar))


#: La version del contrato de tools que implementa este endpoint. Viaja en el descriptor de admin/v1
#: (`auth.mcp.contract`): un cliente que no conoce la version no adivina, rechaza.
CONTRATO_MCP = "mcp/v1"


def nombres_de_tools() -> list[str]:
    """Los nombres que este servidor publica, en el orden en que un cliente los ve.

    SALE DEL REGISTRO Y NO DE UNA LISTA ESCRITA A MANO: el descriptor publica esto en
    `auth.mcp.tools`, y una vidriera que ofrece una tool que el servidor no registro es el mismo bug
    de honestidad que una capacidad publicada sin endpoint.

    `_tool_manager` es privado y se usa igual porque es el unico camino SINCRONICO: `list_tools()` de
    FastMCP es una corutina y el descriptor se arma en una funcion sincronica. Vive ACA --al lado del
    servidor-- para que el acceso privado quede en un solo lugar y el test lo cubra.
    """
    return [t.name for t in mcp_server._tool_manager.list_tools()]


class _PuertaMCP:
    """El ASGI que se monta: delega en la app Streamable HTTP VIGENTE.

    POR QUE NO SE MONTA DIRECTO `mcp_server.streamable_http_app()`, que es lo que hace la instancia
    privada. `StreamableHTTPSessionManager.run()` **se puede correr UNA sola vez por instancia** (lo
    dice su propio RuntimeError), y la suite de este repo entra y sale del lifespan VARIAS veces:
    `tests/test_admin_v1.py` abre `with TestClient(main.app)` en cuatro lugares --una de ellas la
    fixture `cliente`, o sea una vez por test que la use--, que es su forma de probar la app de
    verdad. Con el mount directo, el segundo `with` reventaba con "StreamableHTTPSessionManager
    .run() can only be called once per instance" --medido el 17-sep-2026-- y se llevaba puestos diez
    tests que no tienen nada que ver con MCP.

    Las salidas posibles eran: (a) reescribir esas fixtures para no entrar al lifespan --o sea
    cambiar como se prueba la app por una limitacion del SDK--, o (b) darle al mount una identidad
    estable y crear un session manager NUEVO en cada arranque. Se elige (b): el mount es este objeto,
    que no cambia, y lo que se rehace por lifespan es la app interna.

    En produccion el lifespan corre una sola vez, asi que esto no cambia nada; lo que agrega es que
    un `uvicorn --reload`, un test o un runner que arranque dos veces en el mismo proceso no dejen el
    endpoint muerto. Si el SDK cambia el atributo privado que se resetea, el test
    `tests/test_mcp.py::TestElCicloDeVida` se pone rojo acá y no en el despliegue de alguien.
    """

    def __init__(self):
        self._app = None

    @asynccontextmanager
    async def correr(self):
        # Manager NUEVO por arranque: el anterior quedo marcado como "ya corrio" y su task group
        # desarmado. `_session_manager` es privado del SDK y se resetea a proposito --es la unica
        # costura que tiene-- para que `streamable_http_app()` lo vuelva a crear.
        mcp_server._session_manager = None
        self._app = mcp_server.streamable_http_app()
        try:
            async with mcp_server.session_manager.run():
                yield
        finally:
            self._app = None

    async def __call__(self, scope, receive, send):
        # Un mount NUNCA recibe el scope de lifespan (por eso hay que encadenarlo a mano en
        # `api/main.py`), asi que acá solo llegan http y websocket.
        if self._app is None:
            # Pasa si alguien monta esta app sin encadenar `mcp_lifespan()`. Antes eso era un
            # AssertionError del SDK con el texto "Task group is not initialized"; un 503 con el
            # motivo es lo mismo pero legible, y ademas es el codigo correcto: el servicio esta, el
            # endpoint todavia no.
            from starlette.responses import JSONResponse

            await JSONResponse(
                {"detail": "The MCP endpoint is not running: its session manager was never "
                           "started. Chain mcp_remote.mcp_lifespan() into the app lifespan.",
                 "code": "mcp_not_started"}, status_code=503)(scope, receive, send)
            return
        await self._app(scope, receive, send)


_puerta = _PuertaMCP()


#: Los verbos del transporte Streamable HTTP: POST lleva el JSON-RPC, GET abre el stream SSE y
#: DELETE cierra la sesion. Se declaran para que Starlette conteste 405 --y no el MCP-- ante
#: cualquier otro; ver `rutas_mcp`.
METODOS_MCP = ("GET", "POST", "DELETE")


def get_asgi_app():
    """El ASGI a montar en la raiz. Ver `_PuertaMCP` por que es un objeto estable y no la app."""
    return _puerta


def rutas_mcp() -> list:
    """Las rutas del MCP remoto, para AGREGAR a la app (no montar). Ver el pie de `api/main.py`.

    POR QUE NO ES UN MOUNT (18-sep-2026). Montar la app del SDK en `/mcp` hace que Starlette
    conteste **307 a `/mcp/`**, y un redirect en POST es fragil: varios clientes descartan el
    body. La vuelta era montar en la RAIZ --path interno `/mcp`-- y eso anduvo, pero un
    `Mount` matchea **por path y no por metodo**: con el mount en `/`, CUALQUIER peticion que
    no case con una ruta declarada antes --un verbo equivocado sobre una ruta que existe, un
    path inventado-- caia adentro del MCP, y la API dejaba de contestar **405** en toda su
    superficie.

    LA CURA: dos `Route` EXACTAS con sus metodos declarados. No hay 307 --el path es exacto-- y
    no se come nada --el match es por path Y por metodo--. `/mcp/` va aparte porque un cliente
    que la escriba con barra final no merece un 404. La puerta (`_PuertaMCP`) sigue siendo la
    misma: lo que cambia es COMO se engancha, no que hay detras.
    """
    from starlette.routing import Route

    # EL ENDPOINT TIENE QUE SER UN OBJETO, NO UNA FUNCION: `Route` mira
    # `inspect.isfunction(endpoint)` y, si lo es, la envuelve con `request_response` --o sea la
    # llama con UN argumento-- en vez de tratarla como app ASGI de tres. `_PuertaMCP` ya es un
    # objeto con `__call__`, asi que se pasa tal cual; `methods` sigue valiendo y es lo que da
    # el 405.
    return [Route(ruta, endpoint=get_asgi_app(), methods=list(METODOS_MCP))
            for ruta in ("/mcp", "/mcp/")]


def mcp_lifespan():
    """Context manager a encadenar al lifespan de la app padre.

    Sin esto el endpoint no atiende: montar un sub-app NO ejecuta su lifespan, y el session manager
    del SDK necesita un task group vivo.
    """
    return _puerta.correr()
