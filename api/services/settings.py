"""Configuration of the API: ONE source of truth for everything that comes from the environment.

WHY THIS FILE EXISTS. `services/vector.py` was replicated from the private repo with the retrieval
fixes and imports `services.settings`, a module that never made it into this mirror. The result was
that `from services import vector` raised `ModuleNotFoundError` and the whole API failed to import:
the README listed "the API does not start" under Known gaps. This is that module, rewritten
generic — it reads environment variables and nothing else. There is no instance-specific default
here: no private host, no project id, no Clerk, no Secret Manager.

NO pydantic-settings ON PURPOSE. The private instance uses it; here it would be one more dependency
in a repo whose point is that you can clone it and run it. `os.getenv` plus a frozen dataclass does
the same job with the standard library, and `requirements.txt` stays as short as it is.

BORDER SANITIZER (the scar that pays for it): a secret pasted on Windows or written by a secret
manager can end up with a trailing `\\r`, and a container mounts it exactly as it is.
`compare_digest("key", "key\\r")` then rejects EVERY client, and no HTTP header can carry a `\\r`,
so the credential becomes unreachable and nothing says why. Every string value is `.strip()`ed on
the way in: no field of this model has a legitimate space or newline at its borders, so stripping
can only remove garbage.

FAIL CLOSED. `API_KEY` is not defaulted here: `main.py` refuses to start without it (an empty key
used to authorize every request that carried no header). `instance_name` is validated because it
travels in the admin/v1 descriptor and a malformed one makes the console reject the document.
"""
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

# El .env se carga ACA y no en cada modulo: este es el primer import de la cadena
# (main -> routers -> services), asi que cuando alguien lee una variable ya esta puesta.
load_dotenv()

#: `instance.name` del descriptor admin/v1 (schema admin-descriptor/v1): identificador corto.
PATRON_NOMBRE_INSTANCIA = re.compile(r"^[a-z][a-z0-9-]*$")

#: Entornos que el contrato admite en `instance.environment`. Cualquier otra cosa (el "test" que
#: siembra la suite, por ejemplo) se publica como development: es lo que es, y el schema solo
#: acepta estos tres.
ENTORNOS = ("development", "staging", "production")


def _env(*nombres: str, default: str = "") -> str:
    """Primera variable definida de la lista, saneada. Varios nombres = alias historicos."""
    for nombre in nombres:
        valor = os.getenv(nombre)
        if valor is not None and valor.strip():
            return valor.strip()
    return default


def _env_o_none(*nombres: str) -> str | None:
    """Como `_env`, pero ausente es None y no "": lo distingue quien tiene que fallar cerrado."""
    return _env(*nombres) or None


@dataclass(frozen=True)
class Settings:
    """Lo que la API lee del entorno. Instanciar via `get_settings()`."""

    # ── Neo4j ────────────────────────────────────────────────────────────────
    # Sin default de conexion: una URI inventada aca haria que el servicio parezca sano y
    # falle en la primera consulta. Si falta, /health lo dice y las rutas del grafo dan 503.
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = None

    # ── Auth de la API ───────────────────────────────────────────────────────
    # Una sola clave portadora, para TODA la API (admin/v1 incluido). No hay roles ni JWT en el
    # engine: eso es de la instancia privada. Ver README, "Security notes".
    api_key: str = ""
    environment: str = "production"

    # ── Proveedor de embeddings (services/vector.py, ruta de consulta) ───────
    # Vertex AI con ADC. `global` y no us-central1: los modelos de embedding 2.x/3.x solo
    # existen en el endpoint global (404 en la region).
    gcp_project: str = ""
    gcp_location: str = "global"
    # AI Studio: lo usa el router del /query (services/analyzer.py). Es OTRO camino de auth,
    # y esta separado a proposito (README, "Requirements").
    gcp_api_key: str = ""

    # ── Instancia: lo que publica el descriptor de admin/v1 ──────────────────
    profile: str = "medicina"
    instance_name: str = "medgraph-engine"
    instance_display_name: str = "MedGraph Engine"
    instance_version: str | None = None
    public_base_url: str = "http://localhost:8000"
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    # ── MCP remoto (routers/mcp_remote.py, contrato mcp/v1) ──────────────────
    # Hosts que acepta la proteccion anti DNS-rebinding del SDK de MCP. Vacio = se derivan de
    # `public_base_url` mas localhost (ver `mcp_allowed_hosts_list`): un despliegue detras de un
    # proxy que cambia el `Host` --o con un segundo dominio-- los declara con MCP_ALLOWED_HOSTS,
    # porque si no el SDK contesta "421 Invalid Host header" y el cliente no sabe por que.
    mcp_allowed_hosts: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not PATRON_NOMBRE_INSTANCIA.match(self.instance_name):
            raise ValueError(
                f"INSTANCE_NAME={self.instance_name!r} no sirve como identificador de instancia: "
                f"tiene que cumplir {PATRON_NOMBRE_INSTANCIA.pattern} (minusculas, digitos y "
                "guiones, empezando por letra). Viaja en admin/v1 `instance.name` y la consola "
                "rechaza el descriptor que no lo cumple."
            )

    @property
    def environment_declarable(self) -> str:
        """El entorno tal como lo admite el contrato admin/v1."""
        return self.environment if self.environment in ENTORNOS else "development"

    @property
    def api_base(self) -> str:
        """La base publica, sin barra final. Es lo que publica `instance.api_base`."""
        return self.public_base_url.rstrip("/")

    @property
    def mcp_resource_url(self) -> str:
        """La URL EXACTA del endpoint MCP: la que una persona escribe en su cliente.

        DERIVADA Y NO UN CAMPO APARTE: el contrato `mcp/v1` §4 pide que `auth.mcp.url` cuelgue de
        `instance.api_base`, y con dos campos configurables habria dos valores que pueden diferir.
        """
        return f"{self.api_base}/mcp"

    @property
    def mcp_allowed_hosts_list(self) -> list[str]:
        """Los hosts declarados, o los derivados de `public_base_url` mas localhost.

        El default tiene que servir para el caso que este repo documenta --`uvicorn` local y un
        contenedor detras de su propio dominio-- sin obligar a configurar nada; lo que NO puede
        hacer es aceptar cualquier `Host`, que es justo lo que la proteccion anti DNS-rebinding
        existe para impedir.
        """
        if self.mcp_allowed_hosts:
            return list(self.mcp_allowed_hosts)
        from urllib.parse import urlsplit

        propio = urlsplit(self.public_base_url).netloc
        hosts = [propio] if propio else []
        for h in ("localhost", "127.0.0.1", "localhost:8000", "127.0.0.1:8000",
                  "localhost:8080", "127.0.0.1:8080"):
            if h not in hosts:
                hosts.append(h)
        return hosts


def _leer_entorno() -> Settings:
    origenes = [o.strip() for o in _env("CORS_ORIGINS").split(",") if o.strip()]
    return Settings(
        neo4j_uri=_env_o_none("NEO4J_URI"),
        neo4j_username=_env_o_none("NEO4J_USERNAME"),
        neo4j_password=_env_o_none("NEO4J_PASSWORD"),
        neo4j_database=_env_o_none("NEO4J_DATABASE"),
        api_key=_env("API_KEY"),
        environment=_env("ENVIRONMENT", default="production"),
        # GCP_PROJECT_ID es el alias que usan varios scripts de la raiz; gana GCP_PROJECT.
        gcp_project=_env("GCP_PROJECT", "GCP_PROJECT_ID"),
        gcp_location=_env("GCP_LOCATION", default="global"),
        gcp_api_key=_env("GCP_API_KEY"),
        profile=_env("PROFILE", default="medicina"),
        instance_name=_env("INSTANCE_NAME", default="medgraph-engine"),
        instance_display_name=_env("INSTANCE_DISPLAY_NAME", default="MedGraph Engine"),
        # K_REVISION es la revision que inyecta Cloud Run; se acepta como alias para que un
        # deploy ahi publique su version sin configurar nada.
        instance_version=_env_o_none("INSTANCE_VERSION", "K_REVISION"),
        # API_BASE_URL es como se llamaba en la v1.0 de este archivo main.py.
        public_base_url=_env("PUBLIC_BASE_URL", "API_BASE_URL", default="http://localhost:8000"),
        cors_origins=origenes or ["http://localhost:3000"],
        mcp_allowed_hosts=[h.strip() for h in _env("MCP_ALLOWED_HOSTS").split(",") if h.strip()],
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instancia unica. En tests: sembrar el entorno y llamar `get_settings.cache_clear()`."""
    return _leer_entorno()
