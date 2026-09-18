"""Salud del repo publico: que todo compile, que las dependencias esten declaradas y que no
haya un secreto adentro.

POR QUE ESTOS TRES. Es un repo que la gente clona: los tres fallos que arruinan ese primer
minuto no son de logica.
  1. Un archivo que no PARSEA. El caso que lo motivo fue `mcp_server.py`, que estuvo asi desde
     v1.0 —una edicion de limpieza se comio una coma al sacar una URL privada— y moria en el
     import; nadie lo vio porque no habia un test que abriera los archivos. Ese archivo se
     retiro el 17-sep-2026 (el servidor MCP es ahora un endpoint de la API), pero el test se
     queda: lo que cuida no es ese archivo sino que NINGUNO este roto.
  2. Una dependencia que la suite usa y `requirements.txt` no declara: el clon pasa los tests
     en la maquina del autor y falla en la del otro.
  3. Un secreto versionado. `.env` tiene que estar ignorado y `.env.example` tiene que traer
     placeholders, no la key de alguien.
"""
import ast
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent

#: Directorios que no son codigo del repo (dependencias, cache, datos locales del usuario).
IGNORADOS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv",
             "node_modules", "parsed", "extracted", "examples", "data"}

#: Extensiones que se barren buscando credenciales. NO alcanza con `*.py` (13-sep-2026, item B-8 de
#: la auditoria de exposicion): una key se filtra mucho mas facil en un YAML de configuracion, en un
#: JSON exportado, en el README que alguien copio de su terminal o en un `pyproject.toml` que en el
#: codigo. `.env.example` no tiene extension y entra por nombre.
EXTENSIONES = {".py", ".yaml", ".yml", ".json", ".md", ".toml"}
POR_NOMBRE = {".env.example"}


def _fuentes() -> list:
    return sorted(p for p in RAIZ.rglob("*.py")
                  if not IGNORADOS & set(p.relative_to(RAIZ).parts))


def _archivos() -> list:
    """Todo archivo de texto del repo que puede llevar un secreto o un nombre ajeno adentro."""
    return sorted(p for p in RAIZ.rglob("*")
                  if p.is_file()
                  and (p.suffix in EXTENSIONES or p.name in POR_NOMBRE)
                  and not IGNORADOS & set(p.relative_to(RAIZ).parts))


class TestTodoCompila:
    @pytest.mark.parametrize("ruta", _fuentes(), ids=lambda p: str(p.relative_to(RAIZ)))
    def test_cada_archivo_python_parsea(self, ruta):
        """`ast.parse` y no un import: no hace falta tener instalada la dependencia de cada
        script para saber que el archivo es Python valido."""
        fuente = ruta.read_text(encoding="utf-8")
        try:
            ast.parse(fuente, filename=str(ruta))
        except SyntaxError as e:
            pytest.fail(f"{ruta.relative_to(RAIZ)} no compila: linea {e.lineno}: {e.msg}")

    def test_hay_algo_que_compilar(self):
        """Guard del guard: si el glob dejara de encontrar archivos, el test de arriba pasaria
        vacio y no diria nada."""
        nombres = {p.name for p in _fuentes()}
        assert {"parseo.py", "embeddings.py", "carga.py", "quickstart.py"} <= nombres


class TestDependenciasDeclaradas:
    @staticmethod
    def _requeridas() -> set:
        texto = (RAIZ / "requirements.txt").read_text(encoding="utf-8")
        paquetes = set()
        for linea in texto.splitlines():
            linea = linea.split("#")[0].strip()
            if linea:
                paquetes.add(re.split(r"[<>=!\[ ]", linea)[0].lower())
        return paquetes

    @pytest.mark.parametrize("paquete", ["neo4j", "pymupdf", "python-dotenv", "pyyaml",
                                         "pytest", "ruff", "httpx", "jsonschema"])
    def test_requirements_declara_lo_que_la_suite_y_el_pipeline_usan(self, paquete):
        """`pyyaml` lo importa `pipeline/perfiles.py` (perezoso, pero sin el no hay perfil),
        `pytest` y `ruff` son el gate de CI, `httpx` y `jsonschema` son la suite de admin/v1
        (TestClient y la validacion contra los schemas del contrato), y el resto es el pipeline."""
        assert paquete in self._requeridas(), f"{paquete} no esta en requirements.txt"

    def test_no_se_declara_lo_que_el_engine_no_tiene(self):
        """`markitdown` es la conversion de DOCX/HTML/EPUB del privado, que NO esta replicada
        aca (ver README, "Not in this mirror yet"). Declararla haria creer que si."""
        assert "markitdown" not in self._requeridas()


class TestDependenciasDeLaAPI:
    """`api/requirements.txt` es OTRO archivo y OTRA doctrina: construye una imagen, asi que va
    pineado con `==`. Un `>=` ahi significa que el mismo commit produce una imagen distinta cada
    vez que alguien publica en PyPI, y entonces un rollback no significa nada."""

    @staticmethod
    def _declaradas() -> dict:
        texto = (RAIZ / "api" / "requirements.txt").read_text(encoding="utf-8")
        salida = {}
        for linea in texto.splitlines():
            linea = linea.split("#")[0].strip()
            if linea:
                nombre = re.split(r"[<>=!\[ ]", linea)[0].lower()
                salida[nombre] = linea
        return salida

    @pytest.mark.parametrize("paquete", ["fastapi", "uvicorn", "neo4j", "google-genai",
                                         "python-dotenv", "pymupdf", "slowapi", "pyyaml", "mcp"])
    def test_declara_lo_que_la_api_importa(self, paquete):
        """`pymupdf` porque la API importa `pipeline/parseo.py`, `pyyaml` porque el descriptor de
        admin/v1 se arma desde el perfil de dominio, y `mcp` porque `api/routers/mcp_remote.py`
        monta el endpoint MCP dentro de la misma app (17-sep-2026)."""
        assert paquete in self._declaradas(), f"{paquete} no esta en api/requirements.txt"

    def test_todas_las_dependencias_de_la_imagen_estan_pineadas(self):
        sueltas = [linea for linea in self._declaradas().values() if "==" not in linea]
        assert not sueltas, f"sin pin exacto: {sueltas}"


class TestNadaDeSecretos:
    #: Formas de una credencial de verdad. No se busca la palabra "key": se buscan las FORMAS, con
    #: el largo incluido, porque sin largo `AKIA` matchea cualquier prosa que lo mencione.
    #:
    #: EL BARRIDO SE INCLUYE A SI MISMO, y por eso cada aguja esta escrita de modo que no se
    #: matchee: el prefijo va pegado a una clase de caracteres (`gh[pousr]_[A-Za-z0-9]{36,}`), asi
    #: que en el texto de ESTE archivo lo que sigue al prefijo es un `[` y la corrida mide cero. Un
    #: archivo excluido del scanner es un agujero, aunque sea el del scanner.
    HUELLAS = (
        re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),                        # Google API key
        re.compile(r"neo4j\+s://[a-z0-9]{6,}\.databases\.neo4j\.io"),  # instancia AuraDB real
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),                            # API key estilo OpenAI
        re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),                     # token clasico de GitHub
        re.compile(r"github_pat_[A-Za-z0-9_]{60,}"),                   # token fine-grained
        re.compile(r"whsec_[A-Za-z0-9]{24,}"),                         # secreto de webhook
        re.compile(r"\b(?:pk|sk)_live_[A-Za-z0-9]{16,}"),              # clave de PRODUCCION
        re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),                   # token de Slack
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                           # access key id de AWS
        re.compile(r'"type"\s*:\s*"service_account"'),                 # JSON de service account
    )

    def test_el_gitignore_ignora_el_env(self):
        lineas = {l.strip() for l in
                  (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()}
        assert ".env" in lineas

    def test_el_gitignore_ignora_los_documentos_y_el_catalogo(self):
        """M-1: lo que entra por accidente cuando alguien prueba el pipeline con SUS documentos.
        `catalog.json` es la lista de la biblioteca de quien corra esto: el repo trae el ejemplo."""
        lineas = {l.strip() for l in
                  (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()}
        assert {"*.pdf", "examples/*", "!examples/README.md", ".venv/", "venv/", "/data/",
                "catalog.json"} <= lineas, sorted(lineas)

    def test_el_catalogo_que_viaja_es_el_de_ejemplo(self):
        """A-2: el catalogo REAL nombraba obra por obra el corpus de la instancia privada. El que
        se publica trae dos entradas inventadas, sin fechas y sin lista de pendientes.

        NO se exige que `catalog.json` no exista: el README le dice a quien clona que lo cree
        (`cp catalog.example.json catalog.json`), y un test que castigue el flujo documentado es
        un test roto. Que no se versione lo garantiza el `.gitignore`, y eso lo fija el test de
        arriba."""
        import json

        ejemplo = json.loads((RAIZ / "catalog.example.json").read_text(encoding="utf-8"))
        assert [l["id"] for l in ejemplo["libros"]] == ["ejemplo-tratado-1", "ejemplo-manual-2"]
        assert all(l["fecha_parseo"] is None for l in ejemplo["libros"])
        assert "libros_faltantes" not in ejemplo

    @pytest.mark.parametrize("nombre", [".env.example", "docker-compose.yml", "README.md"])
    def test_los_archivos_de_ejemplo_traen_placeholders(self, nombre):
        texto = (RAIZ / nombre).read_text(encoding="utf-8")
        for huella in self.HUELLAS:
            assert not huella.search(texto), f"{nombre} parece traer una credencial real"

    @pytest.mark.parametrize("ruta", _archivos(), ids=lambda p: str(p.relative_to(RAIZ)))
    def test_ningun_archivo_trae_una_credencial(self, ruta):
        texto = ruta.read_text(encoding="utf-8", errors="replace")
        for huella in self.HUELLAS:
            assert not huella.search(texto), \
                f"{ruta.relative_to(RAIZ)}: parece traer una credencial ({huella.pattern})"

    def test_el_barrido_llega_a_los_archivos_que_no_son_python(self):
        """Guard del guard: si el glob volviera a mirar solo `*.py`, el test de arriba pasaria
        igual y no diria nada. Se nombran uno de cada tipo, el scanner incluido."""
        relativos = {str(p.relative_to(RAIZ)).replace("\\", "/") for p in _archivos()}
        for esperado in ("README.md", ".env.example", "docker-compose.yml", "ruff.toml",
                         "profiles/medicina.yaml", "catalog.example.json", "pipeline/parseo.py",
                         "tests/test_repo.py", ".github/workflows/ci.yml"):
            assert esperado in relativos, (esperado, sorted(relativos)[:20])


class TestNadaDeCorpusAjeno:
    """LO QUE SE SACO EL 13-sep-2026 Y NO PUEDE VOLVER (items C-1/C-2 y A-3 de la auditoria).

    `pipeline/parseo.py` borraba lineas de PDF con regex que nombraban al titular de los derechos y
    al sitio del que habia salido un archivo. En un espejo publico eso no es una regla de limpieza:
    es un recibo de la procedencia del corpus, y viajaba en cada release. La cura fue reemplazarlas
    por un filtro de FORMA (repeticion de encabezado/pie + aviso de derechos sin nombres). Este test
    es el unico que impide que vuelvan, y por eso barre TODO el arbol y no solo ese archivo.

    LAS AGUJAS ESTAN PARTIDAS a proposito, con una clase de un solo caracter en el medio
    (`book[s]medicos`): el literal entero no aparece en este archivo, asi que el barrido puede
    incluirse a si mismo sin acusarse. Si alguien "arregla" el regex escribiendo la palabra
    completa, el test empieza a fallar sobre este archivo, que es exactamente lo que tiene que pasar.
    """

    PROHIBIDOS = (
        (re.compile(r"book[s]medicos", re.I), "el sitio del que salio un PDF"),
        (re.compile(r"El[s]evier", re.I), "el titular de los derechos de un libro"),
        (re.compile(r"©\s*(?:\d{4}\s*)?[A-Z][a-záéíóúñ]"), "un © con un nombre propio detras"),
    )

    @pytest.mark.parametrize("ruta", _archivos(), ids=lambda p: str(p.relative_to(RAIZ)))
    def test_ningun_archivo_nombra_un_titular_ni_un_sitio(self, ruta):
        texto = ruta.read_text(encoding="utf-8", errors="replace")
        for aguja, que_es in self.PROHIBIDOS:
            assert not aguja.search(texto), \
                f"{ruta.relative_to(RAIZ)}: aparece {que_es} ({aguja.pattern})"

    def test_el_filtro_de_maqueta_sigue_siendo_de_forma(self):
        """La mitad positiva: que el reemplazo EXISTA. Sin esto, alguien podria borrar el filtro
        entero y el test de arriba quedaria verde por ausencia."""
        from pipeline import parseo

        # El aviso se escribe con el escape del simbolo (`©`) por la misma razon que las
        # agujas van partidas: el literal completo haria que este archivo se acuse a si mismo.
        assert parseo.clean_text("\u00a9 2024 Casa Editora") == ""
        assert parseo.lineas_repetidas is not None
        assert parseo.MIN_PAGINAS_REPETIDAS >= 2
