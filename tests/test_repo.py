"""Salud del repo publico: que todo compile, que las dependencias esten declaradas y que no
haya un secreto adentro.

POR QUE ESTOS TRES. Es un repo que la gente clona: los tres fallos que arruinan ese primer
minuto no son de logica.
  1. Un archivo que no PARSEA. `mcp_server.py` estuvo asi desde v1.0 —una edicion de limpieza
     se comio una coma al sacar una URL privada— y `python mcp_server.py` moria en el import.
     Nadie lo vio porque no habia un test que abriera los archivos.
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
IGNORADOS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules",
             "parsed", "extracted", "examples"}


def _fuentes() -> list:
    return sorted(p for p in RAIZ.rglob("*.py")
                  if not IGNORADOS & set(p.relative_to(RAIZ).parts))


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
                                         "pytest", "ruff"])
    def test_requirements_declara_lo_que_la_suite_y_el_pipeline_usan(self, paquete):
        """`pyyaml` lo importa `pipeline/perfiles.py` (perezoso, pero sin el no hay perfil),
        `pytest` y `ruff` son el gate de CI, y los otros tres son el pipeline."""
        assert paquete in self._requeridas(), f"{paquete} no esta en requirements.txt"

    def test_no_se_declara_lo_que_el_engine_no_tiene(self):
        """`markitdown` es la conversion de DOCX/HTML/EPUB del privado, que NO esta replicada
        aca (ver README, "Not in this mirror yet"). Declararla haria creer que si."""
        assert "markitdown" not in self._requeridas()


class TestNadaDeSecretos:
    #: Formas de una credencial de verdad. No se busca la palabra "key": se buscan las FORMAS.
    HUELLAS = (
        re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),                     # Google API key
        re.compile(r"neo4j\+s://[a-z0-9]{6,}\.databases\.neo4j\.io"),  # instancia AuraDB real
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
    )

    def test_el_gitignore_ignora_el_env(self):
        lineas = {l.strip() for l in
                  (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()}
        assert ".env" in lineas

    @pytest.mark.parametrize("nombre", [".env.example", "docker-compose.yml", "README.md"])
    def test_los_archivos_de_ejemplo_traen_placeholders(self, nombre):
        texto = (RAIZ / nombre).read_text(encoding="utf-8")
        for huella in self.HUELLAS:
            assert not huella.search(texto), f"{nombre} parece traer una credencial real"

    @pytest.mark.parametrize("ruta", _fuentes(), ids=lambda p: str(p.relative_to(RAIZ)))
    def test_ningun_archivo_python_trae_una_credencial(self, ruta):
        texto = ruta.read_text(encoding="utf-8")
        for huella in self.HUELLAS:
            assert not huella.search(texto), f"{ruta.relative_to(RAIZ)}: credencial hardcodeada"
