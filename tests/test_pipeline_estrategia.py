"""pipeline/estrategia.py + pipeline/perfiles.py — el perfil de dominio gobernando el parseo.

POR QUE. `estrategia.py` saco del codigo los parametros que eran de MEDICINA y no del pipeline:
los patrones de estructura y los tamanos de chunk. El riesgo de un movimiento asi es silencioso:
si el default se corre un solo numero, el material nuevo se chunkea distinto del viejo y nadie
se entera hasta que una busqueda devuelve pasajes cortados. Estos tests fijan las dos mitades
del trato:

  1. sin perfil, byte por byte lo de siempre (los patrones con sus acentos incluidos);
  2. con perfil, los numeros y los patrones del perfil ganan, y una combinacion incoherente
     falla al construirse y no al chunkear;

y una tercera cosa que es del engine: que los YAML del disco (`profiles/`) se resuelvan
relativo a la raiz del repo, carguen, compilen sus regex y digan lo que el codigo dice.
"""
import re
from pathlib import Path

import pytest

from pipeline import estrategia as est
from pipeline import parseo, perfiles

RAIZ = Path(__file__).resolve().parent.parent


class TestDefaultIntacto:
    """Lo que regia antes de existir la estrategia tiene que seguir rigiendo sin perfil."""

    def test_los_numeros_del_default_son_los_historicos(self):
        e = est.POR_DEFECTO
        assert (e.target_size, e.min_size, e.max_size) == (280, 150, 380)
        assert (e.overlap, e.parent_window, e.max_parent_words) == (60, 3, 1200)

    def test_parseo_reexporta_las_mismas_constantes(self):
        # parser_v2.py y los scripts viejos importan de parseo: el traslado no puede romperlos.
        assert parseo.TARGET_SIZE == est.TARGET_SIZE
        assert parseo.MAX_SIZE == est.MAX_SIZE
        assert parseo.MIN_SIZE == est.MIN_SIZE
        assert parseo.OVERLAP_SIZE == est.OVERLAP_SIZE
        assert parseo.STRUCTURE_PATTERNS is est.PATRONES_MEDICINA

    def test_los_patrones_conservan_sus_acentos(self):
        # Se movieron copiando bytes justamente para no perder el acento al retipearlos.
        capitulo = est.PATRONES_MEDICINA["farreras-2020"]["capitulo"]
        assert r"^SECCIÓN\s+[IVXLCDM]+\b" in capitulo
        assert r"^Capítulo\s+\d+" in capitulo

    def test_patrones_de_cae_al_default_para_una_fuente_desconocida(self):
        assert est.POR_DEFECTO.patrones_de("no-existe") == est.PATRONES_MEDICINA["_default"]
        assert est.POR_DEFECTO.patrones_de("farreras-2020") == \
            est.PATRONES_MEDICINA["farreras-2020"]

    def test_perfil_vacio_devuelve_el_default(self):
        assert est.desde_perfil(None) is est.POR_DEFECTO
        assert est.desde_perfil({}) is est.POR_DEFECTO
        # Un perfil que solo declara entidades no toca el chunkeo.
        assert est.desde_perfil({"entities": [{"id": "patologia"}]}).target_size == 280


class TestPerfilManda:
    """Un dominio con otra unidad de sentido declara otros numeros y el pipeline los obedece."""

    def test_los_numeros_del_perfil_pisan_el_default(self):
        # Derecho: el articulo es la unidad, chunks mas grandes y sin casi solape.
        e = est.desde_perfil({"chunk": {"target_size": 700, "min_size": 200, "max_size": 1200,
                                       "overlap": 0, "parent_window": 1,
                                       "max_parent_words": 4000}})
        assert (e.target_size, e.max_size, e.overlap, e.parent_window) == (700, 1200, 0, 1)

    def test_un_perfil_parcial_deja_el_resto_en_default(self):
        e = est.desde_perfil({"chunk": {"target_size": 320}})
        assert e.target_size == 320
        assert e.overlap == est.OVERLAP_SIZE and e.max_size == est.MAX_SIZE

    def test_los_patrones_del_perfil_reemplazan_al_generico(self):
        e = est.desde_perfil({"parse": {"structure_patterns": {
            "capitulo": [r"^LIBRO\s+[IVXLCDM]+"], "seccion": [r"^ART[IÍ]CULO\s+\d+"]}}})
        assert e.patrones_de("cualquiera") == {
            "capitulo": [r"^LIBRO\s+[IVXLCDM]+"], "seccion": [r"^ART[IÍ]CULO\s+\d+"]}

    def test_una_fuente_puede_tener_su_excepcion(self):
        e = est.desde_perfil({"parse": {"por_fuente": {"cn-1994": {"capitulo": [r"^PARTE\s+\w+"]}}}})
        assert e.patrones_de("cn-1994")["capitulo"] == [r"^PARTE\s+\w+"]
        assert e.patrones_de("otra") == est.PATRONES_MEDICINA["_default"]

    def test_con_devuelve_una_copia_sin_tocar_el_original(self):
        otra = est.POR_DEFECTO.con(target_size=500)
        assert otra.target_size == 500 and est.POR_DEFECTO.target_size == 280


class TestFallaCerrado:
    """Numeros incoherentes producen chunks basura que se descubren con el corpus ya cargado."""

    @pytest.mark.parametrize("chunk", [
        {"min_size": 400},                      # min > target
        {"target_size": 500},                   # target > max
        {"overlap": 150},                       # overlap >= min_size: el chunk es casi solape
        {"parent_window": 0},                   # un padre sin hijos
        {"max_parent_words": 100},              # el padre no entra ni un hijo
    ])
    def test_combinacion_incoherente_no_se_construye(self, chunk):
        with pytest.raises(ValueError):
            est.desde_perfil({"chunk": chunk})


class TestElPipelineUsaLaEstrategia:
    """No alcanza con que el objeto exista: las funciones tienen que mirarlo."""

    @staticmethod
    def _paginas(n_palabras: int) -> list:
        texto = " ".join(f"palabra{i}" for i in range(n_palabras))
        return [{"page": 1, "text": texto, "titulo_capitulo": None, "titulo_seccion": None}]

    def test_detect_structure_usa_los_patrones_de_la_estrategia(self):
        paginas = [{"page": 1, "text": "LIBRO II\nDe las obligaciones y su cumplimiento"}]
        derecho = est.desde_perfil({"parse": {"structure_patterns": {
            "capitulo": [r"^LIBRO\s+[IVXLCDM]+"], "seccion": []}}})
        assert parseo.detect_structure(paginas, "cc-2015", derecho)[0]["titulo_capitulo"] == \
            "LIBRO II"
        # El default de medicina no conoce "LIBRO": la diferencia es la estrategia, no el texto.
        # (sin match, detect_structure deja el titulo vacio, no None)
        assert parseo.detect_structure(paginas, "cc-2015")[0]["titulo_capitulo"] == ""

    def test_chunks_mas_grandes_con_otra_estrategia(self):
        paginas = self._paginas(3000)
        chicos, _ = parseo.generate_chunks_v2(paginas, "x")
        grandes, _ = parseo.generate_chunks_v2(
            paginas, "x", estrategia=est.POR_DEFECTO.con(target_size=700, max_size=900,
                                                         overlap=0))
        assert len(grandes) < len(chicos)
        assert max(c["word_count"] for c in grandes) > max(c["word_count"] for c in chicos)

    def test_el_default_de_generate_chunks_v2_no_cambio(self):
        # La firma dice None, pero el resultado tiene que ser el de las constantes de siempre.
        paginas = self._paginas(3000)
        por_defecto, padres_d = parseo.generate_chunks_v2(paginas, "x")
        explicito, padres_e = parseo.generate_chunks_v2(
            paginas, "x", target_size=est.TARGET_SIZE, overlap=est.OVERLAP_SIZE)
        assert [c["text"] for c in por_defecto] == [c["text"] for c in explicito]
        assert [p["text"] for p in padres_d] == [p["text"] for p in padres_e]

    def test_los_argumentos_sueltos_siguen_ganando(self):
        # Llamadores viejos pasan target_size/overlap sin saber de estrategias.
        paginas = self._paginas(2000)
        chunks, _ = parseo.generate_chunks_v2(paginas, "x", target_size=600, overlap=0,
                                              estrategia=est.POR_DEFECTO.con(max_size=900))
        assert max(c["word_count"] for c in chunks) > est.TARGET_SIZE


class TestLosPerfilesDelDisco:
    """`profiles/` es donde el dominio se declara, y es lo que un usuario del engine edita
    para agregar el suyo. Tres cosas tienen que ser ciertas o el YAML es decoracion:
    que se encuentre, que cargue, y que diga lo mismo que el codigo."""

    def test_el_directorio_se_resuelve_relativo_a_la_raiz_del_repo(self):
        assert perfiles.RAIZ == RAIZ
        assert perfiles.DIRECTORIO == RAIZ / "profiles"
        assert perfiles.ruta_de("medicina") == RAIZ / "profiles" / "medicina.yaml"

    def test_estan_los_perfiles_que_el_engine_publica(self):
        """`medicina` (el default de la instancia) y `generico` (el dominio neutral). El
        `derecho.yaml` del privado NO esta vendorizado aca: ver el README."""
        en_disco = {p.stem for p in perfiles.DIRECTORIO.glob("*.yaml")}
        assert {"medicina", "generico"} <= en_disco, en_disco

    def test_medicina_del_yaml_es_el_default_del_codigo(self):
        e = perfiles.estrategia("medicina")
        assert e == est.POR_DEFECTO, (
            "profiles/medicina.yaml dejo de coincidir con el default: un corpus ya cargado se "
            "procesó con el default, re-chunkear con otros numeros lo parte al medio")
        assert e.patrones == est.PATRONES_MEDICINA

    def test_el_perfil_generico_es_el_default_historico(self):
        """Que "otros" sea EXPLICITO, no el default silencioso de medicina. Explicito y, hoy,
        identico: este test es el que garantiza que declararlo no cambio el comportamiento."""
        assert perfiles.estrategia("generico") == est.POR_DEFECTO
        crudo = perfiles.cargar("generico")
        assert crudo["profile"] == "generico" and crudo["version"] == 1
        assert crudo["parse"]["structure_patterns"]["capitulo"] == \
            est.PATRONES_MEDICINA["_default"]["capitulo"]
        assert crudo["parse"]["structure_patterns"]["seccion"] == \
            est.PATRONES_MEDICINA["_default"]["seccion"]

    def test_la_taxonomia_generica_es_minima_y_neutral(self):
        """Domain-neutral: ningun tipo de medicina colado. Un vocabulario grande que nadie
        valido produce extraccion ruidosa."""
        crudo = perfiles.cargar("generico")
        assert [e["id"] for e in crudo["entities"]] == ["concepto", "procedimiento",
                                                        "organizacion"]
        assert [r["id"] for r in crudo["relations"]] == ["RELACIONADO_CON", "PARTE_DE"]
        texto = perfiles.ruta_de("generico").read_text(encoding="utf-8").lower()
        for ajeno in ("patologia", "farmaco", "sintoma", "legal_"):
            assert ajeno not in texto, ajeno

    @pytest.mark.parametrize("ruta", sorted(perfiles.DIRECTORIO.glob("*.yaml")),
                             ids=lambda p: p.stem)
    def test_cada_perfil_del_disco_carga_y_sus_regex_compilan(self, ruta):
        """Recorre el DIRECTORIO, no una lista: un dominio nuevo queda cubierto sin tocar
        este archivo. Un regex roto en el YAML no falla al cargar — falla parseando, con el
        documento ya subido."""
        estrategia = perfiles.estrategia(ruta.stem)
        for grupo in estrategia.patrones.values():
            for lista in grupo.values():
                for patron in lista:
                    re.compile(patron)

    def test_el_default_de_la_instancia_es_medicina(self):
        assert perfiles.POR_DEFECTO == "medicina"
        assert perfiles.ruta_de().stem == "medicina"

    def test_un_dominio_sin_perfil_falla_cerrado(self):
        with pytest.raises(FileNotFoundError):
            perfiles.cargar("contabilidad")
