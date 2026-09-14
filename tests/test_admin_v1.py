"""admin/v1 del engine: que lo que sale por la API sea EXACTAMENTE lo que dice el contrato.

QUE FIJA ESTE ARCHIVO, y por que cada cosa:

  1. El descriptor y cada fuente VALIDAN contra los JSON Schemas de `nomos-contracts` (copias en
     `tests/contracts/`, con un test de paridad que se saltea si el contrato no esta en disco).
     Un descriptor que no valida no lo rechaza nadie hasta que la consola se queda en blanco.
  2. EL DESCRIPTOR SALE DEL PERFIL ACTIVO. Es la prueba de que esto es domain-agnostic y no
     "medicina con otro nombre": con `PROFILE=generico` cambian dominio, entidades y relaciones
     sin tocar una linea de codigo. Si alguien hardcodea una constante de medicina aca, este test
     se pone rojo.
  3. Cada capacidad en false dice POR QUE y COMO se prende. La consola esconde el boton de lo que
     esta apagado: sin nota, quien administra ve una pantalla a la que le falta algo.
  4. Las rutas: auth (el contrato pide 401 hasta en /admin/v1/health), la forma de la pagina, el
     404 con `{detail, code}` y el grafo caido, que es 503 en /stats y 200 con `graph: down` en
     /health —un health que falla cuando el grafo falla no sirve para diagnosticar nada—.

SIN NEO4J Y SIN RED: `services.graph` se reemplaza por un doble que responde las consultas del
modulo por identidad (si alguien cambia un Cypher y no el doble, el test dice cual falta, en vez
de devolver una lista vacia que pasa desapercibida). Los ids son los del `catalog.example.json`.
"""
import json
import os
from pathlib import Path

import pytest

# LAS DEPENDENCIAS DE LA API NO ESTAN EN requirements.txt: viven en api/requirements.txt, que es
# lo que construye la imagen. Quien clona el repo para usar SOLO el pipeline corre `pytest` sin
# fastapi instalado, y este archivo no puede voltearle la suite entera; quien corre la API si
# tiene que verlo fallar. Por eso el skip es condicional: CI exporta
# MEDGRAPH_ENGINE_REQUIRE_API=1 y ahi un skip es una FALLA, no un archivo que pasa en silencio
# (misma doctrina que los markers `integration`/`neo4j` de pytest.ini).
try:
    import fastapi  # noqa: F401
    import slowapi  # noqa: F401
except ModuleNotFoundError as e:  # pragma: no cover - depende del entorno, no del codigo
    if os.getenv("MEDGRAPH_ENGINE_REQUIRE_API") == "1":
        raise
    pytest.skip(f"falta {e.name}: la suite de la API necesita "
                "`pip install -r api/requirements.txt`", allow_module_level=True)

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

import main
from pipeline import perfiles
from services import admin_v1, graph
from services.autorizacion import AdminError
from services.settings import get_settings

RAIZ = Path(__file__).resolve().parent.parent
CONTRATOS = RAIZ / "tests" / "contracts"
#: El repo del contrato, si esta en el mismo disco. Sin el, los tests de paridad se saltean.
NOMOS = RAIZ.parent.parent / "nomos" / "nomos-contracts"


def _schema(nombre: str) -> dict:
    return json.loads((CONTRATOS / f"{nombre}.schema.json").read_text(encoding="utf-8"))


def _validar(nombre: str, doc: dict) -> None:
    errores = sorted(Draft202012Validator(_schema(nombre)).iter_errors(doc),
                     key=lambda e: list(e.path))
    assert not errores, [f"{'/'.join(map(str, e.path))}: {e.message}" for e in errores][:3]


# ---------------------------------------------------------------- el grafo doble
#: Una fuente registrada (tiene `:Book`) y completa.
FILA_REGISTRADA = {
    "id": "ejemplo-tratado-1", "title": "Tratado de Ejemplo", "author": "Autora De Ejemplo, A.",
    "edition": "1", "pages": 1240, "ingested_at": "2026-01-02T03:04:05Z", "ingested_by": "cli",
    "updated_at": "2026-01-02T03:04:05Z",
    "units": 3980, "units_embedded": 3980, "ultima_pagina": 1240,
}
#: Una fuente que existe solo como chunks: alguien cargo con `carga.cargar_libro` y nunca llamo a
#: `carga.registrar_fuente`. Ademas quedo a medias de embeddings (status `parcial`).
FILA_HUERFANA = {
    "id": "ejemplo-manual-2", "units": 902, "units_embedded": 700, "ultima_pagina": 312,
}


class GrafoDoble:
    """Responde por identidad de Cypher y anota lo que le preguntaron."""

    def __init__(self, fuentes=None, huerfanas=None, revienta=False):
        self.fuentes = FILA_REGISTRADA if fuentes is None else fuentes
        self.huerfanas = FILA_HUERFANA if huerfanas is None else huerfanas
        self.revienta = revienta
        self.consultas = []

    def query(self, cypher, params=None, retries=2):
        if self.revienta:
            raise RuntimeError("el grafo no contesta")
        self.consultas.append(cypher)
        if cypher.strip() == "RETURN 1":  # el ping de vida (graph.db_state y /health)
            return [{"1": 1}]
        if cypher == admin_v1.Q_FUENTES:
            return [self.fuentes] if self.fuentes else []
        if cypher == admin_v1.Q_FUENTES_HUERFANAS:
            return [self.huerfanas] if self.huerfanas else []
        if cypher == admin_v1.Q_UNIDADES:
            return [{"units": 4882, "units_embedded": 4680}]
        if cypher == admin_v1.Q_FUENTES_CON_CHUNKS:
            return [{"n": 2}]
        if cypher == admin_v1.Q_FUENTES_SIN_CHUNKS:
            return [{"n": 0}]
        raise AssertionError(f"el doble no conoce esta consulta:\n{cypher}")

    def get_stats(self):
        if self.revienta:
            raise RuntimeError("el grafo no contesta")
        return {"total_nodos": 5120, "total_relaciones": 9400,
                "nodos": {"Chunk": 4882, "Book": 1}, "relaciones": {"SIGUE_A": 4880}}

    def db_state(self):
        return "down" if self.revienta else "connected"


@pytest.fixture
def doble(monkeypatch):
    d = GrafoDoble()
    monkeypatch.setattr(graph, "query", d.query)
    monkeypatch.setattr(graph, "get_stats", d.get_stats)
    monkeypatch.setattr(graph, "db_state", d.db_state)
    admin_v1.reiniciar_cache()
    yield d
    admin_v1.reiniciar_cache()


@pytest.fixture
def grafo_caido(monkeypatch):
    d = GrafoDoble(revienta=True)
    monkeypatch.setattr(graph, "query", d.query)
    monkeypatch.setattr(graph, "get_stats", d.get_stats)
    monkeypatch.setattr(graph, "db_state", d.db_state)
    admin_v1.reiniciar_cache()
    yield d
    admin_v1.reiniciar_cache()


@pytest.fixture
def cliente(doble):
    """La API entera, con la credencial que la instancia tiene configurada."""
    with TestClient(main.app) as c:
        c.headers.update({"Authorization": f"Bearer {get_settings().api_key}"})
        yield c


@pytest.fixture
def perfil_generico(monkeypatch):
    """Cambia el perfil ACTIVO de la instancia, como lo haria una variable de entorno."""
    monkeypatch.setenv("PROFILE", "generico")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------- descriptor
class TestDescriptor:
    def test_cumple_el_contrato(self):
        _validar("admin-descriptor", admin_v1.descriptor())

    def test_los_pasos_son_estados_de_job_v1(self):
        """El stepper de la consola marca `job.steps` contra estos ids: si no son estados de
        job/v1, no hay traduccion posible y el monitor no dibuja nada."""
        estados = set(_schema("job")["properties"]["state"]["enum"])
        assert {p["id"] for p in admin_v1.PIPELINE_STEPS} <= estados

    def test_el_recorrido_publicado_es_el_de_ESTE_pipeline(self):
        """`loaded` ANTES de `vectorized`, que es lo que este repo hace de verdad:
        `pipeline/embeddings.py` embebe los chunks que YA estan en el grafo. Publicar el orden
        canonico seria copiar el de otra instancia y describir algo que aca no pasa."""
        ids = [p["id"] for p in admin_v1.PIPELINE_STEPS]
        assert ids.index("loaded") < ids.index("vectorized")
        assert "uploaded" not in ids and "pending_review" not in ids, \
            "no hay subida ni gate humano: un paso que nunca ocurre es un stepper que miente"

    def test_el_vocabulario_sale_del_perfil_activo(self):
        """EL TEST QUE IMPORTA: sin esto, 'domain-agnostic' es una palabra en el README."""
        d = admin_v1.descriptor()
        assert d["domain"] == "medicina" and d["profile"].startswith("medicina@")
        assert any(e["label"] == "Patologia" for e in d["entity_types"])
        assert "CAUSADA_POR" in d["relation_types"]

    def test_cambiar_el_perfil_cambia_el_descriptor_entero(self, perfil_generico):
        """Se compara contra EL PERFIL, no contra una lista escrita aca: lo que se fija es que el
        descriptor SALGA del YAML, no cuales son hoy las entidades de `generico` (eso es del
        perfil, y cambiarlo no tiene por que poner rojo este test)."""
        p = perfiles.cargar("generico")
        d = admin_v1.descriptor()
        assert d["domain"] == "generico" and d["profile"] == f"generico@{p['version']}"
        assert [e["id"] for e in d["entity_types"]] == [e["id"] for e in p["entities"]]
        assert d["relation_types"] == [r["id"] for r in p["relations"]]
        assert not any(e["label"] == "Patologia" for e in d["entity_types"]), \
            "el descriptor seguia hablando de medicina con otro perfil activo"
        _validar("admin-descriptor", d)

    def test_los_campos_publicados_son_los_que_el_grafo_sabe_leer(self):
        """`CAMPO_A_PROP` es la unica tabla que traduce nombre del contrato -> propiedad del
        `:Book`. Un campo publicado que no este ahi sale siempre vacio."""
        assert {f["name"] for f in admin_v1.SOURCE_FIELDS} == set(admin_v1.CAMPO_A_PROP)

    def test_ningun_campo_es_editable_mientras_no_haya_patch(self):
        assert not admin_v1.CAPABILITIES["edit"]
        assert not any(f["editable"] for f in admin_v1.SOURCE_FIELDS), \
            "un campo editable sin PATCH es un formulario que pierde lo que la persona escribio"

    def test_cada_capacidad_en_false_dice_por_que_y_como(self):
        notas = admin_v1.CAPABILITY_NOTES
        for cap, prendida in admin_v1.CAPABILITIES.items():
            if prendida:
                continue
            assert cap in notas, f"{cap} esta en false y no dice por que ni como se prende"
            n = notas[cap]
            assert n["kind"] in ("domain_decision", "not_built", "not_configured"), cap
            assert len(n["reason"]) >= 12, cap
            # `null` no es un campo sin escribir: es "no hay camino", y solo una decision de
            # dominio puede decir eso.
            assert (n["how_to_unblock"] is None) == (n["kind"] == "domain_decision"), cap
        assert admin_v1.descriptor()["capability_notes"] == notas, "la nota tiene que SALIR por la API"

    def test_ninguna_capacidad_prendida_lleva_nota(self):
        """Una nota que sobra miente al reves: dice que algo no se puede hacer cuando si se puede.
        El schema no lo agarra (pide nota para cada false; no puede prohibir la de un true)."""
        prendidas = {k for k, v in admin_v1.CAPABILITIES.items() if v}
        assert not prendidas & set(admin_v1.CAPABILITY_NOTES)

    def test_no_se_publica_lo_que_no_hay(self):
        """`accepted_formats` y `unit_types` acompañan a capacidades que aca estan apagadas:
        publicarlos seria ofrecer una puerta que no existe (y el schema los exige al reves)."""
        d = admin_v1.descriptor()
        assert "accepted_formats" not in d and "unit_types" not in d
        assert not d["capabilities"]["upload"] and not d["capabilities"]["units"]

    def test_la_credencial_publicada_es_la_que_la_api_pide(self):
        """Sin Clerk: este repo tiene UNA api key y ningun modelo de roles. Publicar `clerk_jwt`
        haria que la consola ofrezca un login que este backend no valida."""
        assert admin_v1.descriptor()["auth"]["schemes"] == ["api_key"]

    def test_schemas_vendorizados_identicos_al_contrato(self):
        if not NOMOS.exists():
            pytest.skip("nomos-contracts no esta en este disco")
        for n in ("admin-descriptor", "admin-source", "job"):
            assert (CONTRATOS / f"{n}.schema.json").read_text(encoding="utf-8") == \
                (NOMOS / "schemas" / f"{n}.schema.json").read_text(encoding="utf-8"), n

    def test_un_perfil_que_no_existe_falla_cerrado(self, monkeypatch):
        """Sin perfil no se sabe que es lo que se esta administrando. `main.py` lo carga al
        arrancar por esto mismo: un descriptor vacio es peor que un servicio que no levanta."""
        monkeypatch.setenv("PROFILE", "no-existe")
        get_settings.cache_clear()
        try:
            with pytest.raises(FileNotFoundError):
                admin_v1.descriptor()
        finally:
            get_settings.cache_clear()

    def test_el_perfil_del_engine_es_el_que_lee_el_pipeline(self):
        """El descriptor y el chunkeo tienen que salir del MISMO archivo: si el descriptor leyera
        una copia propia, la consola podria anunciar un dominio y el parser usar otro."""
        assert perfiles.ruta_de(get_settings().profile).exists()
        assert admin_v1.perfil() is perfiles.cargar(get_settings().profile)


# ---------------------------------------------------------------- estado
class TestEstado:
    def test_health_con_grafo_vivo(self, doble):
        h = admin_v1.health()
        assert h["status"] == "ok" and h["graph"] == "connected"
        assert h["wake_eta_s"] is None, "esta instancia no despierta a nadie: prometer un ETA seria inventar"
        assert h["checked_at"].endswith("Z")

    def test_health_con_grafo_caido_sigue_respondiendo(self, grafo_caido):
        h = admin_v1.health()
        assert h["status"] == "degraded" and h["graph"] == "down"

    def test_stats_usa_los_nombres_del_contrato(self, doble):
        s = admin_v1.stats()
        assert s["nodes_total"] == 5120 and s["relationships_total"] == 9400
        assert s["nodes_by_label"]["Chunk"] == 4882
        assert (s["units_total"], s["units_embedded"], s["sources_total"]) == (4882, 4680, 2)

    def test_stats_se_cachea(self, doble):
        admin_v1.stats()
        consultadas = len(doble.consultas)
        admin_v1.stats()
        assert len(doble.consultas) == consultadas, "contar un grafo entero no se hace dos veces seguidas"


# ---------------------------------------------------------------- fuentes
class TestFuentes:
    def test_cada_fuente_cumple_el_contrato(self, doble):
        for f in admin_v1.listar_fuentes()["items"]:
            _validar("admin-source", f)

    def test_la_fuente_registrada_trae_su_metadata(self, doble):
        f = admin_v1.obtener_fuente("ejemplo-tratado-1")
        assert f["title"] == "Tratado de Ejemplo"
        assert f["fields"] == {"titulo": "Tratado de Ejemplo", "autor": "Autora De Ejemplo, A.",
                               "edicion": "1", "paginas": 1240}
        assert f["stats"]["units"] == 3980 and f["status"] == "ok"
        assert f["status_detail"] is None

    def test_la_fuente_sin_book_aparece_igual_y_dice_que_le_falta(self, doble):
        """El caso que se paga en produccion: `cargar_libro` y `registrar_fuente` son dos llamadas,
        y quien hace solo la primera tendria una consola vacia sobre un grafo lleno."""
        f = admin_v1.obtener_fuente("ejemplo-manual-2")
        assert f["title"] == "ejemplo-manual-2", "sin :Book el titulo es el id, no un None"
        assert f["stats"]["units"] == 902 and f["stats"]["pages"] == 312
        assert f["status"] == "parcial"
        assert "202 of 902" in f["status_detail"]

    def test_semaforo(self):
        assert admin_v1.estado_fuente(100, 100) == "ok"
        assert admin_v1.estado_fuente(100, 90) == "parcial"
        # Una fuente registrada sin un solo chunk no es "vacia": es una carga que no ocurrio.
        assert admin_v1.estado_fuente(0, 0) == "fallido"

    def test_orden_por_defecto_las_mas_grandes_primero(self, doble):
        items = admin_v1.listar_fuentes()["items"]
        assert [f["id"] for f in items] == ["ejemplo-tratado-1", "ejemplo-manual-2"]

    def test_orden_invertido(self, doble):
        items = admin_v1.listar_fuentes(sort="stats.units")["items"]
        assert [f["id"] for f in items] == ["ejemplo-manual-2", "ejemplo-tratado-1"]

    def test_un_sort_no_declarado_es_422(self, doble):
        with pytest.raises(AdminError) as e:
            admin_v1.listar_fuentes(sort="b.password")
        assert e.value.status == 422 and e.value.code == "validation_error"

    def test_filtrar_por_un_campo_que_no_es_filtrable_es_422(self, doble):
        """Lo que no esta declarado filtrable no llega al Cypher ni a la comparacion: el
        descriptor es la lista blanca."""
        with pytest.raises(AdminError) as e:
            admin_v1.listar_fuentes(filtros={"edicion": "1"})
        assert e.value.status == 422 and "not filterable" in e.value.detail

    def test_filtrar_por_un_campo_declarado(self, doble):
        assert len(admin_v1.listar_fuentes(filtros={"autor": "Autora De Ejemplo, A."})["items"]) == 1
        assert admin_v1.listar_fuentes(filtros={"autor": "Nadie"})["items"] == []

    def test_pedir_las_que_tienen_el_campo_vacio(self, doble):
        """El centinela del contrato: sin el no habria forma de pedir "las fuentes sin autor",
        que es justo el grupo que alguien quiere completar."""
        items = admin_v1.listar_fuentes(filtros={"autor": admin_v1.SIN_VALOR})["items"]
        assert [f["id"] for f in items] == ["ejemplo-manual-2"]

    def test_busqueda_libre_por_id_y_titulo(self, doble):
        assert len(admin_v1.listar_fuentes(q="tratado")["items"]) == 1
        assert len(admin_v1.listar_fuentes(q="ejemplo")["items"]) == 2

    def test_kind_desconocido_es_422(self, doble):
        with pytest.raises(AdminError) as e:
            admin_v1.listar_fuentes(kind="norma")
        assert e.value.status == 422

    def test_paginacion_por_cursor(self, doble):
        p1 = admin_v1.listar_fuentes(limit=1)
        assert len(p1["items"]) == 1 and p1["total"] == 2 and p1["next_cursor"]
        p2 = admin_v1.listar_fuentes(limit=1, cursor=p1["next_cursor"])
        assert p2["next_cursor"] is None
        assert {p1["items"][0]["id"], p2["items"][0]["id"]} == {"ejemplo-tratado-1", "ejemplo-manual-2"}

    def test_un_cursor_inventado_es_422_y_no_un_500(self, doble):
        with pytest.raises(AdminError) as e:
            admin_v1.listar_fuentes(cursor="no-es-un-cursor")
        assert e.value.status == 422

    def test_fuente_inexistente(self, doble):
        with pytest.raises(AdminError) as e:
            admin_v1.obtener_fuente("no-existe")
        assert e.value.status == 404 and e.value.code == "not_found"


# ---------------------------------------------------------------- rutas
class TestRutas:
    def test_sin_credencial_todo_admin_v1_es_401(self, doble):
        """El contrato declara 401 en TODA ruta de admin/v1, /health incluido: la de la consola
        no es la sonda publica del servicio.

        Y 401 DE VERDAD, no 500: hasta el 14-sep-2026 el middleware LEVANTABA un HTTPException, y
        una excepcion levantada dentro de un middleware no pasa por los handlers de FastAPI —sube
        al ServerErrorMiddleware y sale como "500 Internal Server Error"—. El acceso quedaba
        denegado igual, pero un cliente no podia distinguir una credencial que falta de un
        servidor roto, y la consola no tenia como pedir la clave.
        """
        with TestClient(main.app, raise_server_exceptions=False) as anonimo:
            for ruta in ("/admin/v1/descriptor", "/admin/v1/health", "/admin/v1/stats",
                         "/admin/v1/sources", "/admin/v1/sources/ejemplo-tratado-1"):
                r = anonimo.get(ruta)
                assert r.status_code == 401, (ruta, r.status_code)
                assert r.json()["code"] == "unauthorized"
                assert r.headers.get("WWW-Authenticate", "").startswith("Bearer")

    def test_la_sonda_publica_del_servicio_sigue_sin_credencial(self, doble):
        """`/health` (la del servicio, no la del contrato) es lo que pega un balanceador: si
        pidiera clave, el balanceador marcaria el servicio como caido."""
        with TestClient(main.app, raise_server_exceptions=False) as anonimo:
            assert anonimo.get("/health").status_code == 200

    def test_descriptor(self, cliente):
        r = cliente.get("/admin/v1/descriptor")
        assert r.status_code == 200
        _validar("admin-descriptor", r.json())

    def test_health(self, cliente):
        r = cliente.get("/admin/v1/health")
        assert r.status_code == 200 and r.json()["graph"] == "connected"

    def test_stats(self, cliente):
        assert cliente.get("/admin/v1/stats").json()["units_total"] == 4882

    def test_pagina_de_fuentes(self, cliente):
        cuerpo = cliente.get("/admin/v1/sources").json()
        assert set(cuerpo) == {"items", "total", "next_cursor"}
        assert cuerpo["total"] == 2
        for item in cuerpo["items"]:
            _validar("admin-source", item)

    def test_filtro_por_query_string_del_contrato(self, cliente):
        """`filter[autor]=...`: los filtros por campo no se declaran en la firma porque dependen
        del descriptor, que depende del perfil."""
        r = cliente.get("/admin/v1/sources", params={"filter[autor]": "Autora De Ejemplo, A."})
        assert [f["id"] for f in r.json()["items"]] == ["ejemplo-tratado-1"]

    def test_una_fuente(self, cliente):
        r = cliente.get("/admin/v1/sources/ejemplo-manual-2")
        assert r.status_code == 200
        _validar("admin-source", r.json())

    def test_el_404_sale_con_detail_y_code(self, cliente):
        r = cliente.get("/admin/v1/sources/no-existe")
        assert r.status_code == 404
        assert r.json() == {"detail": "No such source: no-existe", "code": "not_found"}

    def test_un_filtro_invalido_sale_como_422_del_contrato(self, cliente):
        r = cliente.get("/admin/v1/sources", params={"filter[password]": "x"})
        assert r.status_code == 422 and r.json()["code"] == "validation_error"

    def test_el_grafo_caido_es_503_con_codigo(self, grafo_caido):
        with TestClient(main.app) as c:
            c.headers.update({"Authorization": f"Bearer {get_settings().api_key}"})
            for ruta in ("/admin/v1/stats", "/admin/v1/sources", "/admin/v1/sources/x"):
                r = c.get(ruta)
                assert r.status_code == 503 and r.json()["code"] == "graph_unavailable", ruta
            # ...pero el health NO: un health que se cae con el grafo no sirve para diagnosticar.
            salud = c.get("/admin/v1/health")
            assert salud.status_code == 200 and salud.json()["graph"] == "down"
