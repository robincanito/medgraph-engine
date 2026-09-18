"""El endpoint MCP: que lo que un cliente recibe sea EXACTAMENTE lo que dice el contrato `mcp/v1`.

QUE FIJA ESTE ARCHIVO, y por que cada cosa:

  1. La superficie VALIDA contra `schemas/mcp-tools.schema.json` de nomos-contracts (copia en
     `tests/contracts/`, con test de paridad que se saltea si el contrato no esta en disco). Y se
     lee POR DONDE SE LEE DE VERDAD: `tools/list` con el cliente oficial del SDK contra la app ASGI
     en memoria. El `inputSchema` y el `outputSchema` no estan escritos en `mcp_remote.py` --los
     GENERA el SDK desde las firmas y los modelos Pydantic--, asi que mirar el archivo no prueba
     nada.
  2. Las SIETE obligatorias estan y `evidence` no: es la capacidad opcional y este repo no trae
     ningun adaptador a una fuente externa.
  3. La puerta es la API key: sin credencial `/mcp` contesta 401 --y su cuerpo dice como entrar,
     porque acá no hay servidor de autorizacion al que mandar al cliente--.
  4. El descriptor de admin/v1 publica la MISMA lista de tools que el transporte, y la URL cuelga de
     `instance.api_base`.
  5. El ciclo de vida: el endpoint se puede arrancar dos veces en el mismo proceso. Es la limitacion
     del SDK que rompio diez tests de `test_admin_v1.py` el 17-sep-2026 (ver `_PuertaMCP`).

SIN NEO4J Y SIN RED: `vector.search_hybrid` y `services.graph.query` se doblan. Lo que se prueba acá
es la SUPERFICIE y el mapeo, no el retrieval --de eso se ocupan los tests del pipeline--.
"""
import json
import os
from pathlib import Path

import pytest

# Mismo cinturon que `tests/test_admin_v1.py`: quien clona el repo para usar SOLO el pipeline corre
# `pytest` sin las dependencias de la API, y este archivo no puede voltearle la suite. En CI
# (MEDGRAPH_ENGINE_REQUIRE_API=1) un skip ES una falla.
try:
    import anyio  # noqa: F401
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    import mcp  # noqa: F401
    import slowapi  # noqa: F401
except ModuleNotFoundError as e:  # pragma: no cover - depende del entorno, no del codigo
    if os.getenv("MEDGRAPH_ENGINE_REQUIRE_API") == "1":
        raise
    pytest.skip(f"falta {e.name}: la suite de la API necesita "
                "`pip install -r api/requirements.txt`", allow_module_level=True)

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

import main
from routers import mcp_remote
from services import admin_v1, graph, mcp_tools, vector
from services.settings import get_settings

RAIZ = Path(__file__).resolve().parent.parent
CONTRATOS = RAIZ / "tests" / "contracts"
NOMOS = RAIZ.parent.parent / "nomos" / "nomos-contracts"

CONTRATO = json.loads((CONTRATOS / "mcp-tools.schema.json").read_text(encoding="utf-8"))

#: Las obligatorias y el vocabulario SE LEEN DEL SCHEMA, no se retipean: los `contains` de `tools`
#: son la forma machine-readable de "esta tool tiene que estar". Si el contrato agrega una tool, este
#: archivo no se toca.
OBLIGATORIAS = [c["contains"]["properties"]["name"]["const"]
                for c in CONTRATO["properties"]["tools"]["allOf"]]
VOCABULARIO = CONTRATO["$defs"]["Tool"]["properties"]["name"]["enum"]
PROHIBIDOS = set(CONTRATO["$defs"]["EsquemaDeSalida"]["properties"]["properties"]
                 ["propertyNames"]["not"]["enum"])


# ---------------------------------------------------------------- dobles
#: Un chunk como lo devuelve el retrieval de este repo (`vector.RETORNO_CHUNK`).
CHUNK = {
    "id": "ejemplo-tratado-1_v2_00042",
    "libro": "ejemplo-tratado-1",
    "texto": "Texto de ejemplo del pasaje recuperado. " * 12,
    "pag_inicio": 120, "pag_fin": 121, "palabras": 280,
    "capitulo": "Capitulo 3", "seccion": "Seccion 3.2", "tipo": "body",
    "parent_id": "ejemplo-tratado-1_v2_00042_p", "calidad": "ok", "rrf_score": 0.032,
}


@pytest.fixture
def busqueda(monkeypatch):
    """`search_hybrid` sin grafo ni proveedor de embeddings. Anota como la llamaron."""
    llamadas = []

    def _hibrida(query, top_k=8, libro_id=None, **kw):
        llamadas.append({"query": query, "top_k": top_k, **kw})
        return {"keyword_count": 1, "semantic_count": 1, "results": [dict(CHUNK)],
                "procedencia": {"top_k": {CHUNK["libro"]: 1}, "pool": {CHUNK["libro"]: 9},
                                "garantizados": 0, "candidatos_unicos": 9},
                "degradado": None, "fallas": []}

    monkeypatch.setattr(vector, "search_hybrid", _hibrida)
    return llamadas


@pytest.fixture
def unidad(monkeypatch):
    """`fetch` sin grafo: responde la consulta de unidad por identidad."""
    def _query(cypher, params=None, retries=2):
        if cypher == mcp_tools.Q_UNIDAD:
            if (params or {}).get("id") != CHUNK["id"]:
                return []
            return [{**CHUNK, "indice": 42, "tiene_embedding": True,
                     "libro_titulo": "Tratado de Ejemplo", "autor": "Autora De Ejemplo, A.",
                     "edicion": "1"}]
        raise AssertionError(f"el doble no conoce esta consulta:\n{cypher}")

    monkeypatch.setattr(mcp_tools, "cypher", _query)
    monkeypatch.setattr(graph, "query", _query)
    return _query


# ---------------------------------------------------------------- el cliente del SDK
async def _sesion(headers, hacer):
    """Una sesion MCP completa contra la app ASGI EN MEMORIA, con el cliente oficial del SDK.

    `httpx.ASGITransport` es lo que hace que no haya socket ni puerto: el cliente habla con
    `main.app` --middleware de auth incluido-- como si fuera la red. Si el cliente oficial no puede
    completar `initialize`, no importa que nuestros asserts pasen.
    """
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), headers=headers,
                                 follow_redirects=True) as http:
        # `localhost:8000` y no un dominio inventado: la proteccion anti DNS-rebinding del SDK valida
        # el header `Host` contra los hosts declarados, y con otro contesta 421.
        async with streamable_http_client("http://localhost:8000/mcp", http_client=http) as (r, w, _):
            async with ClientSession(r, w) as sesion:
                inicio = await sesion.initialize()
                return await hacer(sesion, inicio)


@pytest.fixture
def sesion_mcp():
    """`correr(headers, hacer)`: arranca el endpoint, abre una sesion y ejecuta `hacer`.

    El lifespan se entra POR TEST --y no una vez por corrida como en la instancia privada-- porque
    acá se puede: `_PuertaMCP` crea un session manager nuevo en cada arranque. El portal de anyio
    mantiene un event loop propio en un hilo, como hace TestClient por dentro.
    """
    import anyio

    def correr(headers, hacer):
        with anyio.from_thread.start_blocking_portal("asyncio") as portal:
            with portal.wrap_async_context_manager(mcp_remote.mcp_lifespan()):
                return portal.call(_sesion, headers, hacer)

    return correr


@pytest.fixture
def cabeceras():
    return {"X-API-Key": get_settings().api_key}


@pytest.fixture
def superficie(sesion_mcp, cabeceras):
    """La superficie REAL, leida con el cliente del SDK y normalizada al documento del contrato."""
    async def hacer(sesion, inicio):
        listado = await sesion.list_tools()
        return inicio, listado.tools

    inicio, tools = sesion_mcp(cabeceras, hacer)
    return {
        "contract_version": "mcp/v1",
        "instructions": inicio.instructions,
        # NORMALIZAR ES ELEGIR LAS CINCO CLAVES DEL CONTRATO, no recortar lo que molesta: los schemas
        # y las anotaciones viajan TAL CUAL los publico el servidor. `exclude_none` en las
        # anotaciones porque el SDK serializa los hints que nadie fijo y un null no es una anotacion.
        "tools": [{"name": t.name, "description": t.description,
                   "annotations": t.annotations.model_dump(exclude_none=True) if t.annotations
                   else None,
                   "inputSchema": t.inputSchema, "outputSchema": t.outputSchema}
                  for t in tools],
    }


def _tool(superficie, nombre):
    return next(t for t in superficie["tools"] if t["name"] == nombre)


def _identificadores_internos(nodo, camino=""):
    """Donde aparece un nombre prohibido, a cualquier profundidad (items incluidos)."""
    encontrados = []
    if isinstance(nodo, dict):
        for clave, valor in nodo.items():
            aqui = f"{camino}/{clave}"
            if clave in PROHIBIDOS:
                encontrados.append(aqui)
            encontrados += _identificadores_internos(valor, aqui)
    elif isinstance(nodo, list):
        for i, item in enumerate(nodo):
            encontrados += _identificadores_internos(item, f"{camino}/{i}")
    return encontrados


class TestLaSuperficieCumpleElContrato:
    def test_el_documento_entero_valida(self, superficie):
        errores = sorted(Draft202012Validator(CONTRATO).iter_errors(superficie),
                         key=lambda e: list(e.path))
        assert not errores, [f"{'/'.join(map(str, e.path))}: {e.message}" for e in errores][:5]

    @pytest.mark.parametrize("nombre", ["search", "fetch", "list_sources", "search_entities",
                                        "expand_concept", "deep_dive", "unified_query"])
    def test_cada_tool_valida_contra_el_contrato(self, superficie, nombre):
        """Uno por tool: el documento entero ya valida, pero un fallo ahi no dice CUAL se desvio."""
        validador = Draft202012Validator({"$ref": "#/$defs/Tool", "$defs": CONTRATO["$defs"]})
        errores = sorted(validador.iter_errors(_tool(superficie, nombre)),
                         key=lambda e: list(e.path))
        assert not errores, [f"{'/'.join(map(str, e.path))}: {e.message}" for e in errores][:5]

    def test_estan_las_siete_obligatorias_y_ninguna_de_mas(self, superficie):
        nombres = [t["name"] for t in superficie["tools"]]
        assert [n for n in OBLIGATORIAS if n not in nombres] == []
        assert set(nombres) <= set(VOCABULARIO)
        assert len(nombres) == len(set(nombres))

    def test_evidence_no_se_publica(self, superficie):
        """Es la capacidad OPCIONAL y exige una fuente EXTERNA: este repo no trae adaptador. Siete
        tools cumplen el contrato entero; una `evidence` que no consulta nada seria la vidriera que
        miente."""
        assert "evidence" not in [t["name"] for t in superficie["tools"]]
        assert len(superficie["tools"]) == 7

    def test_ninguna_tool_lleva_prefijo_de_instancia(self, superficie):
        """`medgraph_search` era el defecto del `mcp_server.py` que se retiro: el prefijo obliga a
        reescribir el agente al cambiar de grafo, y el cliente ya sabe a que servidor le habla."""
        assert [t["name"] for t in superficie["tools"] if t["name"].startswith("medgraph")] == []

    def test_todas_de_solo_lectura_y_ninguna_de_mundo_abierto(self, superficie):
        for t in superficie["tools"]:
            assert t["annotations"]["readOnlyHint"] is True, t["name"]
            assert t["annotations"]["destructiveHint"] is False, t["name"]
            assert t["annotations"]["openWorldHint"] is False, t["name"]

    def test_la_degradacion_y_la_procedencia_viajan_donde_el_contrato_las_exige(self, superficie):
        for nombre in ("search", "deep_dive", "unified_query"):
            claves = _tool(superficie, nombre)["outputSchema"]["properties"]
            assert "degraded" in claves and "notice" in claves, nombre
            assert "procedencia" in claves, nombre

    def test_ningun_identificador_interno_viaja_a_ninguna_profundidad(self, superficie):
        """El contrato cierra los nombres del PRIMER nivel de cada salida; los items viven en
        `$defs`, y un `trace` o un `ms_total` ahi adentro llegaria igual al cliente."""
        for t in superficie["tools"]:
            for nivel in ("inputSchema", "outputSchema"):
                donde = _identificadores_internos(t[nivel])
                assert donde == [], f"{t['name']}.{nivel}: {donde}"

    def test_cada_tool_declara_outputSchema_y_una_descripcion_de_verdad(self, superficie):
        minimo = CONTRATO["$defs"]["Tool"]["properties"]["description"]["minLength"]
        for t in superficie["tools"]:
            assert t["outputSchema"]["type"] == "object", t["name"]
            assert len(t["description"]) >= minimo, t["name"]

    def test_las_descripciones_no_nombran_el_corpus_de_nadie(self, superficie):
        """Este repo es PUBLICO y su corpus lo pone quien lo clona: una descripcion que nombre una
        obra, una editorial o un `libro_id` real seria la procedencia del corpus de otro viajando en
        cada release (item C-1 de la auditoria de exposicion, 13-sep-2026)."""
        texto = " ".join(t["description"] for t in superficie["tools"])
        for aguja in ("far" + "reras", "harri" + "son", "El" + "sevier", "book" + "smedicos", "pubmed"):
            assert aguja.lower() not in texto.lower(), aguja

    def test_el_servidor_publica_instrucciones_autosuficientes(self, superficie):
        """Los primeros 512 caracteres son lo unico que varios clientes muestran, y son los que
        tienen que decir que lo recuperado es evidencia y que hay que citar."""
        instrucciones = superficie["instructions"] or ""
        assert len(instrucciones) >= 200
        assert "cite" in instrucciones[:512].lower()

    def test_la_telemetria_no_cambia_la_firma_que_ve_el_cliente(self, superficie):
        """`functools.wraps` + `inspect.signature` siguiendo `__wrapped__`: si la firma cambiara, el
        cliente perderia argumentos que el contrato CIERRA."""
        assert list(_tool(superficie, "search")["inputSchema"]["properties"]) == [
            "query", "limit", "filtros", "garantizar", "agrupar"]
        assert list(_tool(superficie, "fetch")["inputSchema"]["properties"]) == ["id"]


class TestLaPuertaEsLaClave:
    def test_sin_credencial_es_401_y_el_cuerpo_dice_como_entrar(self):
        """No hay `resource_metadata` porque no hay authorization server: anunciar un flujo de OAuth
        que no existe mandaria al cliente a un descubrimiento que termina en 404. El cuerpo explica
        la credencial, que es lo unico honesto que se puede decir."""
        with TestClient(main.app, raise_server_exceptions=False) as anonimo:
            r = anonimo.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})

        assert r.status_code == 401
        assert r.json()["code"] == "unauthorized"
        detalle = r.json()["detail"]
        assert "X-API-Key" in detalle and "Bearer" in detalle
        assert "OAuth" in detalle, "tiene que decir que NO hay servidor de autorizacion"
        assert "resource_metadata" not in r.headers.get("WWW-Authenticate", "")

    def test_el_cliente_del_sdk_no_puede_completar_el_handshake_sin_clave(self, sesion_mcp):
        """La mitad negativa: sin credencial NO hay initialize.

        `repr` y no `str`: el cliente del SDK envuelve el fallo en un `ExceptionGroup` de anyio, cuyo
        `str` es "unhandled errors in a TaskGroup" y se traga el codigo de estado.
        """
        async def hacer(sesion, inicio):  # pragma: no cover - no se llega
            return inicio

        with pytest.raises(Exception) as e:
            sesion_mcp({}, hacer)
        assert "401" in repr(e.value), repr(e.value)

    def test_con_la_clave_por_bearer_tambien_entra(self, sesion_mcp):
        async def hacer(sesion, inicio):
            return (await sesion.list_tools()).tools

        tools = sesion_mcp({"Authorization": f"Bearer {get_settings().api_key}"}, hacer)
        assert len(tools) == 7


class TestLasToolsContestanPorElTransporte:
    def test_search_devuelve_la_forma_que_el_contrato_pide(self, sesion_mcp, cabeceras, busqueda):
        async def hacer(sesion, _inicio):
            return await sesion.call_tool("search", {"query": "un tema", "limit": 3})

        r = sesion_mcp(cabeceras, hacer)

        assert not r.isError, r.content[0].text if r.content else r
        item = r.structuredContent["results"][0]
        assert item["id"] == CHUNK["id"] and item["source_id"] == CHUNK["libro"]
        assert item["locator"] == "pp. 120-121" and item["calidad"] == "ok"
        # La URL apunta a la FUENTE en admin/v1: este repo no tiene ruta de lectura por unidad, y un
        # link inventado daria 404 en el primer click.
        assert item["url"].endswith(f"/admin/v1/sources/{CHUNK['libro']}")
        assert r.structuredContent["degraded"] is False
        assert r.structuredContent["procedencia"]["pool"] == {CHUNK["libro"]: 9}
        assert busqueda[0]["top_k"] == 3

    def test_search_pasa_filtros_garantizar_y_agrupar_al_retrieval(self, sesion_mcp, cabeceras,
                                                                   busqueda):
        """Si se perdieran en el camino, la respuesta seria del corpus entero y el modelo no tendria
        como notarlo."""
        async def hacer(sesion, _inicio):
            return await sesion.call_tool("search", {
                "query": "un tema", "limit": 3,
                "filtros": {"libro_ids": ["ejemplo-tratado-1"]},
                "garantizar": {"libro_ids": ["ejemplo-tratado-1"], "cupo": 2},
                "agrupar": "padre"})

        r = sesion_mcp(cabeceras, hacer)

        assert not r.isError
        assert busqueda[0]["filtros"] == {"libro_ids": ["ejemplo-tratado-1"]}
        assert busqueda[0]["garantizar"] == {"libro_ids": ["ejemplo-tratado-1"], "cupo": 2}
        assert busqueda[0]["agrupar"] == "padre"

    def test_fetch_abre_la_unidad_con_su_procedencia(self, sesion_mcp, cabeceras, unidad):
        async def hacer(sesion, _inicio):
            return await sesion.call_tool("fetch", {"id": CHUNK["id"]})

        r = sesion_mcp(cabeceras, hacer)

        assert not r.isError, r.content[0].text if r.content else r
        u = r.structuredContent
        assert u["id"] == CHUNK["id"] and u["text"].startswith("Texto de ejemplo")
        meta = u["metadata"]
        assert meta["source_id"] == CHUNK["libro"] and meta["source_title"] == "Tratado de Ejemplo"
        assert meta["page_start"] == 120 and meta["locator"] == "pp. 120-121"
        assert meta["embedded"] is True and meta["truncated"] is False

    def test_un_id_que_no_existe_es_un_error_de_tool_y_no_un_documento_vacio(self, sesion_mcp,
                                                                            cabeceras, unidad):
        """El contrato lo exige (§3.2): un texto vacio se leeria como "esa fuente no dice nada"."""
        async def hacer(sesion, _inicio):
            return await sesion.call_tool("fetch", {"id": "no-existe_00000"})

        r = sesion_mcp(cabeceras, hacer)

        assert r.isError
        assert "no unit with id" in r.content[0].text.lower()


class TestDegradacion:
    """La regla 4 del contrato: una lista vacia por CAIDA no se puede leer igual que una ausencia."""

    def test_si_el_retrieval_se_cae_la_lista_vacia_viene_con_aviso(self, monkeypatch):
        def _revienta(*a, **kw):
            raise RuntimeError("el grafo no contesta")
        monkeypatch.setattr(vector, "search_hybrid", _revienta)

        r = mcp_tools.search("un tema")

        assert r["results"] == []
        assert r["degraded"] is True
        assert "could NOT be queried" in r["notice"] or "could not be queried" in r["notice"]

    def test_una_busqueda_sin_material_NO_degrada(self, monkeypatch):
        """La otra mitad: el corpus contesto y no hay nada. Marcarlo como degradado haria que el
        modelo desconfie de una respuesta correcta."""
        monkeypatch.setattr(vector, "search_hybrid",
                            lambda *a, **kw: {"keyword_count": 0, "semantic_count": 0,
                                              "results": [], "procedencia": {},
                                              "degradado": None, "fallas": []})

        r = mcp_tools.search("un tema")

        assert r["results"] == [] and r["degraded"] is False and r["notice"] is None

    def test_una_ruta_caida_es_una_degradacion_PARCIAL(self, monkeypatch):
        monkeypatch.setattr(vector, "search_hybrid",
                            lambda *a, **kw: {"keyword_count": 0, "semantic_count": 1,
                                              "results": [dict(CHUNK)], "procedencia": {},
                                              "degradado": "parcial",
                                              "fallas": [{"ruta": "keyword", "error": "X"}]})

        r = mcp_tools.search("un tema")

        assert r["results"] and r["degraded"] is True and "partial" in r["notice"]

    def test_unified_query_sin_analizador_busca_igual_y_lo_dice(self, monkeypatch, busqueda):
        """El analizador necesita un proveedor y puede no estar configurado. La tool no puede
        devolver un error ni una lista vacia muda: busca la pregunta tal cual y lo avisa."""
        from services import analyzer

        def _sin_proveedor(_pregunta):
            raise RuntimeError("no hay proveedor configurado")
        monkeypatch.setattr(analyzer, "analyze_query", _sin_proveedor)

        r = mcp_tools.unified_query("una pregunta", top_k=3)

        assert r["bibliography"]["count"] == 1
        assert r["sub_queries"] == ["una pregunta"]
        assert "the query router did not answer" in r["notice"]


class TestElDescriptorDiceLoMismoQueElTransporte:
    """Dos puertas, una verdad. El descriptor es lo que lee la consola; el transporte, lo que recibe
    un cliente. Si divergen, la consola publica una URL o una lista de tools que no existe."""

    def test_publica_la_capacidad_y_el_bloque_juntos(self):
        d = admin_v1.descriptor()
        assert d["capabilities"]["mcp"] is True
        assert "mcp" in d["auth"]
        assert "mcp" not in d["capability_notes"], "una capacidad que funciona no lleva nota"

    def test_publica_la_version_del_contrato(self):
        assert admin_v1.descriptor()["auth"]["mcp"]["contract"] == \
            CONTRATO["properties"]["contract_version"]["const"]

    def test_la_lista_de_tools_es_LA_DEL_TRANSPORTE(self, superficie):
        assert admin_v1.descriptor()["auth"]["mcp"]["tools"] == \
            [t["name"] for t in superficie["tools"]]

    def test_la_url_cuelga_de_la_base_publica(self):
        d = admin_v1.descriptor()
        assert d["auth"]["mcp"]["url"] == f"{d['instance']['api_base']}/mcp"

    def test_no_publica_oauth_porque_no_hay_authorization_server(self):
        """`oauth` describe el estado de un AS (DCR/CIMD leidos de su metadata publica) y acá no hay
        ninguno: la puerta es una clave. El contrato lee la AUSENCIA del bloque como "esta instancia
        autentica su MCP de otra manera"; publicarlo en `null` NO es una opcion, porque el schema lo
        declara `type: object` y el descriptor entero dejaria de validar."""
        assert "oauth" not in admin_v1.descriptor()["auth"]["mcp"]

    def test_el_descriptor_sigue_validando_contra_el_contrato(self):
        schema = json.loads((CONTRATOS / "admin-descriptor.schema.json").read_text(encoding="utf-8"))
        errores = sorted(Draft202012Validator(schema).iter_errors(admin_v1.descriptor()),
                         key=lambda e: list(e.path))
        assert not errores, [f"{'/'.join(map(str, e.path))}: {e.message}" for e in errores][:3]


class TestElCicloDeVida:
    """El endpoint tiene que poder arrancar DOS veces en el mismo proceso.

    `StreamableHTTPSessionManager.run()` corre una sola vez por instancia, y este repo entra y sale
    del lifespan varias veces en su suite (`with TestClient(main.app)`). `_PuertaMCP` crea un manager
    nuevo por arranque; si eso dejara de funcionar --porque el SDK cambio el atributo que se
    resetea-- el sintoma seria diez tests de admin/v1 en rojo hablando de otra cosa. Se fija acá.
    """

    def test_dos_arranques_seguidos(self, cabeceras):
        async def hacer(sesion, _inicio):
            return (await sesion.list_tools()).tools

        import anyio

        for _ in range(2):
            with anyio.from_thread.start_blocking_portal("asyncio") as portal:
                with portal.wrap_async_context_manager(mcp_remote.mcp_lifespan()):
                    assert len(portal.call(_sesion, cabeceras, hacer)) == 7

    def test_sin_arrancar_el_endpoint_contesta_503_y_no_un_error_del_sdk(self, cabeceras):
        """Si alguien monta la app sin encadenar el lifespan, el SDK tira un AssertionError con el
        texto "Task group is not initialized". Un 503 con el motivo es el mismo hecho, legible."""
        with TestClient(main.app, raise_server_exceptions=False) as cliente:
            pass  # el lifespan ya salio: la puerta quedo sin app

        r = cliente.post("/mcp", headers=cabeceras,
                         json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert r.status_code == 503
        assert r.json()["code"] == "mcp_not_started"


class TestParidadDeContrato:
    def test_el_schema_vendorizado_es_identico_al_de_nomos_contracts(self):
        """La copia vale offline; el original manda. Si el contrato cambia y esto no, se pone rojo —
        que es exactamente para lo que existe la copia."""
        if not NOMOS.exists():
            pytest.skip("nomos-contracts no esta en este disco")
        assert (CONTRATOS / "mcp-tools.schema.json").read_text(encoding="utf-8") == \
            (NOMOS / "schemas" / "mcp-tools.schema.json").read_text(encoding="utf-8")


class TestElStdioSeRetiro:
    """`mcp_server.py` se borro el 17-sep-2026 y no puede volver por la ventana.

    Era un servidor stdio con once tools `medgraph_*` --tres contra rutas que este espejo ya no
    expone-- que le hablaba por HTTP a una API corriendo aparte. Dos copias de la superficie de tools
    es exactamente lo que el contrato `mcp/v1` existe para impedir.
    """

    def test_el_archivo_no_esta(self):
        assert not (RAIZ / "mcp_server.py").exists()

    def test_ningun_documento_manda_a_correrlo(self):
        """Un README que manda a correr un archivo que no existe es peor que no documentarlo.

        Lo que se prohibe es la INSTRUCCION, no la mencion: la nota que explica que se retiro tiene
        que poder nombrarlo --si no, nadie que vuelva a buscarlo entiende a donde se fue--.
        """
        for nombre in ("README.md", "SECURITY.md", "ruff.toml", "CONTRIBUTING.md", ".env.example"):
            texto = (RAIZ / nombre).read_text(encoding="utf-8")
            assert "python mcp_server.py" not in texto, nombre
            assert "MEDGRAPH_API_URL" not in texto, f"{nombre}: variable del stdio retirado"

    def test_el_readme_lo_declara_retirado_y_apunta_al_endpoint(self):
        readme = (RAIZ / "README.md").read_text(encoding="utf-8")
        fila = next(ln for ln in readme.splitlines()
                    if ln.startswith("| `mcp_server.py`"))
        assert "gone" in fila.lower()
        assert "### The MCP endpoint" in readme
