"""pipeline/embeddings.py — el texto que se embebe, la llamada al proveedor y la POLITICA.

`pipeline/` es un espejo byte a byte del pipeline privado; estos tests son la regresion de las
curas que se replicaron el 13-sep-2026, adaptadas a los imports del engine. Ninguna llamada
real: el cliente es un doble, el grafo son dos funciones y `dormir` esta inyectado, asi que la
suite no espera ni un segundo ni gasta un centavo.

Incidentes que fijan estos tests:
  - jun-2026: `contents=[str, str]` devolvia UN vector mezcla de todos y `zip()` truncaba en
    silencio. Se certifica el formato dict por texto y el largo de la salida.
  - 18-ago-2026: el proveedor acepta UN contenido por llamada (la migracion a Vertex lo
    impuso), o sea que el "lote" es del lado nuestro y cada texto se PAGA aparte.
  - 13-sep-2026 (C1/D10): `llamadas` contaba lotes y subestimaba el gasto hasta EMBED_BATCH
    veces. Ahora cuenta llamadas PAGAS, reintentos y la que revento incluidas.
  - 13-sep-2026 (decision C): un lote perdido por una caida quedaba perdido PARA SIEMPRE y el
    documento cerraba como exito. `vectorizar_libro` agrega la SEGUNDA PASADA de los faltantes.
"""
import ast
from pathlib import Path

import pytest

from pipeline import embeddings

RAIZ = Path(__file__).resolve().parent.parent


class _Emb:
    def __init__(self, n):
        self.values = [0.001 * n] * 8


class _Resp:
    def __init__(self, n):
        self.embeddings = [_Emb(n)]


class _Models:
    def __init__(self, fallar_en=()):
        self.llamadas = []
        # numeros de llamada (1-based) que revientan, para inyectar 429 / 5xx
        self.fallar_en = set(fallar_en)

    def embed_content(self, model, contents):
        self.llamadas.append((model, contents))
        assert len(contents) == 1, "el proveedor acepta UN contenido por llamada"
        if len(self.llamadas) in self.fallar_en:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return _Resp(len(self.llamadas))


class _Cliente:
    def __init__(self, fallar_en=()):
        self.models = _Models(fallar_en)


class _Grafo:
    """query/write dobles: registra las escrituras y sirve los chunks sin embedding."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.consultas = []
        self.escrituras = []

    def query(self, cypher, params=None):
        self.consultas.append((cypher, params))
        return list(self.chunks)

    def write(self, cypher, params=None):
        self.escrituras.append((cypher, params))

    @property
    def ids_guardados(self):
        return [u["id"] for _, p in self.escrituras for u in p["updates"]]


class _Reloj:
    """`dormir` inyectado: los tests no esperan 15 segundos de verdad."""

    def __init__(self):
        self.esperas = []

    def __call__(self, segundos):
        self.esperas.append(segundos)


def _chunks(n, texto="cuerpo"):
    return [{"id": f"lib_v2_{i:05d}", "text": texto, "titulo_capitulo": None,
             "titulo_seccion": None, "tipo_contenido": "body"} for i in range(n)]


class TestBuildEmbeddingText:
    def test_prefijo_con_tildes_y_tipo(self):
        chunk = {"titulo_capitulo": "Nefrología", "titulo_seccion": "Inmunodepresores",
                 "tipo_contenido": "tabla", "text": "cuerpo"}
        assert embeddings.build_embedding_text(chunk) == \
            "Capítulo: Nefrología. Sección: Inmunodepresores. Tipo: tabla. cuerpo"

    def test_body_no_lleva_tipo(self):
        chunk = {"titulo_capitulo": "A", "titulo_seccion": "B", "tipo_contenido": "body",
                 "text": "x"}
        assert embeddings.build_embedding_text(chunk) == "Capítulo: A. Sección: B. x"

    def test_sin_metadata_devuelve_el_texto_pelado(self):
        assert embeddings.build_embedding_text({"text": "solo texto"}) == "solo texto"

    def test_el_prefijo_sin_tilde_no_es_el_canonico(self):
        """La copia vieja de la API escribia 'Capitulo:' sin tilde: distinto vector."""
        chunk = {"titulo_capitulo": "A", "text": "x"}
        assert "Capítulo:" in embeddings.build_embedding_text(chunk)
        assert "Capitulo:" not in embeddings.build_embedding_text(chunk)


class TestGenerateEmbeddings:
    def test_un_vector_por_texto_en_orden(self):
        c = _Cliente()
        out = embeddings.generate_embeddings(c, ["a", "b", "c"])
        assert len(out) == 3
        assert len(c.models.llamadas) == 3, "una llamada por texto"

    def test_formato_dict_por_contenido_no_string(self):
        c = _Cliente()
        embeddings.generate_embeddings(c, ["hola"])
        model, contents = c.models.llamadas[0]
        assert model == embeddings.EMBEDDING_MODEL == "gemini-embedding-2"
        assert contents == [{"parts": [{"text": "hola"}]}]

    def test_trunca_a_max_chars(self):
        c = _Cliente()
        embeddings.generate_embeddings(c, ["x" * 5000])
        enviado = c.models.llamadas[0][1][0]["parts"][0]["text"]
        assert len(enviado) == embeddings.MAX_CHARS == 2000

    def test_lista_vacia(self):
        assert embeddings.generate_embeddings(_Cliente(), []) == []

    def test_el_contador_cuenta_antes_de_llamar(self):
        """Si la llamada revienta igual se pago: el contador la suma ANTES."""
        contador = {}
        with pytest.raises(RuntimeError):
            embeddings.generate_embeddings(_Cliente(fallar_en={1}), ["a"], contador=contador)
        assert contador == {"llamadas": 1}

    def test_dims_declaradas(self):
        assert embeddings.EMBEDDING_DIMS == 3072


class TestVectorizarFaltantes:
    """LA politica de vectorizacion, UNA para todos los caminos.

    Antes vivia dos veces (el CLI con 3 reintentos y truncado, la API sin reintento y sin
    truncar): el MISMO documento terminaba con distinto numero de embeddings segun por donde
    entro, y los chunks que quedaban sin vector no se contaban en ninguna metrica.
    """

    def _correr(self, chunks, cliente=None, reloj=None, **kw):
        grafo, reloj = _Grafo(chunks), reloj or _Reloj()
        cliente = cliente or _Cliente()
        stats = embeddings.vectorizar_faltantes(grafo.query, grafo.write, cliente, "lib",
                                                dormir=reloj, **kw)
        return stats, grafo, cliente, reloj

    def test_camino_feliz_dos_lotes_veinte_llamadas_pagas(self):
        stats, grafo, cliente, _ = self._correr(_chunks(20), lote=10)
        # `llamadas` cuenta lo que el proveedor COBRA: un texto por llamada, no un lote (C1/D10).
        assert stats == {"embebidos": 20, "sin_vector": 0, "lotes_fallidos": 0, "llamadas": 20}
        assert len(grafo.escrituras) == 2, "una escritura por lote"
        assert grafo.ids_guardados == [c["id"] for c in _chunks(20)]
        assert len(cliente.models.llamadas) == 20, "el proveedor recibe un texto por llamada"

    def test_usa_la_consulta_canonica_y_el_cypher_canonico(self):
        _, grafo, _, _ = self._correr(_chunks(3))
        assert grafo.consultas == [(embeddings.CYPHER_CHUNKS_SIN_EMBEDDING, {"lid": "lib"})]
        assert grafo.escrituras[0][0] == embeddings.CYPHER_GUARDAR_EMBEDDINGS

    def test_sin_chunks_no_paga_ni_escribe(self):
        stats, grafo, cliente, reloj = self._correr([])
        assert stats == {"embebidos": 0, "sin_vector": 0, "lotes_fallidos": 0, "llamadas": 0}
        assert grafo.escrituras == [] and cliente.models.llamadas == [] and reloj.esperas == []

    def test_429_en_el_segundo_lote_reintenta_y_sale_bien(self):
        """El 429 pega en la llamada 11 (primer texto del lote 2): se reintenta el LOTE."""
        stats, grafo, _, reloj = self._correr(_chunks(20), cliente=_Cliente(fallar_en={11}),
                                              lote=10)
        assert stats["embebidos"] == 20 and stats["sin_vector"] == 0
        assert stats["lotes_fallidos"] == 0
        assert stats["llamadas"] == 21, "10 del lote 1 + la que revento + 10 del reintento"
        assert 5 in reloj.esperas, "la primera espera son 5 segundos"
        assert grafo.ids_guardados == [c["id"] for c in _chunks(20)]

    def test_la_espera_crece_y_la_ultima_se_repite(self):
        stats, _, _, reloj = self._correr(_chunks(5), cliente=_Cliente(fallar_en={1, 2, 3}),
                                          lote=5, reintentos=4)
        assert stats["embebidos"] == 5
        # 3 intentos fallidos -> esperas 5, 10 y 10 (la ultima de `esperas` se repite)
        assert [e for e in reloj.esperas if e in (5, 10)] == [5, 10, 10]

    def test_un_lote_que_falla_siempre_no_se_lleva_a_los_demas(self):
        """El lote falla, los otros siguen, y el faltante SE CUENTA."""
        # el lote 2 arranca en la llamada 11; la 11 revienta al primer texto, el intento 2
        # (12-21) revienta en la 21 y el intento 3 (22-31) en la 31: 21 llamadas pagas por 0
        # vectores.
        cliente = _Cliente(fallar_en={11, 21, 31})
        stats, grafo, _, _ = self._correr(_chunks(30), cliente=cliente, lote=10)
        assert stats["lotes_fallidos"] == 1
        assert stats["sin_vector"] == 10, "el lote perdido son 10 chunks"
        assert stats["embebidos"] == 20, "los otros dos lotes se guardaron"
        assert stats["llamadas"] == 41, "10 + 21 pagas del lote perdido + 10"
        guardados = grafo.ids_guardados
        assert len(guardados) == 20
        assert "lib_v2_00010" not in guardados and "lib_v2_00019" not in guardados

    def test_trunca_el_texto_a_max_chars(self):
        _, _, cliente, _ = self._correr(_chunks(1, texto="x" * 9000), max_chars=2000)
        enviado = cliente.models.llamadas[0][1][0]["parts"][0]["text"]
        assert len(enviado) == 2000, "el CLI truncaba y la API no: ahora truncan los dos"

    def test_el_default_de_max_chars_es_el_del_modulo(self):
        assert embeddings.MAX_CHARS_EMBED == embeddings.MAX_CHARS == 2000

    def test_desalineacion_no_guarda_nada_del_lote(self, monkeypatch):
        """El bug de junio: `contents=[str, str]` devolvia un vector mezcla y zip() truncaba."""
        real = embeddings.generate_embeddings
        monkeypatch.setattr(embeddings, "generate_embeddings",
                            lambda c, textos, **kw: real(c, textos, **kw)[:-1])
        grafo, reloj = _Grafo(_chunks(10)), _Reloj()
        stats = embeddings.vectorizar_faltantes(grafo.query, grafo.write, _Cliente(), "lib",
                                                lote=10, dormir=reloj)
        assert grafo.escrituras == [], "no se guarda NADA del lote desalineado"
        assert stats == {"embebidos": 0, "sin_vector": 10, "lotes_fallidos": 1, "llamadas": 30}

    def test_el_lote_por_defecto_es_el_del_modulo(self):
        stats, grafo, _, _ = self._correr(_chunks(25))
        assert embeddings.EMBED_BATCH == 10
        assert len(grafo.escrituras) == 3 and stats["llamadas"] == 25

    def test_on_progress_como_siempre(self):
        avances = []
        self._correr(_chunks(20), lote=10,
                     on_progress=lambda step, pct, msg: avances.append((step, pct, msg)))
        assert avances == [("vectorize", 50, "10/20 embeddings"),
                           ("vectorize", 100, "20/20 embeddings")]

    def test_pausa_entre_lotes_no_despues_del_ultimo(self):
        _, _, _, reloj = self._correr(_chunks(30), lote=10, pausa=0.3)
        assert reloj.esperas == [0.3, 0.3], "2 pausas para 3 lotes"

    def test_emite_embed_lote_por_intento(self, caplog):
        """El rastro que un oraculo de QA lee desde AFUERA (pipeline/eventos.py)."""
        cliente = _Cliente(fallar_en={11, 21, 31})
        with caplog.at_level("INFO"):
            self._correr(_chunks(20), cliente=cliente, lote=10)
        eventos = [r for r in caplog.records if getattr(r, "evento", None) == "embed_lote"]
        assert [(r.campos["intento"], r.campos["estado"], r.campos["n_textos"]) for r in eventos] \
            == [(1, "ok", 10), (1, "reintento", 10), (2, "reintento", 10), (3, "fallo", 10)]
        detalles = [r.campos.get("detalle") for r in eventos]
        assert detalles[0] is None, "el ok no lleva detalle"
        assert "429" in detalles[1] and len(detalles[1]) <= 200
        assert all(r.campos["libro_id"] == "lib" and isinstance(r.campos["ms"], int)
                   for r in eventos)


class _GrafoQueSeVacia(_Grafo):
    """`query` devuelve lo que TODAVIA no se guardo: es como se comporta el grafo de verdad.

    `_Grafo` devuelve siempre los mismos chunks (le alcanza para una pasada). La segunda
    pasada de `vectorizar_libro` vuelve a PREGUNTAR que falta, asi que con el doble plano
    veria otra vez el libro entero y el test no probaria nada: probaria que se puede pagar
    dos veces.
    """

    def query(self, cypher, params=None):
        self.consultas.append((cypher, params))
        guardados = set(self.ids_guardados)
        return [c for c in self.chunks if c["id"] not in guardados]


class TestVectorizarLibro:
    """LA SEGUNDA PASADA (decision C, 13-sep-2026): el paso no cierra con faltantes sin
    reintentar. Un lote perdido por un 5xx quedaba perdido para siempre —nadie volvia a pedir
    esos chunks— y el documento igual terminaba como exito."""

    def _correr(self, chunks, cliente=None, **kw):
        grafo = _GrafoQueSeVacia(chunks)
        cliente = cliente or _Cliente()
        stats = embeddings.vectorizar_libro(grafo.query, grafo.write, cliente, "lib",
                                            dormir=_Reloj(), **kw)
        return stats, grafo, cliente

    def test_un_libro_completo_hace_una_sola_pasada(self):
        """D10: la segunda pasada NO ocurre si no falta nada. Cero llamadas de mas — el
        `sin_vector == 0` de la primera corta el bucle sin volver ni al grafo."""
        stats, grafo, cliente = self._correr(_chunks(20), lote=10)
        assert stats == {"embebidos": 20, "sin_vector": 0, "lotes_fallidos": 0,
                         "llamadas": 20, "pasadas": 1}
        assert len(cliente.models.llamadas) == 20
        assert len(grafo.consultas) == 1, "una sola consulta de faltantes"

    def test_el_lote_perdido_se_recupera_en_la_segunda_pasada(self):
        """La caida duro los tres intentos del lote 2 y se fue: el libro termina ENTERO."""
        # El lote 2 arranca en la 11 y sus tres intentos arrancan en 11, 21 y 31; la 41 no
        # existe, asi que la segunda pasada (que ve solo esos 10 chunks) sale bien.
        cliente = _Cliente(fallar_en={11, 21, 31})
        stats, grafo, _ = self._correr(_chunks(30), cliente=cliente, lote=10)
        assert stats["sin_vector"] == 0 and stats["embebidos"] == 30
        assert stats["pasadas"] == 2 and stats["lotes_fallidos"] == 1
        assert stats["llamadas"] == 51, "41 de la primera pasada + 10 de la segunda"
        assert sorted(grafo.ids_guardados) == [c["id"] for c in _chunks(30)]

    def test_una_caida_persistente_deja_el_faltante_contado(self):
        """Si el proveedor sigue caido, la segunda pasada no salva nada y `sin_vector` lo dice:
        `embebidos` y `llamadas` son la SUMA de las dos pasadas, `sin_vector` es el de la
        ultima — cuantos quedaron."""
        # Los tres intentos de la primera pasada sobre el lote 2 (11, 21, 31) y los tres de la
        # segunda (42, 43, 44: ve SOLO esos 10 chunks y cada intento muere en su primer texto).
        cliente = _Cliente(fallar_en={11, 21, 31, 42, 43, 44})
        stats, grafo, _ = self._correr(_chunks(30), cliente=cliente, lote=10)
        assert stats["sin_vector"] == 10 and stats["embebidos"] == 20
        assert stats["pasadas"] == 2 and stats["lotes_fallidos"] == 2
        assert stats["llamadas"] == 44, "41 de la primera + 3 intentos que mueren en el 1er texto"
        assert len(grafo.ids_guardados) == 20

    def test_pasadas_uno_es_la_politica_de_antes(self):
        """`pasadas=1` es lo que usa el re-vectorizado a pedido: ahi no hay nada que esperar,
        porque quien apreto el boton decide cuando volver a intentar."""
        cliente = _Cliente(fallar_en={11, 21, 31})
        stats, _, _ = self._correr(_chunks(30), cliente=cliente, lote=10, pasadas=1)
        assert stats["pasadas"] == 1 and stats["sin_vector"] == 10

    def test_el_default_son_dos_pasadas(self):
        assert embeddings.PASADAS == 2


class TestUnaSolaFuenteDeVerdad:
    """El modelo, el prefijo y la sentencia que guarda el vector se escriben UNA vez.

    OJO, y esta dicho en el README: en este repo publico `vectorize.py` (raiz) es LEGACY y
    todavia tiene su propio bucle y su propio modelo escrito a mano; el test no lo tapa, lo
    deja afuera a proposito. Lo que se fija aca es lo que si es canonico: el paquete y la
    capa de consulta de la API, que ya delega.
    """

    def test_el_paquete_es_el_unico_que_nombra_el_modelo(self):
        texto = (RAIZ / "pipeline/embeddings.py").read_text(encoding="utf-8")
        assert '"gemini-embedding-2"' in texto

    def test_la_api_de_consulta_delega_en_el_paquete(self):
        codigo = (RAIZ / "api/services/vector.py").read_text(encoding="utf-8")
        codigo = "\n".join(l for l in codigo.splitlines() if not l.strip().startswith("#"))
        assert "from pipeline import embeddings" in codigo
        assert '"gemini-embedding-2"' not in codigo, "la API redefine el modelo"
        arbol = ast.parse(codigo)
        defs = {n.name for n in arbol.body if isinstance(n, ast.FunctionDef)}
        assert defs & {"build_embedding_text", "generate_embeddings", "crear_cliente"} == set()

    def test_ningun_writer_guarda_embedding_sin_forma(self):
        """La UNICA sentencia que guarda embeddings vive en el paquete y lleva
        `embedding_forma`: sin eso no se puede saber con que prefijo se embebio un chunk."""
        assert "embedding_forma" in embeddings.CYPHER_GUARDAR_EMBEDDINGS
        assert embeddings.EMBEDDING_FORMA == "canonico"
        for rel in ("api/services/vector.py", "migrate_chunks.py", "upload_chunks.py"):
            codigo = (RAIZ / rel).read_text(encoding="utf-8")
            codigo = "\n".join(l for l in codigo.splitlines() if not l.strip().startswith("#"))
            assert "SET c.embedding = u.embedding" not in codigo, f"{rel} guarda a mano"

    def test_el_paquete_no_importa_genai_al_cargar(self):
        arbol = ast.parse((RAIZ / "pipeline/embeddings.py").read_text(encoding="utf-8"))
        top = {n.module for n in arbol.body if isinstance(n, ast.ImportFrom)} | \
              {a.name for n in arbol.body if isinstance(n, ast.Import) for a in n.names}
        assert not any("genai" in (m or "") for m in top)
