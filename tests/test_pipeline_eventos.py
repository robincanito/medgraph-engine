"""pipeline/eventos.py — los eventos nombrados del pipeline: el CONTRATO del observador.

POR QUE EXISTE. Una corrida del pipeline dejaba prosa ("  [3] 41 embeddings"): no se podia
atribuir una llamada paga a un documento, ni contar reintentos, ni saber por que paso iba una
ingesta. Estos tests fijan los NOMBRES y los CAMPOS, porque quien mide una corrida los lee
desde AFUERA: si alguien renombra `n_textos` o mete un campo llamado `message`, esto se pone
rojo antes de que las metricas empiecen a mentir.

Sobre `logging` de la biblioteca estandar y NADA MAS: el modulo tiene que servir igual en este
repo, que no tiene el formateador de logs del privado. Cero red, cero grafo: el logger es un
handler en memoria.
"""
import logging
from pathlib import Path

import pytest

from pipeline import eventos

RAIZ = Path(__file__).resolve().parent.parent


class _Captura(logging.Handler):
    """Handler que guarda los records enteros: se miran `evento` y `campos`, no el texto."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    @property
    def eventos(self):
        return [(r.evento, dict(r.campos)) for r in self.records if hasattr(r, "evento")]


@pytest.fixture
def log_capturado():
    log = logging.getLogger("test_eventos")
    log.handlers = []
    log.setLevel(logging.INFO)
    log.propagate = False
    captura = _Captura()
    log.addHandler(captura)
    yield log, captura
    log.handlers = []


class TestEmitir:
    def test_pone_evento_y_campos_en_el_record(self, log_capturado):
        log, captura = log_capturado
        eventos.emitir(log, "embed_lote", libro_id="farreras", n_textos=10, intento=1,
                       estado="ok", ms=812)
        assert captura.eventos == [("embed_lote", {"libro_id": "farreras", "n_textos": 10,
                                                   "intento": 1, "estado": "ok", "ms": 812})]

    def test_el_mensaje_del_log_es_el_nombre_del_evento(self, log_capturado):
        log, captura = log_capturado
        eventos.emitir(log, "llm_llamada", paso="extract")
        assert captura.records[0].getMessage() == "llm_llamada"
        assert captura.records[0].levelno == logging.INFO

    @pytest.mark.parametrize("clave", sorted(eventos.CLAVES_RESERVADAS))
    def test_una_clave_del_sobre_json_es_ValueError(self, log_capturado, clave):
        """Un campo llamado `message` o `severity` pisaria el sobre del log en silencio."""
        log, captura = log_capturado
        with pytest.raises(ValueError, match=clave):
            eventos.emitir(log, "embed_lote", **{clave: "x"})
        assert captura.eventos == [], "no se emite nada si la validacion falla"

    def test_sin_campos_tambien_vale(self, log_capturado):
        log, captura = log_capturado
        eventos.emitir(log, "ingest_paso")
        assert captura.eventos == [("ingest_paso", {})]


class TestPaso:
    def test_inicio_y_fin_con_ms_y_n(self, log_capturado):
        log, captura = log_capturado
        with eventos.paso(log, "farreras", "vectorize", origen="cli") as p:
            p.campos["n"] = 41
        nombres = [n for n, _ in captura.eventos]
        assert nombres == ["ingest_paso", "ingest_paso"]
        inicio, fin = (c for _, c in captura.eventos)
        assert inicio == {"libro_id": "farreras", "paso": "vectorize", "origen": "cli",
                          "estado": "inicio"}
        assert fin["estado"] == "fin" and fin["n"] == 41
        assert fin["libro_id"] == "farreras" and fin["paso"] == "vectorize"
        assert fin["origen"] == "cli"
        assert isinstance(fin["ms"], int) and fin["ms"] >= 0

    def test_sin_n_el_fin_no_lo_inventa(self, log_capturado):
        log, captura = log_capturado
        with eventos.paso(log, "x", "download", origen="api"):
            pass
        assert "n" not in captura.eventos[1][1]

    def test_error_lleva_tipo_y_detalle_y_RELANZA(self, log_capturado):
        log, captura = log_capturado
        with pytest.raises(RuntimeError, match="se cayo el proveedor"):
            with eventos.paso(log, "farreras", "upload", origen="api") as p:
                p.campos["n"] = 7
                raise RuntimeError("se cayo el proveedor")
        _, err = captura.eventos[1]
        assert err["estado"] == "error"
        assert err["tipo"] == "RuntimeError"
        assert err["detalle"] == "se cayo el proveedor"
        assert isinstance(err["ms"], int)
        assert err["n"] == 7, "lo que el cuerpo alcanzo a contar sobrevive al error"

    def test_el_detalle_se_corta_en_200(self, log_capturado):
        log, captura = log_capturado
        with pytest.raises(ValueError):
            with eventos.paso(log, "x", "parse", origen="cli"):
                raise ValueError("z" * 500)
        assert len(captura.eventos[1][1]["detalle"]) == eventos.MAX_DETALLE == 200

    def test_el_cuerpo_no_puede_pisar_los_campos_fijos(self, log_capturado):
        """`estado`, `ms`, `paso` y `origen` son del contrato, no del cuerpo."""
        log, captura = log_capturado
        with eventos.paso(log, "x", "parse", origen="cli") as p:
            p.campos["estado"] = "mentira"
            p.campos["paso"] = "otro"
        _, fin = captura.eventos[1]
        assert fin["estado"] == "fin" and fin["paso"] == "parse"

    def test_el_vocabulario_esta_declarado(self):
        assert eventos.PASOS == ("download", "classify", "parse", "upload", "vectorize",
                                 "fuente", "extract")
        assert eventos.ESTADOS_PASO == ("inicio", "fin", "error")
        assert eventos.ORIGENES == ("cli", "api")
        assert eventos.ESTADOS_LOTE == ("ok", "reintento", "fallo")


class TestSoloLoggingEstandar:
    """El paquete tiene que funcionar en un repo pelado: nada de modulos del privado."""

    @pytest.mark.parametrize("rel", ["pipeline/eventos.py", "pipeline/embeddings.py",
                                     "pipeline/carga.py", "pipeline/parseo.py"])
    def test_ningun_modulo_del_pipeline_importa_bitacora(self, rel):
        texto = (RAIZ / rel).read_text(encoding="utf-8")
        codigo = "\n".join(l for l in texto.splitlines() if not l.strip().startswith("#"))
        assert "import bitacora" not in codigo, f"{rel} importa bitacora (no existe aca)"
