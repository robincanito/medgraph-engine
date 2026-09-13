"""pipeline/carga.py — la carga IDEMPOTENTE de un documento al grafo.

Incidentes que fijan estos tests:
  - el CLI hacia CREATE sin borrar lo previo: re-ingestar fallaba por la constraint de id
    mientras el docstring decia "idempotente";
  - la API hacia DETACH DELETE + CREATE: el documento desaparecia durante la carga y se
    perdian las entidades extraidas y los embeddings ya pagados.

Ningun grafo: `write` y `query` son dobles que registran (cypher, params). El objetivo no es
probar Neo4j —eso pide un banco, y en este repo no hay— sino que las SENTENCIAS sean las
canonicas: MERGE por id, huerfanos borrados contra los ids cargados, relaciones derivadas
recomputadas, y fail-closed antes de escribir una sola linea.
"""
from pathlib import Path

import pytest

from pipeline import carga

RAIZ = Path(__file__).resolve().parent.parent


def _chunk(i, libro="libro-x", **extra):
    c = {"id": f"{libro}_v2_{i:05d}", "libro_id": libro, "text": f"texto {i}", "page_start": 1,
         "page_end": 1, "word_count": 2, "chunk_index": i,
         "parent_id": f"{libro}_v2_parent_00000"}
    c.update(extra)
    return c


def _parent(i, libro="libro-x"):
    return {"id": f"{libro}_v2_parent_{i:05d}", "libro_id": libro, "text": "t",
            "page_start": 1, "page_end": 1, "word_count": 1}


class _Escritor:
    def __init__(self):
        self.llamadas = []

    def __call__(self, cypher, params=None):
        self.llamadas.append((" ".join(cypher.split()), params or {}))

    def cyphers(self):
        return [c for c, _ in self.llamadas]


class TestPreparar:
    def test_defaults_para_lo_ausente(self):
        out = carga.preparar([_chunk(0)], carga.CAMPOS_CHUNK)
        assert out[0]["id"] == "libro-x_v2_00000"
        assert out[0]["props"]["tipo_contenido"] == "body"
        assert out[0]["props"]["version"] == 2
        assert set(out[0]["props"]) == set(carga.CAMPOS_CHUNK)

    def test_obligatorio_ausente_falla_cerrado(self):
        c = _chunk(0)
        del c["text"]
        with pytest.raises(KeyError):
            carga.preparar([c], carga.CAMPOS_CHUNK)

    def test_solo_guarda_los_campos_declarados(self):
        out = carga.preparar([_chunk(0, embedding=[1.0], otra="x")], carga.CAMPOS_CHUNK)
        assert "embedding" not in out[0]["props"] and "otra" not in out[0]["props"]


class TestCargarLibro:
    def test_merge_por_id_y_huerfanos_fuera(self):
        w = _Escritor()
        r = carga.cargar_libro(w, "libro-x", [_chunk(0), _chunk(1)], [_parent(0)])
        cy = " || ".join(w.cyphers())
        assert "CREATE (" not in cy, "la carga es MERGE por id, nunca CREATE"
        assert "MERGE (c:Chunk {id: chunk.id})" in cy
        assert "MERGE (p:ParentChunk {id: parent.id})" in cy
        assert "WHERE NOT c.id IN $ids DETACH DELETE c" in cy
        assert "WHERE NOT p.id IN $ids DETACH DELETE p" in cy
        assert r == {"children": 2, "parents": 1}

    def test_texto_cambiado_invalida_embedding_y_forma(self):
        """Si el texto de un chunk cambio, el vector que se pago ya no lo representa: se pone
        en null y el paso de vectorizado lo vuelve a pedir. La comparacion va contra el texto
        VIEJO, o sea ANTES de pisarlo."""
        w = _Escritor()
        carga.cargar_libro(w, "libro-x", [_chunk(0)], [_parent(0)])
        merge = next(c for c in w.cyphers() if "MERGE (c:Chunk" in c)
        assert "ON MATCH SET c.embedding = CASE WHEN c.text <> chunk.props.text THEN null" in merge
        assert "c.embedding_forma = CASE WHEN c.text <> chunk.props.text THEN null" in merge
        assert merge.index("ON MATCH SET") < merge.index("SET c += chunk.props")

    def test_relaciones_derivadas_se_recomputan(self):
        w = _Escritor()
        carga.cargar_libro(w, "libro-x", [_chunk(0)], [_parent(0)])
        cy = " || ".join(w.cyphers())
        assert "MERGE (child)-[:CHILD_OF]->(parent)" in cy
        assert "WHERE p.id <> child.parent_id DELETE r" in cy, \
            "CHILD_OF viejo de un chunk re-parentado"
        assert "[r:SIGUE_A]->(:Chunk) DELETE r" in cy, "SIGUE_A es derivada: se borra y se reteje"
        assert "MERGE (c1)-[:SIGUE_A]->(c2)" in cy

    def test_los_huerfanos_se_calculan_contra_los_ids_cargados(self):
        w = _Escritor()
        carga.cargar_libro(w, "libro-x", [_chunk(0), _chunk(1)], [_parent(0)])
        _, params = next(l for l in w.llamadas if "NOT c.id IN $ids" in l[0])
        assert params == {"lid": "libro-x", "ids": ["libro-x_v2_00000", "libro-x_v2_00001"]}

    def test_va_en_lotes(self, monkeypatch):
        monkeypatch.setattr(carga, "BATCH_SIZE", 2)
        w = _Escritor()
        carga.cargar_libro(w, "libro-x", [_chunk(i) for i in range(5)], [_parent(0)])
        assert sum(1 for c in w.cyphers() if "MERGE (c:Chunk" in c) == 3

    def test_sin_chunks_no_toca_el_grafo(self):
        """Fail-closed: un parse vacio (un PDF escaneado, por ejemplo) no puede vaciar lo que
        ya estaba cargado."""
        w = _Escritor()
        with pytest.raises(ValueError):
            carga.cargar_libro(w, "libro-x", [], [])
        assert w.llamadas == []

    def test_libro_id_ajeno_falla_antes_de_escribir(self):
        w = _Escritor()
        with pytest.raises(ValueError):
            carga.cargar_libro(w, "otro-libro", [_chunk(0)], [_parent(0)])
        assert w.llamadas == []

    def test_un_parent_ajeno_tambien_aborta(self):
        w = _Escritor()
        with pytest.raises(ValueError):
            carga.cargar_libro(w, "libro-x", [_chunk(0)], [_parent(0, libro="otro")])
        assert w.llamadas == []

    def test_reporta_progreso(self):
        eventos = []
        carga.cargar_libro(_Escritor(), "libro-x", [_chunk(0)], [_parent(0)],
                           on_progress=lambda step, pct, msg: eventos.append((step, pct)))
        assert {s for s, _ in eventos} == {"upload"}
        assert eventos[-1][1] == 90


class _Grafo:
    """Doble de query+write que responde segun el Cypher (no segun el orden de llamada)."""

    def __init__(self, units=3, parents=1, relations=140, orphans=2, lotes_chunks=2):
        self.escrituras, self.lecturas = [], []
        self.n = dict(units=units, parents=parents, relations=relations, orphans=orphans)
        self.lotes_chunks = lotes_chunks
        self.vinculos = 1

    def query(self, cypher, params=None):
        c = " ".join(cypher.split())
        self.lecturas.append((c, params or {}))
        for k in ("units", "parents", "relations", "orphans"):
            if f"AS {k}" in c:
                return [{k: self.n[k]}]
        if "AS vinculados" in c:
            self.vinculos -= 1
            return [{"vinculados": 1 if self.vinculos >= 0 else 0}]
        if "RETURN chunks, emb" in c:
            return [{"chunks": self.n["units"], "emb": self.n["units"]}]
        if "DETACH DELETE c" in c:
            self.lotes_chunks -= 1
            return [{"borrados": 1000 if self.lotes_chunks >= 0 else 0}]
        if "DETACH DELETE p" in c:
            return [{"borrados": 0}]
        raise AssertionError(f"query inesperada: {c[:80]}")

    def write(self, cypher, params=None):
        self.escrituras.append((" ".join(cypher.split()), params or {}))


class TestFuenteYBorrado:
    def test_registrar_fuente_merge_book_contains_en_lotes_y_conteos(self):
        g = _Grafo(units=7)
        r = carga.registrar_fuente(g.query, g.write, "libro-x",
                                   {"title": "T", "author": None, "source_kind": "catedra"})
        assert r == {"chunk_count": 7, "embedded_count": 7}
        cy, params = g.escrituras[0]
        assert "MERGE (b:Book {id: $lid})" in cy and "SET b += $props" in cy
        assert params["props"] == {"title": "T", "source_kind": "catedra"}, \
            "los None no pisan lo que ya habia"
        assert sum(1 for c, _ in g.lecturas if "AS vinculados" in c) == 2, \
            "vincula hasta que no queda nada"
        assert any("SET b.chunk_count = chunks" in c for c, _ in g.lecturas)

    def test_dry_run_solo_cuenta(self):
        g = _Grafo()
        p = carga.borrar_libro(g.query, g.write, "libro-x", dry_run=True)
        assert p == {"source_id": "libro-x", "dry_run": True, "units": 3, "parents": 1,
                     "relations": 140, "entities_orphaned": 2, "orphan_policy": "keep",
                     "deleted_at": None}
        assert g.escrituras == [] and not any("DELETE" in c for c, _ in g.lecturas)

    def test_borrado_real_en_lotes_y_book_al_final(self):
        g = _Grafo(lotes_chunks=3)
        p = carga.borrar_libro(g.query, g.write, "libro-x", dry_run=False, lote=1000)
        assert p["dry_run"] is False and p["deleted_at"]
        borrados = [c for c, _ in g.lecturas if "DETACH DELETE c" in c]
        assert len(borrados) == 4, "tres lotes con datos y uno vacio que corta"
        assert any("DETACH DELETE p" in c for c, _ in g.lecturas)
        assert g.escrituras[-1][0] == "MATCH (b:Book {id: $lid}) DETACH DELETE b"
        assert p["orphan_policy"] == "keep"


class TestUnaSolaCopia:
    """La carga idempotente vive en el paquete y en ningun otro lado.

    OJO, y esta dicho en el README: `migrate_chunks.py` y `upload_chunks.py` (raiz) son el
    camino LEGACY de este repo publico y todavia hacen `CREATE` por su cuenta. No se los mete
    en este test —un test que fije el defecto lo vuelve intocable— pero el paquete si tiene
    que seguir siendo MERGE-only.
    """

    def test_el_paquete_no_crea_nodos_a_mano(self):
        codigo = (RAIZ / "pipeline/carga.py").read_text(encoding="utf-8")
        cyphers = "\n".join(getattr(carga, n) for n in dir(carga)
                            if n.startswith("CYPHER_") and isinstance(getattr(carga, n), str))
        assert "CREATE (" not in cyphers, "alguna sentencia del paquete volvio a CREATE"
        assert "MERGE (c:Chunk {id: chunk.id})" in codigo
