"""Basic tests for the root scripts that are NOT part of the mirrored pipeline.

Kept from v1.0 (and still green): it covers `extract_entities.py` and `dedup_entities.py` —
the entity-extraction and deduplication steps, which have no counterpart under `pipeline/` and
are this repo's own code — plus the chunking constants re-exported by `parser_v2.py` and a few
hygiene checks (no real credentials in `.env.example` or `docker-compose.yml`).

The regression suite for the mirrored pipeline lives in `test_pipeline_*.py`; repo-wide health
(everything parses, every dependency declared, no secrets) lives in `test_repo.py`.
"""

import os
import sys

import pytest

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParser:
    """Test PDF parser and chunking logic."""

    def test_normalize_name(self):
        from extract_entities import normalize_name
        assert normalize_name("  Otitis Media Aguda  ") == "otitis media aguda"
        assert normalize_name("hipertensión...") == "hipertensión"
        assert normalize_name("  múltiples   espacios  ") == "múltiples espacios"
        assert normalize_name("") == ""

    def test_clave_por_acentos(self):
        """La regla del dedup: sin tildes y en minusculas, y nada mas."""
        from dedup_entities import clave_por_acentos
        assert clave_por_acentos("hipertensión") == "hipertension"
        assert clave_por_acentos("DIAGNÓSTICO") == "diagnostico"
        assert clave_por_acentos("normal") == "normal"

    def test_el_canonico_es_el_de_mayor_degree(self):
        """Reescrito el 22-sep-2026: antes se elegia por (tiene acentos, `freq`), y `freq` se
        sobreescribe en cada carga de libro -- en el grafo es la del ULTIMO libro que cargo la
        entidad, no la del corpus. El degree acumula, y es lo que decide quien gana en la
        recuperacion."""
        from dedup_entities import canonico_de
        canon, dups = canonico_de([
            {"eid": "a", "nombre": "hipertension", "freq": 10, "degree": 3},
            {"eid": "b", "nombre": "hipertensión", "freq": 5, "degree": 40},
        ])
        assert canon["nombre"] == "hipertensión" and [d["eid"] for d in dups] == ["a"]

    def test_los_grupos_llevan_los_labels(self):
        """Nunca entre labels distintos, y por construccion: la clave del grupo ES (labels, nombre
        plegado). Agrupar label por label dejaba fundir un nodo de dos labels con uno de uno."""
        from dedup_entities import grupos_de
        entidades = [
            {"eid": "1", "nombre": "hipertensión", "labels": ["Patologia"], "freq": 1, "degree": 9},
            {"eid": "2", "nombre": "hipertension", "labels": ["Patologia"], "freq": 1, "degree": 2},
            {"eid": "3", "nombre": "diabetes", "labels": ["Patologia"], "freq": 1, "degree": 1},
            {"eid": "4", "nombre": "hipertension", "labels": ["Hallazgo"], "freq": 1, "degree": 5},
        ]
        grupos = grupos_de(entidades)
        assert len(grupos) == 1
        assert grupos[0]["labels"] == ["Patologia"]
        assert grupos[0]["canonico"]["nombre"] == "hipertensión"
        assert [d["eid"] for d in grupos[0]["duplicados"]] == ["2"]

    def test_la_fusion_escribe_la_procedencia_de_la_arista(self):
        """El Cypher que funde dos nodos es UNO SOLO (`pipeline/fusion.py`) y funde las cuatro
        listas paralelas de la arista: sin eso, fundir borraba el linaje del fragmento."""
        from pipeline.fusion import sentencias_de_grupo
        sent = sentencias_de_grupo("C", ["D"], ["SE_TRATA_CON"])
        assert [s["nombre"] for s in sent] == ["reapuntar_salientes", "reapuntar_entrantes",
                                               "canonico", "borrar_dups"]
        for campo in ("chunks", "libros", "perfiles", "evidencias"):
            assert f"nr.{campo}" in sent[0]["cypher"]


class TestChunking:
    """Test chunking parameters and structure."""

    def test_chunk_config_values(self):
        from parser_v2 import MAX_SIZE, MIN_SIZE, OVERLAP_SIZE, TARGET_SIZE
        assert TARGET_SIZE == 280
        assert MIN_SIZE == 150
        assert MAX_SIZE == 380
        assert OVERLAP_SIZE == 60
        assert MIN_SIZE < TARGET_SIZE < MAX_SIZE

    def test_parent_config(self):
        from parser_v2 import MAX_PARENT_WORDS, PARENT_WINDOW
        assert PARENT_WINDOW == 3
        assert MAX_PARENT_WORDS == 1200


class TestExtraction:
    """Test entity extraction validation logic."""

    def test_validate_entity_types(self):
        from extract_entities import ENTITY_TYPES
        expected = {
            "patologia", "estructura_anatomica", "procedimiento",
            "farmaco", "grupo_farmacologico", "agente", "signo",
            "sintoma", "metodo_dx", "hallazgo", "parametro",
        }
        assert ENTITY_TYPES == expected

    def test_validate_relation_types(self):
        from extract_entities import RELATION_TYPES
        assert "CAUSADA_POR" in RELATION_TYPES
        assert "SE_TRATA_CON" in RELATION_TYPES
        assert "SE_DIAGNOSTICA_CON" in RELATION_TYPES
        assert "ASOCIADA_A" in RELATION_TYPES

    def test_type_to_label_mapping(self):
        from extract_entities import TYPE_TO_LABEL
        assert TYPE_TO_LABEL["patologia"] == "Patologia"
        assert TYPE_TO_LABEL["farmaco"] == "Farmaco"
        assert TYPE_TO_LABEL["estructura_anatomica"] == "EstructuraAnatomica"

    def test_validate_extraction_filters_invalid(self):
        from extract_entities import _validate_extraction
        data = {
            "entidades": [
                {"nombre": "otitis media", "tipo": "patologia", "sinonimos": []},
                {"nombre": "x", "tipo": "patologia", "sinonimos": []},  # too short
                {"nombre": "valid", "tipo": "INVALID_TYPE", "sinonimos": []},  # bad type
                {"nombre": "", "tipo": "patologia", "sinonimos": []},  # empty
            ],
            "relaciones": []
        }
        result = _validate_extraction(data, {"id": "test", "libro_id": "test"})
        assert len(result["entidades"]) == 1
        assert result["entidades"][0]["nombre"] == "otitis media"


class TestCanonicalization:
    """Test entity canonicalization logic."""

    def test_canonicalize_merges_duplicates(self):
        from extract_entities import canonicalize_entities
        extractions = [
            {
                "entidades": [
                    {"nombre": "otitis media", "tipo": "patologia", "sinonimos": ["OMA"]},
                ],
                "relaciones": [],
                "chunk_id": "c1",
            },
            {
                "entidades": [
                    {"nombre": "otitis media", "tipo": "patologia", "sinonimos": []},
                ],
                "relaciones": [],
                "chunk_id": "c2",
            },
        ]
        entities, relations = canonicalize_entities(extractions)
        # Should merge into 1 entity
        otitis = [e for e in entities if e["nombre"] == "otitis media"]
        assert len(otitis) == 1
        assert otitis[0]["freq"] == 2
        assert "OMA" in otitis[0]["sinonimos"]


class TestAPIConfig:
    """Test API configuration basics."""

    def test_env_example_exists(self):
        assert os.path.exists(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env.example")
        )

    def test_requirements_no_vertexai(self):
        req_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "requirements.txt"
        )
        with open(req_path) as f:
            content = f.read()
        assert "vertexai" not in content.lower()
        assert "aiplatform" not in content.lower()

    def test_docker_compose_exists(self):
        assert os.path.exists(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docker-compose.yml")
        )

    def test_docker_compose_no_real_credentials(self):
        compose_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docker-compose.yml"
        )
        with open(compose_path) as f:
            content = f.read()
        assert "changeme" in content
        assert "AIzaSy" not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
