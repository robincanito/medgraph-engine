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

    def test_strip_accents(self):
        from dedup_entities import strip_accents
        assert strip_accents("hipertensión") == "hipertension"
        assert strip_accents("diagnóstico") == "diagnostico"
        assert strip_accents("farmacología") == "farmacologia"
        assert strip_accents("normal") == "normal"

    def test_has_accents(self):
        from dedup_entities import has_accents
        assert has_accents("hipertensión") is True
        assert has_accents("hipertension") is False

    def test_pick_canonical(self):
        from dedup_entities import pick_canonical
        nodes = [
            {"nombre": "hipertension", "freq": 10, "sinonimos": []},
            {"nombre": "hipertensión", "freq": 5, "sinonimos": ["HTA"]},
        ]
        canon = pick_canonical(nodes)
        assert canon["nombre"] == "hipertensión"  # prefers accented

    def test_find_accent_groups(self):
        from dedup_entities import find_accent_groups
        entities = [
            {"nombre": "hipertensión"},
            {"nombre": "hipertension"},
            {"nombre": "diabetes"},
        ]
        groups = find_accent_groups(entities)
        assert len(groups) == 1
        assert "hipertension" in groups
        assert len(groups["hipertension"]) == 2


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
