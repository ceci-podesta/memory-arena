"""
Tests para el loader de LongMemEval.

Los tests de integracion (que descargan el dataset real) se skipean si no hay
conexion a HuggingFace. Los unit tests siempre corren.
"""

import pytest

from memory_arena.benchmarks.longmemeval import (
    SUBSET_ORACLE,
    SUBSET_S_CLEANED,
    VALID_SUBSETS,
    LongMemEvalSample,
    Session,
    _parse_sample,
    load_longmemeval,
)
from memory_arena.memories.base import Turn


def _huggingface_disponible() -> bool:
    try:
        import requests
        requests.head("https://huggingface.co", timeout=3)
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# Unit tests (no requieren red ni descarga)
# ------------------------------------------------------------

def test_subsets_validos_incluyen_los_esperados():
    assert SUBSET_ORACLE in VALID_SUBSETS
    assert SUBSET_S_CLEANED in VALID_SUBSETS


def test_subset_invalido_levanta_error():
    with pytest.raises(ValueError):
        load_longmemeval("subset_que_no_existe")


def test_parse_sample_construye_dataclass_correcto():
    """_parse_sample debe mapear bien el JSON crudo a nuestros dataclasses."""
    raw = {
        "question_id": "q1",
        "question": "What did I eat for breakfast?",
        "question_type": "single-session-user",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer": "Pancakes",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": [
            "2023/05/20 (Sat) 02:21",
            "2023/05/21 (Sun) 09:15",
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I had pancakes."},
                {"role": "assistant", "content": "Sounds tasty."},
            ],
            [
                {"role": "user", "content": "The weather is nice."},
            ],
        ],
    }

    sample = _parse_sample(raw)

    assert isinstance(sample, LongMemEvalSample)
    assert sample.question_id == "q1"
    assert sample.expected_answer == "Pancakes"
    assert len(sample.haystack) == 2
    assert isinstance(sample.haystack[0], Session)
    assert sample.haystack[0].session_id == "s1"
    assert len(sample.haystack[0].turns) == 2
    assert isinstance(sample.haystack[0].turns[0], Turn)
    assert sample.haystack[0].turns[0].role == "user"
    assert sample.haystack[0].turns[0].content == "I had pancakes."


def test_parse_sample_preserva_orden_de_sesiones():
    """La sesion i en las tres listas paralelas debe quedar agrupada."""
    raw = {
        "question_id": "q1",
        "question": "?",
        "question_type": "x",
        "question_date": "d",
        "answer": "a",
        "answer_session_ids": [],
        "haystack_session_ids": ["alpha", "beta", "gamma"],
        "haystack_dates": ["d1", "d2", "d3"],
        "haystack_sessions": [[], [], []],
    }

    sample = _parse_sample(raw)

    assert [s.session_id for s in sample.haystack] == ["alpha", "beta", "gamma"]
    assert [s.date for s in sample.haystack] == ["d1", "d2", "d3"]


# ------------------------------------------------------------
# Integration tests (requieren descarga de HuggingFace)
# ------------------------------------------------------------

@pytest.mark.skipif(
    not _huggingface_disponible(),
    reason="HuggingFace no accesible",
)
def test_load_oracle_con_limit():
    """Smoke test: cargar 2 samples del subset oracle debe funcionar."""
    samples = load_longmemeval(SUBSET_ORACLE, limit=2)
    assert len(samples) == 2
    assert all(isinstance(s, LongMemEvalSample) for s in samples)
    # Cada sample debe tener al menos una sesion en el haystack.
    assert all(len(s.haystack) > 0 for s in samples)
