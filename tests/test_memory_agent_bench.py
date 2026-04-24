
"""Tests del loader de MemoryAgentBench."""

from __future__ import annotations

import pytest

from memory_arena.benchmarks.memory_agent_bench import (
    ALL_SPLITS,
    MABSample,
    SPLIT_AR,
    SPLIT_CR,
    SUB_AR_LONGMEMEVAL_S,
    SUB_CR_FACTCONSOL_SH_6K,
    load_mab,
)


def test_constants_and_splits():
    assert SPLIT_AR in ALL_SPLITS
    assert SPLIT_CR in ALL_SPLITS
    assert len(ALL_SPLITS) == 4


def test_sample_validates_parallel_lengths():
    with pytest.raises(ValueError):
        MABSample(
            sample_id="x",
            source="s",
            context="ctx",
            questions=["q1", "q2"],
            answers=[["a"]],  # 1 answer for 2 questions -> debe romper
            question_ids=["id1", "id2"],
        )


def test_invalid_split_raises():
    with pytest.raises(ValueError, match="split desconocido"):
        load_mab("NoExiste", "whatever")


@pytest.mark.integration
def test_load_ar_longmemeval_s_small():
    """Integration test: requiere el dataset cacheado localmente.

    Carga un subset chico del sub-dataset longmemeval_s para validar formato.
    """
    samples = load_mab(SPLIT_AR, SUB_AR_LONGMEMEVAL_S, max_samples=2)

    assert len(samples) == 2
    for s in samples:
        assert isinstance(s, MABSample)
        assert s.source == SUB_AR_LONGMEMEVAL_S
        assert isinstance(s.context, str) and len(s.context) > 0
        assert len(s.questions) >= 1
        assert len(s.questions) == len(s.answers) == len(s.question_ids)
        # Cada answer debe ser lista de strings
        for a in s.answers:
            assert isinstance(a, list)
            for alt in a:
                assert isinstance(alt, str)


@pytest.mark.integration
def test_load_cr_factconsolidation_small():
    samples = load_mab(SPLIT_CR, SUB_CR_FACTCONSOL_SH_6K, max_samples=2)
    assert 1 <= len(samples) <= 2  # el sub-dataset puede tener muy pocos
    for s in samples:
        assert s.source == SUB_CR_FACTCONSOL_SH_6K
        assert len(s.questions) == len(s.answers)


@pytest.mark.integration
def test_load_unknown_sub_dataset_raises():
    with pytest.raises(ValueError, match="No hay samples"):
        load_mab(SPLIT_AR, "sub_dataset_que_no_existe")

