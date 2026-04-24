"""Integration test del runner MAB con NoMemoria + Ollama local."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_CR,
    SUB_CR_FACTCONSOL_SH_6K,
    load_mab,
)
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.no_memory import NoMemoria

OLLAMA_AVAILABLE = shutil.which("ollama") is not None


@pytest.mark.integration
@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="requiere ollama local")
def test_run_mab_smoke_cr_factconsol(tmp_path):
    """Smoke test: NoMemoria sobre 1 sample de CR con tope de 3 preguntas.

    Valida que:
    - el runner corre end-to-end sin errores.
    - escribe un JSONL con N preguntas.
    - cada registro tiene las keys esperadas.
    """
    samples = load_mab(SPLIT_CR, SUB_CR_FACTCONSOL_SH_6K, max_samples=1)
    assert len(samples) == 1

    # Recortamos a 3 preguntas para smoke test rápido
    s = samples[0]
    s_truncated = type(s)(
        sample_id=s.sample_id,
        source=s.source,
        context=s.context,
        questions=s.questions[:3],
        answers=s.answers[:3],
        question_ids=s.question_ids[:3],
    )

    llm = OllamaClient(model="llama3.2:3b")
    strategy = NoMemoria()

    meta = run_strategy_mab(
        strategy=strategy,
        samples=[s_truncated],
        llm=llm,
        strategy_name="no_memory",
        split=SPLIT_CR,
        sub_dataset=SUB_CR_FACTCONSOL_SH_6K,
        output_dir=tmp_path,
        max_new_tokens=30,
    )

    jsonl_files = list((tmp_path / "responses").glob("*.jsonl"))
    assert len(jsonl_files) == 1
    out_path = jsonl_files[0]


    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        rec = json.loads(line)
        assert set(rec.keys()) >= {
            "sample_id",
            "source",
            "question_id",
            "question",
            "gold_answers",
            "system_answer",
        }
        assert rec["source"] == SUB_CR_FACTCONSOL_SH_6K
        assert isinstance(rec["system_answer"], str)
