"""Tests del runner de Fase A."""

import json
import socket
from pathlib import Path

import pytest

from memory_arena.benchmarks.longmemeval import SUBSET_ORACLE, load_longmemeval
from memory_arena.evaluation.runner import run_strategy
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.no_memory import NoMemoria


def _ollama_up(host: str = "127.0.0.1", port: int = 11434) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _ollama_up(),
    reason="Requiere servidor Ollama corriendo en 127.0.0.1:11434",
)
def test_run_strategy_escribe_jsonl_y_metadata(tmp_path: Path):
    samples = load_longmemeval(SUBSET_ORACLE, limit=2)
    strategy = NoMemoria()
    llm = OllamaClient()

    metadata = run_strategy(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name="no_memory",
        benchmark_name="longmemeval_oracle",
        output_dir=tmp_path,
    )

    # Metadata completa
    assert metadata.run_id.endswith("_no_memory_longmemeval_oracle")
    assert metadata.num_samples == 2
    assert metadata.ended_at is not None
    assert metadata.duration_seconds is not None and metadata.duration_seconds > 0

    # Archivo de metadata existe y es parseable
    runs_file = tmp_path / "runs" / f"{metadata.run_id}.json"
    assert runs_file.exists()
    with open(runs_file, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["strategy"] == "no_memory"

    # JSONL de respuestas: 2 lineas, una por sample
    responses_file = tmp_path / "responses" / f"{metadata.run_id}.jsonl"
    assert responses_file.exists()
    with open(responses_file, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2

    record = json.loads(lines[0])
    # Campos minimos
    for field_name in (
        "sample_id",
        "question",
        "question_type",
        "expected_answer",
        "system_answer",
        "retrieved_context",
        "latency_s",
    ):
        assert field_name in record, f"falta {field_name}"

    # NoMemoria siempre devuelve contexto vacio
    assert record["retrieved_context"] == []
    assert isinstance(record["system_answer"], str) and record["system_answer"]
