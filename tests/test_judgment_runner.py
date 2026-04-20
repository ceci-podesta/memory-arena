"""Test end-to-end del judgment_runner con Mistral 7B."""

import json
import socket
from pathlib import Path

import pytest

from memory_arena.evaluation.judge import MistralJudge
from memory_arena.evaluation.judgment_runner import run_judgment


def _ollama_up(host: str = "127.0.0.1", port: int = 11434) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _ollama_up(),
    reason="Requiere servidor Ollama corriendo con modelo mistral:7b",
)
def test_run_judgment_escribe_jsonl_y_devuelve_stats(tmp_path: Path):
    # JSONL sintetico: 1 respuesta obviamente correcta, 1 obviamente incorrecta
    responses_path = tmp_path / "fake_run_smoke_longmemeval_oracle.jsonl"
    fake_records = [
        {
            "sample_id": "fake_001",
            "question": "What is the capital of France?",
            "question_type": "single-session-user",
            "question_date": "2024/01/01",
            "expected_answer": "Paris",
            "system_answer": "The capital of France is Paris.",
            "retrieved_context": [],
            "latency_s": 0.5,
        },
        {
            "sample_id": "fake_002",
            "question": "What is the capital of France?",
            "question_type": "single-session-user",
            "question_date": "2024/01/01",
            "expected_answer": "Paris",
            "system_answer": "The capital of France is Madrid.",
            "retrieved_context": [],
            "latency_s": 0.5,
        },
    ]
    with open(responses_path, "w", encoding="utf-8") as f:
        for r in fake_records:
            f.write(json.dumps(r) + "\n")

    judge = MistralJudge()
    stats = run_judgment(
        responses_path=responses_path,
        judge=judge,
        output_dir=tmp_path,
        judge_name="mistral",
    )

    # Estructura de stats
    assert stats["total"] == 2
    assert stats["judge_model"] == "mistral:7b"
    assert stats["counts_by_type"]["single-session-user"] == 2
    assert 0.0 <= stats["overall_accuracy"] <= 1.0

    # Output file bien formado
    out_path = tmp_path / "judgments" / f"{responses_path.stem}__mistral.jsonl"
    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        for field_name in ("sample_id", "question_type", "label", "judge_model", "raw_response"):
            assert field_name in rec
        assert isinstance(rec["label"], bool)

    # Smoke check del criterio del juez: con casos obvios, deberia acertar.
    # No lo hago assert duro porque el LLM puede fallar ocasionalmente
    # y no queremos un test flaky. Si falla mucho, es senal para el informe.
    labels = [json.loads(line)["label"] for line in lines]
    print(f"\n  Labels del smoke test: {labels} (esperado [True, False])")
