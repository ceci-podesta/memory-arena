"""
Juez Mistral sobre LongMemEval oracle — VerbatimRAG.
Lee el JSONL más reciente de longmemeval_oracle en results/responses/
y corre el juez MistralJudge (Fase B).

Uso:
    uv run python scripts/run_longmemeval_judge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.evaluation.judge import MistralJudge
from memory_arena.evaluation.judgment_runner import run_judgment
from memory_arena.llm.ollama_client import OllamaClient

RESULTS_DIR = REPO_ROOT / "results"
RESPONSES_DIR = RESULTS_DIR / "responses"
JUDGE_MODEL = "mistral:7b"
JUDGE_NAME = "mistral"


def find_latest_longmemeval_jsonl() -> Path | None:
    """Busca el JSONL más reciente de longmemeval_oracle en results/responses/."""
    candidates = [
        p for p in RESPONSES_DIR.glob("*.jsonl") if "longmemeval_oracle" in p.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    jsonl = find_latest_longmemeval_jsonl()
    if jsonl is None:
        print(f"ERROR: no hay JSONL de longmemeval_oracle en {RESPONSES_DIR}")
        print("Corré primero: uv run python scripts/run_longmemeval_verbatimrag.py")
        return

    print(f"Input:  {jsonl.relative_to(REPO_ROOT)}")
    print(f"Juez:   {JUDGE_MODEL}")

    llm = OllamaClient(model=JUDGE_MODEL)
    judge = MistralJudge(llm=llm)

    result = run_judgment(
        responses_path=jsonl,
        judge=judge,
        output_dir=RESULTS_DIR,
        judge_name=JUDGE_NAME,
    )

    print(f"\nOutput:   {result['output_path']}")
    print(f"Total:    {result['total']} samples juzgados")
    print(f"Accuracy: {result['overall_accuracy']}")
    print(f"Tiempo:   {result['total_latency_s']:.1f}s")
    print("\nAccuracy por question_type:")
    for qtype, acc in result["by_type"].items():
        n = result["counts_by_type"][qtype]
        print(f"  {qtype}: {acc} (n={n})")


if __name__ == "__main__":
    main()
