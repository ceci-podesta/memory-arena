"""
LongMemEval — Fase B: juez LLM (Mistral 7B) sobre respuestas de Fase A.

Lee el JSONL más reciente de results/responses/ que matchee el subset
configurado, aplica MistralJudge y escribe:
  - results/judgments/<run_id>__mistral.jsonl
  - results/runs/<run_id_juez>.json

Corre después de run_longmemeval_graph.py (o cualquier Fase A de LongMemEval).

Uso:
    uv run python scripts/run_longmemeval_judge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.benchmarks.longmemeval import SUBSET_ORACLE
from memory_arena.evaluation.judge import MistralJudge
from memory_arena.evaluation.judgment_runner import run_judgment
from memory_arena.llm.ollama_client import OllamaClient

SUBSET = SUBSET_ORACLE       # debe coincidir con el Fase A que se quiere juzgar
JUDGE_MODEL = "mistral:7b"
JUDGE_NAME = "mistral"

RESPONSES_DIR = REPO_ROOT / "results" / "responses"


def find_latest_jsonl(subset: str) -> Path | None:
    if not RESPONSES_DIR.exists():
        return None
    candidates = [p for p in RESPONSES_DIR.glob("*.jsonl") if subset in p.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    responses_path = find_latest_jsonl(SUBSET)
    if responses_path is None:
        print(f"ERROR: no se encontró ningún JSONL con '{SUBSET}' en {RESPONSES_DIR}")
        print("Corré primero: uv run python scripts/run_longmemeval_graph.py")
        sys.exit(1)

    print(f"JSONL de entrada: {responses_path.name}")

    judge_llm = OllamaClient(model=JUDGE_MODEL)
    judge = MistralJudge(llm=judge_llm)

    stats = run_judgment(
        responses_path=responses_path,
        judge=judge,
        output_dir=REPO_ROOT / "results",
        judge_name=JUDGE_NAME,
    )

    print(f"\nFase B completada.")
    print(f"  Judgment JSONL:   {stats['output_path']}")
    print(f"  Samples juzgados: {stats['total']}")
    print(f"  Overall accuracy: {stats['overall_accuracy']:.3f}")
    print(f"\nAccuracy por tipo de pregunta:")
    for qtype, acc in sorted(stats["by_type"].items()):
        n = stats["counts_by_type"][qtype]
        print(f"  {qtype:<30} {acc:.3f}  (n={n})")


if __name__ == "__main__":
    main()
