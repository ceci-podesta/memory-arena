"""
Corre juez Mistral sobre un JSONL de respuestas de LongMemEval.

Uso:
    uv run python scripts/run_longmemeval_judge.py --responses results/responses/<run_id>.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_arena.evaluation.judge import MistralJudge
from memory_arena.evaluation.judgment_runner import run_judgment
from memory_arena.llm.ollama_client import OllamaClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--judge-name", default="mistral")
    parser.add_argument("--judge-model", default="mistral:7b")
    args = parser.parse_args()

    if not args.responses.exists():
        raise SystemExit(f"No existe el archivo de responses: {args.responses}")

    judge_llm = OllamaClient(model=args.judge_model)
    judge = MistralJudge(llm=judge_llm)

    print("LongMemEval judgment run")
    print(f"responses: {args.responses}")
    print(f"judge_name: {args.judge_name}")
    print(f"judge_model: {args.judge_model}")

    stats = run_judgment(
        responses_path=args.responses,
        judge=judge,
        output_dir=Path("results"),
        judge_name=args.judge_name,
    )

    print("\nJudgment terminado.")
    print(f"total: {stats['total']}")
    print(f"overall_accuracy: {stats['overall_accuracy']}")
    print(f"avg_latency_per_item_s: {stats['avg_latency_per_item_s']}")
    print(f"output_path: {stats['output_path']}")
    print(f"run_metadata_path: {stats['run_metadata_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
