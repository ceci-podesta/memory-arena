"""
Versión resumable de run_longmemeval.py.

Uso:
    # Arrancar fresh (igual que el original):
    uv run python scripts/run_longmemeval_resumable.py --strategy a_mem --subset s

    # Reanudar una corrida interrumpida:
    uv run python scripts/run_longmemeval_resumable.py \\
        --strategy a_mem --subset s \\
        --resume results/responses/<RUN_ID>.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.benchmarks.longmemeval import (
    SUBSET_M_CLEANED,
    SUBSET_ORACLE,
    SUBSET_S_CLEANED,
    load_longmemeval,
)
from memory_arena.evaluation.judge import MistralJudge
from memory_arena.evaluation.judgment_runner import run_judgment
from memory_arena.evaluation.runner_resumable import run_strategy_resumable
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.base import MemoriaBase
from memory_arena.memories.no_memory import NoMemoria


SUBSET_ALIASES = {
    "oracle": SUBSET_ORACLE,
    "s": SUBSET_S_CLEANED,
    "m": SUBSET_M_CLEANED,
}


def build_strategy(name: str) -> MemoriaBase:
    if name == "no_memoria":
        return NoMemoria()
    if name == "a_mem":
        from memory_arena.memories.a_mem import AgenticMemory
        return AgenticMemory()
    raise SystemExit(f"Strategy desconocida: {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="a_mem")
    parser.add_argument("--subset", default="s", choices=list(SUBSET_ALIASES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path al JSONL existente para reanudar",
    )
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    subset_real = SUBSET_ALIASES[args.subset]
    print(
        f"=== LongMemEval RESUMABLE — strategy={args.strategy}  "
        f"subset={subset_real} ==="
    )

    samples = load_longmemeval(subset_real, limit=args.limit)
    print(f"  {len(samples)} samples cargados")

    strategy = build_strategy(args.strategy)
    llm = OllamaClient()
    print(f"  modelo evaluado: {llm.model}")
    if args.resume:
        print(f"  reanudando desde: {args.resume}")

    print("\n--- Fase A (generación) ---")
    metadata = run_strategy_resumable(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name=args.strategy,
        benchmark_name=subset_real,
        output_dir=REPO_ROOT / "results",
        resume_path=args.resume,
    )
    print(f"  run_id: {metadata.run_id}")
    print(f"  duración: {metadata.duration_seconds}s")

    responses_path = (
        args.resume
        if args.resume
        else REPO_ROOT / "results" / "responses" / f"{metadata.run_id}.jsonl"
    )
    print(f"  responses: {responses_path}")

    if args.skip_judge:
        print("\n(saltando juez por --skip-judge)")
        return 0

    print("\n--- Fase B (juez Mistral) ---")
    judge = MistralJudge()
    result = run_judgment(
        responses_path=responses_path,
        judge=judge,
        output_dir=REPO_ROOT / "results",
        judge_name="mistral",
    )
    print(f"  total juzgados: {result['total']}")
    print(f"  overall_accuracy: {result['overall_accuracy']}")
    print("  por question_type:")
    for qt, acc in sorted(result["by_type"].items()):
        n = result["counts_by_type"][qt]
        print(f"    {qt:<32} {acc:.4f} (n={n})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
