"""
D.CR — SummarizedMemory sobre MemoryAgentBench / Conflict Resolution.

Por default corre los 8 sub-datasets de CR. Para desarrollo, usar limites:

    uv run python scripts/run_d_cr_summarized.py --max-subdatasets 1 --max-samples 1
    uv run python scripts/run_d_cr_summarized.py --max-subdatasets 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_CR,
    SUB_CR_FACTCONSOL_MH_6K,
    SUB_CR_FACTCONSOL_MH_32K,
    SUB_CR_FACTCONSOL_MH_64K,
    SUB_CR_FACTCONSOL_MH_262K,
    SUB_CR_FACTCONSOL_SH_6K,
    SUB_CR_FACTCONSOL_SH_32K,
    SUB_CR_FACTCONSOL_SH_64K,
    SUB_CR_FACTCONSOL_SH_262K,
    load_mab,
)
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.summarized import SummarizedMemory


SUB_DATASETS = [
    SUB_CR_FACTCONSOL_SH_6K,
    SUB_CR_FACTCONSOL_SH_32K,
    SUB_CR_FACTCONSOL_SH_64K,
    SUB_CR_FACTCONSOL_SH_262K,
    SUB_CR_FACTCONSOL_MH_6K,
    SUB_CR_FACTCONSOL_MH_32K,
    SUB_CR_FACTCONSOL_MH_64K,
    SUB_CR_FACTCONSOL_MH_262K,
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-subdatasets", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--summarize-every", type=int, default=100)
    parser.add_argument("--keep-recent", type=int, default=3)
    parser.add_argument("--summary-max-tokens", type=int, default=512)
    parser.add_argument("--document-chunk-chars", type=int, default=30000)
    parser.add_argument("--max-document-chunks", type=int, default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    llm = OllamaClient(model="llama3.2:3b")
    strategy = SummarizedMemory(
        llm=llm,
        summarize_every=args.summarize_every,
        keep_recent=args.keep_recent,
        summary_max_tokens=args.summary_max_tokens,
        document_chunk_chars=args.document_chunk_chars,
        max_document_chunks=args.max_document_chunks,
    )

    sub_datasets = SUB_DATASETS
    if args.max_subdatasets is not None:
        sub_datasets = sub_datasets[: args.max_subdatasets]

    print("D.CR — SummarizedMemory")
    print(f"subdatasets a correr: {len(sub_datasets)}")
    print(f"max_samples: {args.max_samples}")
    print(f"llm model: {llm.model}")
    print(f"summarize_every: {args.summarize_every}")
    print(f"keep_recent: {args.keep_recent}")
    print(f"summary_max_tokens: {args.summary_max_tokens}")
    print(f"document_chunk_chars: {args.document_chunk_chars}")
    print(f"max_document_chunks: {args.max_document_chunks}")
    print(f"answer_max_tokens: {args.answer_max_tokens}")

    run_summary: list[tuple[str, str, float]] = []
    t_all = time.perf_counter()

    for sub in sub_datasets:
        print(f"\n=== {sub} ===", flush=True)
        try:
            samples = load_mab(SPLIT_CR, sub, max_samples=args.max_samples)
            total_questions = sum(len(sample.questions) for sample in samples)
            print(
                f"  samples: {len(samples)} | preguntas: {total_questions}",
                flush=True,
            )

            t0 = time.perf_counter()
            meta = run_strategy_mab(
                strategy=strategy,
                samples=samples,
                llm=llm,
                strategy_name="summarized",
                split=SPLIT_CR,
                sub_dataset=sub,
                output_dir=Path("results"),
                max_new_tokens=args.answer_max_tokens,
            )
            elapsed = time.perf_counter() - t0
            print(f"  duracion: {elapsed / 60:.1f} min", flush=True)
            run_summary.append((sub, meta.run_id, elapsed))
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            run_summary.append((sub, "FAILED", 0.0))

    total = time.perf_counter() - t_all
    print(f"\n=== TOTAL: {total / 60:.1f} min ===", flush=True)
    print("\nResumen:")
    for sub, run_id, elapsed in run_summary:
        print(f"  {sub}: {run_id} ({elapsed / 60:.1f} min)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
