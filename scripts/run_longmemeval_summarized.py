"""
Corre SummarizedMemory sobre LongMemEval.

Uso:
    uv run python scripts/run_longmemeval_summarized.py --subset oracle --limit 5
    uv run python scripts/run_longmemeval_summarized.py --subset s_cleaned --limit 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_arena.benchmarks.longmemeval import (
    SUBSET_M_CLEANED,
    SUBSET_ORACLE,
    SUBSET_S_CLEANED,
    load_longmemeval,
)
from memory_arena.evaluation.runner import run_strategy
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.summarized import SummarizedMemory


SUBSET_ALIASES = {
    "oracle": SUBSET_ORACLE,
    "s": SUBSET_S_CLEANED,
    "s_cleaned": SUBSET_S_CLEANED,
    "m": SUBSET_M_CLEANED,
    "m_cleaned": SUBSET_M_CLEANED,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=sorted(SUBSET_ALIASES.keys()),
        default="oracle",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--summarize-every", type=int, default=8)
    parser.add_argument("--keep-recent", type=int, default=3)
    parser.add_argument("--summary-max-tokens", type=int, default=1024)
    parser.add_argument("--document-chunk-chars", type=int, default=30000)
    parser.add_argument("--max-document-chunks", type=int, default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    args = parser.parse_args()

    subset_real = SUBSET_ALIASES[args.subset]

    llm = OllamaClient()
    strategy = SummarizedMemory(
        llm=llm,
        summarize_every=args.summarize_every,
        keep_recent=args.keep_recent,
        summary_max_tokens=args.summary_max_tokens,
        document_chunk_chars=args.document_chunk_chars,
        max_document_chunks=args.max_document_chunks,
    )

    print("SummarizedMemory LongMemEval run")
    print(f"subset: {subset_real}")
    print(f"limit: {args.limit}")
    print(f"llm model: {llm.model}")
    print(f"summarize_every: {args.summarize_every}")
    print(f"keep_recent: {args.keep_recent}")
    print(f"summary_max_tokens: {args.summary_max_tokens}")
    print(f"document_chunk_chars: {args.document_chunk_chars}")
    print(f"max_document_chunks: {args.max_document_chunks}")
    print(f"answer_max_tokens: {args.answer_max_tokens}")

    samples = load_longmemeval(subset_real, limit=args.limit)
    print(f"samples cargados: {len(samples)}")

    metadata = run_strategy(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name="summarized",
        benchmark_name=subset_real,
        output_dir=Path("results"),
        max_new_tokens=args.answer_max_tokens,
    )

    print("\nCorrida terminada.")
    print(f"run_id: {metadata.run_id}")
    print(f"responses: results/responses/{metadata.run_id}.jsonl")
    print(f"metadata: results/runs/{metadata.run_id}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
