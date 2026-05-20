"""
Smoke test de SummarizedMemory sobre MemoryAgentBench / Conflict Resolution.

Uso:
    uv run python scripts/run_summarized_smoke.py --max-samples 1
    uv run python scripts/run_summarized_smoke.py --max-samples 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_CR,
    SUB_CR_FACTCONSOL_SH_6K,
    load_mab,
)
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.summarized import SummarizedMemory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--summarize-every", type=int, default=8)
    parser.add_argument("--keep-recent", type=int, default=3)
    parser.add_argument("--summary-max-tokens", type=int, default=384)
    parser.add_argument("--document-chunk-chars", type=int, default=30000)
    parser.add_argument("--max-document-chunks", type=int, default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    args = parser.parse_args()

    llm = OllamaClient()
    strategy = SummarizedMemory(
        llm=llm,
        summarize_every=args.summarize_every,
        keep_recent=args.keep_recent,
        summary_max_tokens=args.summary_max_tokens,
        document_chunk_chars=args.document_chunk_chars,
        max_document_chunks=args.max_document_chunks,
    )

    print("SummarizedMemory smoke test")
    print(f"split: {SPLIT_CR}")
    print(f"sub_dataset: {SUB_CR_FACTCONSOL_SH_6K}")
    print(f"max_samples: {args.max_samples}")
    print(f"llm model: {llm.model}")
    print(f"summarize_every: {args.summarize_every}")
    print(f"keep_recent: {args.keep_recent}")
    print(f"summary_max_tokens: {args.summary_max_tokens}")
    print(f"document_chunk_chars: {args.document_chunk_chars}")
    print(f"max_document_chunks: {args.max_document_chunks}")
    print(f"answer_max_tokens: {args.answer_max_tokens}")

    samples = load_mab(
        split=SPLIT_CR,
        sub_dataset=SUB_CR_FACTCONSOL_SH_6K,
        max_samples=args.max_samples,
    )
    total_questions = sum(len(sample.questions) for sample in samples)
    print(f"samples cargados: {len(samples)}")
    print(f"preguntas totales: {total_questions}")

    metadata = run_strategy_mab(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name="summarized",
        split=SPLIT_CR,
        sub_dataset=SUB_CR_FACTCONSOL_SH_6K,
        output_dir=Path("results"),
        max_new_tokens=args.answer_max_tokens,
    )

    print("\nSmoke terminado.")
    print(f"run_id: {metadata.run_id}")
    print(f"responses: results/responses/{metadata.run_id}.jsonl")
    print(f"metadata: results/runs/{metadata.run_id}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
