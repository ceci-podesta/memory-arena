"""
D.TTL recsys — SummarizedMemory sobre MemoryAgentBench / recsys_redial_full.

Este script completa TTL para SummarizedMemory: `run_d_ttl_summarized.py`
cubre los 5 sub-datasets ICL, y este archivo cubre `recsys_redial_full`, que
se scorea aparte con `scripts/score_d_ttl_recsys.py`.

Uso:
    uv run python scripts/run_d_ttl_recsys_summarized.py --max-samples 1
    uv run python scripts/run_d_ttl_recsys_summarized.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_TTL,
    SUB_TTL_RECSYS_REDIAL,
    load_mab,
)
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.summarized import SummarizedMemory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--summarize-every", type=int, default=100)
    parser.add_argument("--keep-recent", type=int, default=3)
    parser.add_argument("--summary-max-tokens", type=int, default=512)
    parser.add_argument("--document-chunk-chars", type=int, default=30000)
    parser.add_argument("--max-document-chunks", type=int, default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
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

    print("=" * 72)
    print(f"D.TTL recsys — SummarizedMemory sobre {SUB_TTL_RECSYS_REDIAL}")
    print(f"split: {SPLIT_TTL}")
    print(f"max_samples: {args.max_samples}")
    print(f"llm model: {llm.model}")
    print(f"summarize_every: {args.summarize_every}")
    print(f"keep_recent: {args.keep_recent}")
    print(f"summary_max_tokens: {args.summary_max_tokens}")
    print(f"document_chunk_chars: {args.document_chunk_chars}")
    print(f"max_document_chunks: {args.max_document_chunks}")
    print(f"answer_max_tokens: {args.answer_max_tokens}")
    print("=" * 72)

    samples = load_mab(
        split=SPLIT_TTL,
        sub_dataset=SUB_TTL_RECSYS_REDIAL,
        max_samples=args.max_samples,
    )
    total_questions = sum(len(sample.questions) for sample in samples)
    print(f"samples cargados: {len(samples)}")
    print(f"preguntas totales: {total_questions}")

    if samples:
        first = samples[0]
        print(
            f"\nSample #0: sample_id={first.sample_id} "
            f"n_questions={len(first.questions)}"
        )
        print(f"gold_answers[0]: {first.answers[0]}")
        print(f"question[0] primeros 160 chars: {first.questions[0][:160]!r}")

    metadata = run_strategy_mab(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name="summarized",
        split=SPLIT_TTL,
        sub_dataset=SUB_TTL_RECSYS_REDIAL,
        output_dir=Path("results"),
        max_new_tokens=args.answer_max_tokens,
    )

    print("\nRun completado.")
    print(f"run_id: {metadata.run_id}")
    print(f"duracion: {metadata.duration_seconds}s")
    print(f"metadata: results/runs/{metadata.run_id}.json")
    print(f"responses: results/responses/{metadata.run_id}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
