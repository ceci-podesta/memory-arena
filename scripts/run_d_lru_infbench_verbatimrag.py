"""
D.LRU infbench_sum — VerbatimRAG sobre infbench_sum_eng_shots2 solamente.
Usa este script cuando detective_qa ya está corrido y solo falta infbench_sum.
"""
from __future__ import annotations

from pathlib import Path

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_LRU,
    SUB_LRU_INFBENCH_SUM,
    load_mab,
)
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.verbatim_rag import VerbatimRAG


def main() -> None:
    llm = OllamaClient()
    strategy = VerbatimRAG()

    print(f"LLM model: {llm.model}")
    print(f"num_ctx: {llm.num_ctx}")
    print(f"max_new_tokens: {llm.max_new_tokens}")
    print(f"seed: {llm.seed}")

    print(f"\n=== {SUB_LRU_INFBENCH_SUM} ===", flush=True)
    samples = load_mab(SPLIT_LRU, SUB_LRU_INFBENCH_SUM)
    total_q = sum(len(s.questions) for s in samples)
    print(f"  samples: {len(samples)}  |  total_questions: {total_q}", flush=True)

    meta = run_strategy_mab(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name="verbatim_rag",
        split=SPLIT_LRU,
        sub_dataset=SUB_LRU_INFBENCH_SUM,
        output_dir=Path("results"),
    )

    print(f"\nrun_id: {meta.run_id}", flush=True)
    print(f"Siguiente paso: uv run python scripts/run_e_lru_judge.py", flush=True)


if __name__ == "__main__":
    main()
