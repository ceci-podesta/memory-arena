"""
D.CR — Corrida GraphMemory sobre MemoryAgentBench / Conflict_Resolution.
"""
from __future__ import annotations

from pathlib import Path

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_CR,
    SUB_CR_FACTCONSOL_MH_262K,
    SUB_CR_FACTCONSOL_MH_32K,
    SUB_CR_FACTCONSOL_MH_64K,
    SUB_CR_FACTCONSOL_MH_6K,
    SUB_CR_FACTCONSOL_SH_262K,
    SUB_CR_FACTCONSOL_SH_32K,
    SUB_CR_FACTCONSOL_SH_64K,
    SUB_CR_FACTCONSOL_SH_6K,
    load_mab,
)
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.graph_memory import GraphMemory


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


def main() -> None:
    llm = OllamaClient()
    strategy = GraphMemory(llm=llm)

    print(f"LLM model: {llm.model}")
    print(f"num_ctx: {llm.num_ctx}")
    print(f"max_new_tokens: {llm.max_new_tokens}")
    print(f"seed: {llm.seed}")
    print(f"Sub-datasets a correr: {len(SUB_DATASETS)}")

    run_ids: dict[str, str] = {}
    for sub in SUB_DATASETS:
        print(f"\n=== {sub} ===", flush=True)
        samples = load_mab(SPLIT_CR, sub)
        total_q = sum(len(s.questions) for s in samples)
        print(f"  samples: {len(samples)}  |  total_questions: {total_q}", flush=True)

        meta = run_strategy_mab(
            strategy=strategy,
            samples=samples,
            llm=llm,
            strategy_name="graph_memory",
            split=SPLIT_CR,
            sub_dataset=sub,
            output_dir=Path("results"),
        )
        run_ids[sub] = meta.run_id

    print("\n\n=== RESUMEN D.CR ===")
    for sub, rid in run_ids.items():
        print(f"  {sub}: {rid}")


if __name__ == "__main__":
    main()
