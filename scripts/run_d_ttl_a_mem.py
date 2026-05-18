"""
D.TTL — Corrida A-MEM sobre MemoryAgentBench / Test_Time_Learning.

Itera 5 sub-datasets ICL (Banking77, Clinic150, NLU, TREC coarse, TREC fine).
recsys_redial_full se deja afuera (mismo criterio que NoMemoria — requiere
scoring especial via Bloque F).
"""
from __future__ import annotations

import time
from pathlib import Path

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_TTL,
    SUB_TTL_ICL_BANKING77,
    SUB_TTL_ICL_CLINIC150,
    SUB_TTL_ICL_NLU,
    SUB_TTL_ICL_TREC_COARSE,
    SUB_TTL_ICL_TREC_FINE,
    load_mab,
)
from memory_arena.evaluation.mab_eta_helper import run_strategy_mab_with_eta as run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.a_mem import AgenticMemory


SUB_DATASETS = [
    SUB_TTL_ICL_BANKING77,
    SUB_TTL_ICL_CLINIC150,
    SUB_TTL_ICL_NLU,
    SUB_TTL_ICL_TREC_COARSE,
    SUB_TTL_ICL_TREC_FINE,
]


def main() -> None:
    llm = OllamaClient()
    strategy = AgenticMemory(llm=llm, chunk_size=1500, chunk_overlap=100)

    print(f"LLM model: {llm.model}")
    print(f"chunk_size: {strategy._chunk_size}")
    print(f"Sub-datasets a correr: {len(SUB_DATASETS)} (recsys_redial_full omitido)")

    run_ids: dict[str, str] = {}
    t_all = time.perf_counter()
    for sub in SUB_DATASETS:
        print(f"\n=== {sub} ===", flush=True)
        try:
            samples = load_mab(SPLIT_TTL, sub)
            total_q = sum(len(s.questions) for s in samples)
            print(f"  samples: {len(samples)}  |  total_questions: {total_q}", flush=True)
            t0 = time.perf_counter()
            meta = run_strategy_mab(
                strategy=strategy,
                samples=samples,
                llm=llm,
                strategy_name="a_mem",
                split=SPLIT_TTL,
                sub_dataset=sub,
                output_dir=Path("results"),
            )
            elapsed = time.perf_counter() - t0
            print(f"  duracion: {elapsed / 60:.1f} min", flush=True)
            run_ids[sub] = meta.run_id
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            run_ids[sub] = "FAILED"

    total = time.perf_counter() - t_all
    print(f"\n=== TOTAL D.TTL: {total / 60:.1f} min ===", flush=True)
    print("\n=== RESUMEN D.TTL ===")
    for sub, rid in run_ids.items():
        print(f"  {sub}: {rid}")


if __name__ == "__main__":
    main()
