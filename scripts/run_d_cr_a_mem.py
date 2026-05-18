"""
D.CR — Corrida A-MEM sobre MemoryAgentBench / Conflict_Resolution.

Itera 8 sub-datasets factconsolidation (single-hop y multi-hop, contextos
de 6k a 262k chars). A-MEM chunkea con chunk_size=1500.
"""
from __future__ import annotations

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
from memory_arena.evaluation.mab_eta_helper import run_strategy_mab_with_eta as run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.a_mem import AgenticMemory


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
    llm = OllamaClient(model="llama3.2:3b")
    strategy = AgenticMemory(llm=llm, chunk_size=1500, chunk_overlap=100)

    print(f"chunk_size: {strategy._chunk_size}")
    print(f"chunk_overlap: {strategy._chunk_overlap}")

    run_summary: list[tuple[str, str, float]] = []
    t_all = time.perf_counter()
    for sub in SUB_DATASETS:
        print(f"\n=== {sub} ===", flush=True)
        try:
            samples = load_mab(SPLIT_CR, sub)
            print(f"  {len(samples)} sample(s)", flush=True)
            t0 = time.perf_counter()
            meta = run_strategy_mab(
                strategy=strategy,
                samples=samples,
                llm=llm,
                strategy_name="a_mem",
                split=SPLIT_CR,
                sub_dataset=sub,
            )
            elapsed = time.perf_counter() - t0
            print(f"  duracion: {elapsed / 60:.1f} min", flush=True)
            run_summary.append((sub, meta.run_id, elapsed))
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            run_summary.append((sub, "FAILED", 0.0))

    total = time.perf_counter() - t_all
    print(f"\n=== TOTAL D.CR: {total / 60:.1f} min ===", flush=True)
    print("\nResumen:")
    for sub, run_id, elapsed in run_summary:
        print(f"  {sub}: {run_id} ({elapsed / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
