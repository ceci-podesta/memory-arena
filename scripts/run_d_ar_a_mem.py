"""
D.AR — Corrida A-MEM sobre MemoryAgentBench / Accurate_Retrieval.

Itera todos los sub_datasets de AR (6 sub-datasets, 22 samples totales) y
persiste un JSONL por cada uno en results/responses/, mas su metadata
en results/runs/.

A-MEM chunkea cada contexto en notas atomicas (chunk_size=1500 chars con
overlap 100), genera K/G/X via LLM, top-k cosine para link generation y
memory evolution. Costo computacional alto en sub-datasets con contextos
grandes (ruler_qa1_197K, ruler_qa2_421K, eventqa_131K).
"""
from __future__ import annotations

import time
from pathlib import Path

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_AR,
    SUB_AR_EVENTQA_131K,
    SUB_AR_EVENTQA_65K,
    SUB_AR_EVENTQA_FULL,
    SUB_AR_LONGMEMEVAL_S,
    SUB_AR_RULER_QA1,
    SUB_AR_RULER_QA2,
    load_mab,
)
from memory_arena.evaluation.mab_eta_helper import run_strategy_mab_with_eta as run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.a_mem import AgenticMemory


SUB_DATASETS = [
    SUB_AR_LONGMEMEVAL_S,
    SUB_AR_EVENTQA_FULL,
    SUB_AR_EVENTQA_65K,
    SUB_AR_EVENTQA_131K,
    SUB_AR_RULER_QA1,
    SUB_AR_RULER_QA2,
]


def main() -> None:
    llm = OllamaClient()
    strategy = AgenticMemory(llm=llm, chunk_size=1500, chunk_overlap=100)

    print(f"LLM model: {llm.model}")
    print(f"num_ctx: {llm.num_ctx}")
    print(f"chunk_size: {strategy._chunk_size}")
    print(f"chunk_overlap: {strategy._chunk_overlap}")
    print(f"Sub-datasets a correr: {len(SUB_DATASETS)}")

    run_ids: dict[str, str] = {}
    t_all = time.perf_counter()
    for sub in SUB_DATASETS:
        print(f"\n=== {sub} ===", flush=True)
        try:
            samples = load_mab(SPLIT_AR, sub)
            total_q = sum(len(s.questions) for s in samples)
            print(f"  samples: {len(samples)}  |  total_questions: {total_q}", flush=True)
            t0 = time.perf_counter()
            meta = run_strategy_mab(
                strategy=strategy,
                samples=samples,
                llm=llm,
                strategy_name="a_mem",
                split=SPLIT_AR,
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
    print(f"\n=== TOTAL D.AR: {total / 60:.1f} min ===", flush=True)
    print("\nResumen:")
    for sub, rid in run_ids.items():
        print(f"  {sub}: {rid}")


if __name__ == "__main__":
    main()
