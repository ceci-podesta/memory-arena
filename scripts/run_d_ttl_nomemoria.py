"""
D.TTL — Corrida NoMemoria sobre MemoryAgentBench / Test_Time_Learning.

Itera 5 sub_datasets ICL (Banking77, Clinic150, NLU, TREC coarse, TREC fine).
El sub_dataset recsys_redial_full se deja afuera: requiere Bloque F
(entity2id + fuzzy matching) para scoring correcto — default metrics
no son aplicables al output estructurado de recsys.

Persiste un JSONL por sub_dataset en results/responses/ y metadata
por corrida en results/runs/.
"""
from __future__ import annotations

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
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.no_memory import NoMemoria


SUB_DATASETS = [
    SUB_TTL_ICL_BANKING77,
    SUB_TTL_ICL_CLINIC150,
    SUB_TTL_ICL_NLU,
    SUB_TTL_ICL_TREC_COARSE,
    SUB_TTL_ICL_TREC_FINE,
]


def main() -> None:
    llm = OllamaClient()
    strategy = NoMemoria()

    print(f"LLM model: {llm.model}")
    print(f"num_ctx: {llm.num_ctx}")
    print(f"max_new_tokens: {llm.max_new_tokens}")
    print(f"seed: {llm.seed}")
    print(f"Sub-datasets a correr: {len(SUB_DATASETS)} (recsys_redial_full omitido)")

    run_ids: dict[str, str] = {}
    for sub in SUB_DATASETS:
        print(f"\n=== {sub} ===", flush=True)
        samples = load_mab(SPLIT_TTL, sub)
        total_q = sum(len(s.questions) for s in samples)
        print(f"  samples: {len(samples)}  |  total_questions: {total_q}", flush=True)

        meta = run_strategy_mab(
            strategy=strategy,
            samples=samples,
            llm=llm,
            strategy_name="no_memoria",
            split=SPLIT_TTL,
            sub_dataset=sub,
            output_dir=Path("results"),
        )
        run_ids[sub] = meta.run_id

    print("\n\n=== RESUMEN D.TTL ===")
    for sub, rid in run_ids.items():
        print(f"  {sub}: {rid}")


if __name__ == "__main__":
    main()
