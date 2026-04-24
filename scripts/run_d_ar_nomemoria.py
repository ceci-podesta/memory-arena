"""
D.AR — Corrida NoMemoria sobre MemoryAgentBench / Accurate_Retrieval.

Itera todos los sub_datasets de AR (6 sub-datasets, 22 samples totales) y
persiste un JSONL por cada uno en results/responses/, más su metadata
correspondiente en results/runs/.

Pattern mirror de scripts/run_d_cr_nomemoria.py: no devuelve nada,
sólo imprime un resumen de run_ids al final para que después score_d_ar.py
los levante.

Nota sobre context sizes:
  - longmemeval_s*: contextos chat-style moderados (5 samples).
  - eventqa_full/65K/131K: contextos largos sobre eventos.
  - ruler_qa1_197K / ruler_qa2_421K: contextos MUY largos (RULER).

NoMemoria ignora el contexto (piso honesto), así que el tamaño no afecta
la calidad — sólo el tiempo de carga del dataset desde HF.
"""
from __future__ import annotations

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
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.no_memory import NoMemoria


SUB_DATASETS = [
    SUB_AR_LONGMEMEVAL_S,
    SUB_AR_EVENTQA_FULL,
    SUB_AR_EVENTQA_65K,
    SUB_AR_EVENTQA_131K,
    SUB_AR_RULER_QA1,
    SUB_AR_RULER_QA2,
]


def main() -> None:
    llm = OllamaClient()  # defaults desde experimental_config
    strategy = NoMemoria()

    print(f"LLM model: {llm.model}")
    print(f"num_ctx: {llm.num_ctx}")
    print(f"max_new_tokens: {llm.max_new_tokens}")
    print(f"seed: {llm.seed}")
    print(f"Sub-datasets a correr: {len(SUB_DATASETS)}")

    run_ids: dict[str, str] = {}
    for sub in SUB_DATASETS:
        print(f"\n=== {sub} ===", flush=True)
        samples = load_mab(SPLIT_AR, sub)
        total_q = sum(len(s.questions) for s in samples)
        print(f"  samples: {len(samples)}  |  total_questions: {total_q}", flush=True)

        meta = run_strategy_mab(
            strategy=strategy,
            samples=samples,
            llm=llm,
            strategy_name="no_memoria",
            split=SPLIT_AR,
            sub_dataset=sub,
            output_dir=Path("results"),
        )
        run_ids[sub] = meta.run_id

    print("\n\n=== RESUMEN D.AR ===")
    for sub, rid in run_ids.items():
        print(f"  {sub}: {rid}")


if __name__ == "__main__":
    main()
