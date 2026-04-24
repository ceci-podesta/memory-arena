"""
D.LRU — Corrida NoMemoria sobre MemoryAgentBench / Long_Range_Understanding.

Itera 2 sub_datasets:
  - detective_qa (10 samples): QA narrativo sobre novelas detectivescas.
  - infbench_sum_eng_shots2 (100 samples): resumen de documentos largos.

Ambos son tareas generativas de respuesta larga. Las 7 metricas default
(EM, substring_em, F1, ROUGE*) son una senal tibia pero no la medida
correcta de calidad — capturan overlap lexico, no fidelidad semantica.

La evaluacion final de D.LRU va a venir del juez LLM Mistral (Bloque E).
Corremos la Fase A ahora igual porque los JSONL son independientes del scorer:
cuando tengamos el juez, rescoreamos los mismos JSONL sin re-generar.
"""
from __future__ import annotations

from pathlib import Path

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_LRU,
    SUB_LRU_DETECTIVE_QA,
    SUB_LRU_INFBENCH_SUM,
    load_mab,
)
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.no_memory import NoMemoria


SUB_DATASETS = [
    SUB_LRU_DETECTIVE_QA,
    SUB_LRU_INFBENCH_SUM,
]


def main() -> None:
    llm = OllamaClient()
    strategy = NoMemoria()

    print(f"LLM model: {llm.model}")
    print(f"num_ctx: {llm.num_ctx}")
    print(f"max_new_tokens: {llm.max_new_tokens}")
    print(f"seed: {llm.seed}")
    print(f"Sub-datasets a correr: {len(SUB_DATASETS)}")

    run_ids: dict[str, str] = {}
    for sub in SUB_DATASETS:
        print(f"\n=== {sub} ===", flush=True)
        samples = load_mab(SPLIT_LRU, sub)
        total_q = sum(len(s.questions) for s in samples)
        print(f"  samples: {len(samples)}  |  total_questions: {total_q}", flush=True)

        meta = run_strategy_mab(
            strategy=strategy,
            samples=samples,
            llm=llm,
            strategy_name="no_memoria",
            split=SPLIT_LRU,
            sub_dataset=sub,
            output_dir=Path("results"),
        )
        run_ids[sub] = meta.run_id

    print("\n\n=== RESUMEN D.LRU ===")
    for sub, rid in run_ids.items():
        print(f"  {sub}: {rid}")


if __name__ == "__main__":
    main()
