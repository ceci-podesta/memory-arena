"""
D.CR completo — NoMemoria sobre los 8 sub-datasets de Conflict Resolution.
Genera 8 JSONL en results/responses/ y 8 metadata en results/runs/.
Al final imprime el resumen de run_ids y duraciones para scorear.
"""
import time
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
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.no_memory import NoMemoria

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
    strategy = NoMemoria()
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
                strategy_name="no_memoria",
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
    print(f"\n=== TOTAL: {total / 60:.1f} min ===", flush=True)
    print("\nResumen:")
    for sub, run_id, elapsed in run_summary:
        print(f"  {sub}: {run_id} ({elapsed / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
