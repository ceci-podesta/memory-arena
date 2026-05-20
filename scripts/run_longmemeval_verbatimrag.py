"""
LongMemEval oracle — VerbatimRAG.
Genera un JSONL en results/responses/ y metadata en results/runs/.

Para smoke test: poné MAX_SAMPLES = 5.
Para corrida real: MAX_SAMPLES = None (500 samples).
"""
import time

from memory_arena.benchmarks.longmemeval import SUBSET_ORACLE, load_longmemeval
from memory_arena.evaluation.runner import run_strategy
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.verbatim_rag import VerbatimRAG

MAX_SAMPLES = None # cambiar a None para corrida real completa


def main() -> None:
    llm = OllamaClient(model="llama3.2:3b")
    strategy = VerbatimRAG()

    print(f"Cargando LongMemEval {SUBSET_ORACLE}...", flush=True)
    samples = load_longmemeval(SUBSET_ORACLE, limit=MAX_SAMPLES)
    print(f"  {len(samples)} sample(s) cargados", flush=True)

    t0 = time.perf_counter()
    meta = run_strategy(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name="verbatim_rag",
        benchmark_name=SUBSET_ORACLE,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nrun_id: {meta.run_id}", flush=True)
    print(f"duracion: {elapsed / 60:.1f} min", flush=True)
    print(f"Para scorear: uv run python scripts/score_longmemeval.py {meta.run_id}", flush=True)


if __name__ == "__main__":
    main()
