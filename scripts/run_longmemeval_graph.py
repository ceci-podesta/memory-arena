"""
LongMemEval — Fase A: generación de respuestas con GraphMemory.

Corre GraphMemory sobre el subset `longmemeval_oracle` (solo la sesión con
evidencia, ~500 samples). Es el subset que usó NoMemoria como baseline.

Para comparar contra `longmemeval_s_cleaned` (40 sesiones) descomentar la
constante correspondiente — tarda considerablemente más.

Uso:
    uv run python scripts/run_longmemeval_graph.py

Salida:
    results/responses/<run_id>.jsonl   — respuestas (Fase A)
    results/runs/<run_id>.json         — metadata de la corrida
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.benchmarks.longmemeval import (
    SUBSET_ORACLE,
    # SUBSET_S_CLEANED,  # descomentar para el haystack de 40 sesiones
    load_longmemeval,
)
from memory_arena.evaluation.runner import run_strategy
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.graph_memory import GraphMemory

SUBSET = SUBSET_ORACLE
MAX_SAMPLES: int | None = None  # None = dataset completo; 1-5 para smoke test


def main() -> None:
    llm = OllamaClient()
    strategy = GraphMemory(llm=llm)

    print(f"LLM model:      {llm.model}")
    print(f"num_ctx:        {llm.num_ctx}")
    print(f"seed:           {llm.seed}")
    print(f"Subset:         {SUBSET}")
    print(f"MAX_SAMPLES:    {MAX_SAMPLES or 'completo'}")

    samples = load_longmemeval(SUBSET, limit=MAX_SAMPLES)
    print(f"Samples cargados: {len(samples)}", flush=True)

    meta = run_strategy(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name="graph_memory",
        benchmark_name=SUBSET,
        output_dir=REPO_ROOT / "results",
    )

    print(f"\nFase A completada.")
    print(f"  run_id:  {meta.run_id}")
    print(f"  JSONL:   results/responses/{meta.run_id}.jsonl")
    print(f"\nSiguiente paso (juez):")
    print(f"  uv run python scripts/run_longmemeval_judge.py")


if __name__ == "__main__":
    main()
