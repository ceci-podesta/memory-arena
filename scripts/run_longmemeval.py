"""
Corre LongMemEval end-to-end (Fase A + juez Mistral) para una estrategia.

Uso:
    uv run python scripts/run_longmemeval.py
    uv run python scripts/run_longmemeval.py --strategy graph_memory
    uv run python scripts/run_longmemeval.py --strategy no_memoria --subset s
    uv run python scripts/run_longmemeval.py --limit 5     # smoke test

Acepta:
    --strategy NAME     default: no_memoria (debe matchear el strategy_name que
                        usa el script run_d_* de la estrategia)
    --subset SUBSET     oracle | s | m (default: oracle)
    --limit N           limita a los primeros N samples (default: todos)
    --skip-judge        solo corre Fase A, no corre el juez
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.benchmarks.longmemeval import (
    SUBSET_M_CLEANED,
    SUBSET_ORACLE,
    SUBSET_S_CLEANED,
    load_longmemeval,
)
from memory_arena.evaluation.judge import MistralJudge
from memory_arena.evaluation.judgment_runner import run_judgment
from memory_arena.evaluation.runner import run_strategy
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.base import MemoriaBase
from memory_arena.memories.no_memory import NoMemoria


SUBSET_ALIASES = {
    "oracle": SUBSET_ORACLE,
    "s": SUBSET_S_CLEANED,
    "m": SUBSET_M_CLEANED,
}

# Mapping de strategy_name (CLI) -> clase de estrategia.
# Cuando otros agreguen estrategias, suman su entry acá.
STRATEGY_CLASSES: dict[str, type[MemoriaBase]] = {
    "no_memoria": NoMemoria,
}


def build_strategy(name: str) -> MemoriaBase:
    if name in STRATEGY_CLASSES:
        return STRATEGY_CLASSES[name]()
    # Import dinámico para no crashear si una estrategia no está disponible.
    try:
        if name == "verbatim_rag":
            from memory_arena.memories.verbatim_rag import VerbatimRAG
            return VerbatimRAG()
        if name == "summarized":
            from memory_arena.memories.summarized import SummarizedMemory
            return SummarizedMemory()
        if name == "a_mem":
            from memory_arena.memories.a_mem import AgenticMemory
            return AgenticMemory()
        if name == "graph_memory":
            from memory_arena.memories.graph_memory import GraphMemory
            return GraphMemory()
    except ImportError as e:
        raise SystemExit(
            f"No se pudo importar la estrategia {name!r}: {e}.\n"
            f"Si todavía no la implementaste, usá --strategy no_memoria."
        )
    raise SystemExit(
        f"Strategy desconocida: {name!r}. "
        f"Opciones registradas: {list(STRATEGY_CLASSES)} + "
        f"verbatim_rag/summarized/a_mem/graph_memory (si están implementadas)."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="no_memoria")
    parser.add_argument("--subset", default="oracle",
                        choices=list(SUBSET_ALIASES))
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar a N samples (default: todos)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Solo Fase A; no correr el juez Mistral")
    args = parser.parse_args()

    subset_real = SUBSET_ALIASES[args.subset]
    print(f"=== LongMemEval — strategy={args.strategy}  subset={subset_real} ===")

    print(f"\nCargando samples de {subset_real}...")
    samples = load_longmemeval(subset_real, limit=args.limit)
    print(f"  {len(samples)} samples cargados")

    strategy = build_strategy(args.strategy)
    llm = OllamaClient()
    print(f"  modelo evaluado: {llm.model}")

    print(f"\n--- Fase A (generación) ---")
    metadata = run_strategy(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name=args.strategy,
        benchmark_name=subset_real,
        output_dir=REPO_ROOT / "results",
    )
    print(f"  run_id: {metadata.run_id}")
    print(f"  duración: {metadata.duration_seconds}s")
    responses_path = REPO_ROOT / "results" / "responses" / f"{metadata.run_id}.jsonl"
    print(f"  responses: {responses_path.relative_to(REPO_ROOT)}")

    if args.skip_judge:
        print("\n(saltando juez por --skip-judge)")
        return 0

    print(f"\n--- Fase B (juez Mistral) ---")
    judge = MistralJudge()
    print(f"  juez: {judge.llm.model}")
    result = run_judgment(
        responses_path=responses_path,
        judge=judge,
        output_dir=REPO_ROOT / "results",
        judge_name="mistral",
    )
    print(f"  total juzgados: {result['total']}")
    print(f"  overall_accuracy: {result['overall_accuracy']}")
    print(f"  por question_type:")
    for qt, acc in sorted(result["by_type"].items()):
        n = result["counts_by_type"][qt]
        print(f"    {qt:<32} {acc:.4f} (n={n})")
    print(f"\n  judgments: {result['output_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
