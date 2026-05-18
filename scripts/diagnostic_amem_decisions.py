"""
Diagnostico A-MEM: distribucion de decisiones evolutivas.

Corre la ingestion de UN sample real de detective_qa (truncado a ~8k chars
para que tarde 3-8 min en una RTX 3060), captura el log verbose, y reporta:
  - cuantas notas se crearon
  - distribucion de decisiones (NO_EVOLUTION / STRENGTHEN / UPDATE_NEIGHBOR
    / STRENGTHEN_AND_UPDATE)
  - LLM calls promedio por nota

Salida en stdout. NO toca results/ ni nada del repo. Read-only sobre el dataset.

Uso:
  uv run python scripts/diagnostic_amem_decisions.py
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_LRU,
    SUB_LRU_DETECTIVE_QA,
    load_mab,
)
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.a_mem import AgenticMemory
from memory_arena.memories.base import Turn

MAX_CONTEXT_CHARS = 8_000


def main() -> None:
    print("=" * 70)
    print("DIAGNOSTICO A-MEM: distribucion de decisiones evolutivas")
    print("=" * 70)

    print("\n[1/4] Cargando detective_qa sample 0...")
    samples = load_mab(SPLIT_LRU, SUB_LRU_DETECTIVE_QA)
    sample = samples[0]
    full_len = len(sample.context)
    truncated = sample.context[:MAX_CONTEXT_CHARS]
    print(f"  sample_id      : {sample.sample_id}")
    print(f"  context full   : {full_len} chars")
    print(f"  context usado  : {len(truncated)} chars (truncado para diagnostico)")

    print("\n[2/4] Inicializando A-MEM con verbose=True...")
    llm = OllamaClient()
    print(f"  LLM model      : {llm.model}")
    print(f"  num_ctx        : {llm.num_ctx}")
    strategy = AgenticMemory(
        llm=llm,
        chunk_size=1500,
        chunk_overlap=100,
        verbose=True,
    )

    print("\n[3/4] Ejecutando ingestion (esto puede tardar 3-8 min)...")
    t0 = time.perf_counter()

    buf = StringIO()
    with redirect_stdout(buf):
        strategy.store(
            Turn(
                role="document",
                content=truncated,
                session_id=sample.sample_id,
                date=None,
            )
        )

    elapsed = time.perf_counter() - t0

    log = buf.getvalue()
    sys.stdout.write(log)
    sys.stdout.flush()

    print("\n[4/4] Parseando decisiones...")

    notes_created = len(strategy._notes)
    decisions = re.findall(r"-> decision: (\w+)", log)
    counts = Counter(decisions)
    total_dec = sum(counts.values())

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Tiempo ingestion       : {elapsed/60:.2f} min")
    print(f"Notas creadas          : {notes_created}")
    print(f"Decisiones registradas : {total_dec}  (la primer nota no evoluciona)")

    if total_dec == 0:
        print("\n  (sin decisiones - texto muy corto o un solo chunk)")
        return

    print("\nDistribucion:")
    for k, v in counts.most_common():
        pct = 100 * v / total_dec
        print(f"  {k:<28} {v:>4}  ({pct:5.1f}%)")

    strengthen = counts.get("STRENGTHEN", 0)
    update = counts.get("UPDATE_NEIGHBOR", 0)
    both = counts.get("STRENGTHEN_AND_UPDATE", 0)

    extra_calls = strengthen + update + 2 * both
    avg_extra = extra_calls / total_dec
    avg_total = 2 + avg_extra

    print(f"\nLLM calls por nota (con vecinos):")
    print(f"  baseline (analyze + evolution_decision): 2")
    print(f"  extra promedio                         : {avg_extra:.2f}")
    print(f"  total promedio por nota                : {avg_total:.2f}")

    if full_len > len(truncated):
        ratio = full_len / len(truncated)
        proj_notes = int(notes_created * ratio)
        proj_calls = int(proj_notes * avg_total)
        print(f"\nProyeccion al sample completo ({full_len} chars):")
        print(f"  notas estimadas        : ~{proj_notes}")
        print(f"  LLM calls de ingestion : ~{proj_calls}")


if __name__ == "__main__":
    main()
