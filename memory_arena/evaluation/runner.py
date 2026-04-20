"""
memory_arena.evaluation.runner
-------------------------------
Fase A del pipeline de evaluación: generación de respuestas.

Para cada sample del benchmark:
  1. reset() de la estrategia.
  2. store() de todos los turnos del haystack.
  3. retrieve() del contexto relevante.
  4. Llamada al LLM con pregunta + contexto.
  5. Persistencia a JSONL (una línea por sample).

La evaluación (Fase B) es un módulo aparte: lee este JSONL y aplica
jueces. Separar las dos fases permite probar distintos jueces sin
re-generar respuestas (que es lo caro).
"""

import json
import time
from pathlib import Path

from memory_arena.benchmarks.longmemeval import LongMemEvalSample
from memory_arena.evaluation.run_metadata import (
    RunMetadata,
    finalize_run,
    start_run,
)
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.base import MemoriaBase, Turn


def run_strategy(
    strategy: MemoriaBase,
    samples: list[LongMemEvalSample],
    llm: OllamaClient,
    strategy_name: str,
    benchmark_name: str,
    output_dir: Path = Path("results"),
    top_k: int = 5,
) -> RunMetadata:
    """Corre una estrategia sobre una lista de samples y persiste resultados.

    Args:
        strategy: Instancia de una estrategia de memoria.
        samples: Lista de samples del benchmark (ya cargados).
        llm: Cliente para el LLM evaluado.
        strategy_name: Nombre corto para el run_id y filenames (ej "no_memory").
        benchmark_name: Ídem (ej "longmemeval_oracle").
        output_dir: Raíz de `results/` (default: ./results).
        top_k: Cuántos fragmentos pedirle a retrieve().

    Returns:
        La metadata de la corrida (con run_id, duración, etc.).
    """
    metadata = start_run(
        strategy=strategy_name,
        benchmark=benchmark_name,
        model=llm.model,
        num_samples=len(samples),
    )

    responses_path = output_dir / "responses" / f"{metadata.run_id}.jsonl"
    responses_path.parent.mkdir(parents=True, exist_ok=True)

    with open(responses_path, "w", encoding="utf-8") as f:
        for sample in samples:
            record = _process_sample(strategy, sample, llm, top_k)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # persistir turno a turno por si se cae la corrida

    runs_path = output_dir / "runs" / f"{metadata.run_id}.json"
    finalize_run(metadata, runs_path)
    return metadata


def _process_sample(
    strategy: MemoriaBase,
    sample: LongMemEvalSample,
    llm: OllamaClient,
    top_k: int,
) -> dict:
    strategy.reset()

    for session in sample.haystack:
        for turn in session.turns:
            enriched = Turn(
                role=turn.role,
                content=turn.content,
                session_id=session.session_id,
                date=session.date,
            )
            strategy.store(enriched)

    context = strategy.retrieve(sample.question, top_k=top_k)
    prompt = _build_prompt(sample.question, sample.question_date, context)

    t0 = time.perf_counter()
    system_answer = llm.chat([{"role": "user", "content": prompt}])
    latency_s = time.perf_counter() - t0

    return {
        "sample_id": sample.question_id,
        "question": sample.question,
        "question_type": sample.question_type,
        "question_date": sample.question_date,
        "expected_answer": sample.expected_answer,
        "system_answer": system_answer,
        "retrieved_context": context,
        "latency_s": round(latency_s, 3),
    }


def _build_prompt(question: str, question_date: str, context: list[str]) -> str:
    """Prompt v1. Simple y auditable; iteraremos si hace falta."""
    if context:
        context_block = "\n\n".join(f"- {c}" for c in context)
        return (
            f"Today's date: {question_date}\n\n"
            f"Relevant context from previous conversations:\n{context_block}\n\n"
            f"Based on the above, answer the following question concisely.\n"
            f"Question: {question}"
        )
    return (
        f"Today's date: {question_date}\n\n"
        f"Answer the following question concisely.\n"
        f"Question: {question}"
    )
