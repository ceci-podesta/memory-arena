"""
memory_arena.evaluation.mab_runner
-----------------------------------
Runner para MemoryAgentBench con diseño inject-once-query-many.
Para cada MABSample:
  1. reset() de la estrategia (memoria limpia).
  2. store(turn) una sola vez con el contexto largo envuelto en un Turn sintético.
  3. Por cada pregunta del sample: retrieve(question) -> prompt -> llm.chat() -> JSONL.
No hace scoring: solo produce las responses. El scoring vive en mab_scoring.py (Bloque C).
Sigue el mismo patrón que evaluation/runner.py (LongMemEval), adaptado a la
semántica "inject once, query multiple times" de MAB.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from memory_arena.benchmarks.memory_agent_bench import MABSample
from memory_arena.evaluation.run_metadata import RunMetadata, finalize_run, start_run
from memory_arena.experimental_config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RETRIEVAL_TOP_K,
)
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.base import MemoriaBase, Turn


def run_strategy_mab(
    strategy: MemoriaBase,
    samples: list[MABSample],
    llm: OllamaClient,
    strategy_name: str,
    split: str,
    sub_dataset: str,
    output_dir: Path = Path("results"),
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> RunMetadata:
    """Corre una estrategia de memoria sobre samples de MAB (inject-once-query-many).
    Args:
        strategy: instancia de MemoriaBase (NoMemoria, RAGMemoria, etc.).
        samples: lista de MABSample (ya cargados con load_mab).
        llm: cliente del LLM evaluado.
        strategy_name: nombre corto para el run_id y filenames (ej: "no_memory").
        split: AR/TTL/LRU/CR.
        sub_dataset: identificador real de HF (ej: "longmemeval_s*", "detective_qa").
        output_dir: base dir. Se crean output_dir/responses/ y output_dir/runs/.
        max_new_tokens: limite de tokens de salida del LLM (mapea a num_predict).
        top_k: cuántos fragmentos pedirle a retrieve().
    Returns:
        RunMetadata finalizada (con start/end_ts, duración, etc.).
    """
    # "*" y "/" no se llevan bien con algunos FS; sanitizamos el nombre del benchmark.
    sub_clean = sub_dataset.replace("*", "_star").replace("/", "_")
    benchmark_name = f"mab_{split}_{sub_clean}"
    metadata = start_run(
        strategy=strategy_name,
        benchmark=benchmark_name,
        model=llm.model,
        num_samples=len(samples),
    )
    responses_path = output_dir / "responses" / f"{metadata.run_id}.jsonl"
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    total_questions = 0
    try:
        with open(responses_path, "w", encoding="utf-8") as f:
            for sample in samples:
                records = _process_sample(
                    strategy=strategy,
                    sample=sample,
                    llm=llm,
                    split=split,
                    sub_dataset=sub_dataset,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                )
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()  # persistir pregunta a pregunta por si se cae la corrida
                    total_questions += 1
    finally:
        runs_path = output_dir / "runs" / f"{metadata.run_id}.json"
        finalize_run(metadata, runs_path)
    print(
        f"[mab_runner] {strategy_name} | {benchmark_name} | "
        f"{len(samples)} samples -> {total_questions} questions | "
        f"responses: {responses_path}"
    )
    return metadata


def _process_sample(
    strategy: MemoriaBase,
    sample: MABSample,
    llm: OllamaClient,
    split: str,
    sub_dataset: str,
    max_new_tokens: int,
    top_k: int,
) -> list[dict]:
    """Procesa un MABSample completo: inyecta el contexto una vez, responde N preguntas.
    Devuelve una lista de dicts (uno por pregunta), lista para serializar como JSONL.
    """
    # Inject-once: reset + store del contexto completo como un único Turn sintético.
    strategy.reset()
    strategy.store(_sample_to_ingestion_turn(sample))
    records: list[dict] = []
    for q_idx, (question, question_id) in enumerate(
        zip(sample.questions, sample.question_ids)
    ):
        retrieved = strategy.retrieve(question, top_k=top_k)
        prompt = _build_prompt_mab(question, retrieved)
        t0 = time.perf_counter()
        system_answer = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        latency_s = time.perf_counter() - t0
        records.append(
            {
                "split": split,
                "sub_dataset": sub_dataset,
                "source": sample.source,
                "sample_id": sample.sample_id,
                "question_id": question_id,
                "question_idx": q_idx,
                "question": question,
                "gold_answers": sample.answers[q_idx],
                "system_answer": system_answer,
                "retrieved_context": retrieved,
                "retrieved_count": len(retrieved),
                "latency_s": round(latency_s, 3),
            }
        )
    return records


def _sample_to_ingestion_turn(sample: MABSample) -> Turn:
    """Envuelve el contexto largo del sample en un Turn sintético para ingestarlo.
    MAB inyecta el contexto como un bloque único; no tenemos turnos reales de
    diálogo como en LongMemEval. Usamos role='document' para señalizar que es
    ingesta documental, y session_id=sample_id para trazabilidad en logs.
    Las estrategias que ignoran `role` (NoMemoria, verbatim+RAG) no se
    enteran; las que lo usan pueden diferenciar documento vs. turno real.
    """
    return Turn(
        role="document",
        content=sample.context,
        session_id=sample.sample_id,
        date=None,
    )


def _build_prompt_mab(question: str, retrieved: list[str]) -> str:
    """Construye el prompt para MAB.
    No usa framing conversacional ni "today's date": MAB evalúa retrieval y
    reasoning sobre documentos, no conversación temporal como LongMemEval.
    """
    if retrieved:
        context_block = "\n\n---\n\n".join(retrieved)
        return (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely."
        )
    return f"Question: {question}\n\nAnswer concisely."
