"""
memory_arena.evaluation.runner_resumable
-----------------------------------------
Variante de runner.py que permite reanudar una corrida interrumpida.

Si se le pasa ``resume_path`` con un JSONL existente, lee los ``sample_id``
ya procesados, skipea esos samples, y appendea los nuevos al MISMO archivo.
Si ``resume_path`` es None, comportamiento idéntico al runner original.

No toca runner.py original (infra compartida del equipo).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from memory_arena.benchmarks.longmemeval import LongMemEvalSample
from memory_arena.evaluation.run_metadata import (
    RunMetadata,
    finalize_run,
    start_run,
)
from memory_arena.experimental_config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RETRIEVAL_TOP_K,
)
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.base import MemoriaBase, Turn


def run_strategy_resumable(
    strategy: MemoriaBase,
    samples: list[LongMemEvalSample],
    llm: OllamaClient,
    strategy_name: str,
    benchmark_name: str,
    output_dir: Path = Path("results"),
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    resume_path: Path | None = None,
) -> RunMetadata:
    """Corre la estrategia con soporte de reanudación.

    Si ``resume_path`` apunta a un JSONL existente, lee los sample_id ya
    procesados, los skipea, y appendea los nuevos al mismo archivo.
    """
    already_done: set[str] = set()
    if resume_path is not None and resume_path.exists():
        with open(resume_path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    qid = record.get("sample_id")
                    if qid:
                        already_done.add(qid)
                except json.JSONDecodeError:
                    pass  # linea malformada -> ignorar
        n_pending = len(samples) - len(already_done)
        print(
            f"[resumable] Reanudando desde {resume_path.name}: "
            f"{len(already_done)} samples ya procesados, "
            f"{n_pending} pendientes"
        )
        responses_path = resume_path
        metadata = start_run(
            strategy=strategy_name,
            benchmark=benchmark_name + "_resume",
            model=llm.model,
            num_samples=n_pending,
        )
        file_mode = "a"
    else:
        metadata = start_run(
            strategy=strategy_name,
            benchmark=benchmark_name,
            model=llm.model,
            num_samples=len(samples),
        )
        responses_path = output_dir / "responses" / f"{metadata.run_id}.jsonl"
        responses_path.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "w"

    try:
        with open(responses_path, file_mode, encoding="utf-8") as f:
            for sample in samples:
                if sample.question_id in already_done:
                    continue
                record = _process_sample(
                    strategy, sample, llm, top_k, max_new_tokens
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        runs_path = output_dir / "runs" / f"{metadata.run_id}.json"
        finalize_run(metadata, runs_path)

    return metadata


def _process_sample(
    strategy: MemoriaBase,
    sample: LongMemEvalSample,
    llm: OllamaClient,
    top_k: int,
    max_new_tokens: int,
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
    system_answer = llm.chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
    )
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


def _build_prompt(
    question: str, question_date: str, context: list[str]
) -> str:
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
