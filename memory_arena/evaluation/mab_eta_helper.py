"""
memory_arena.evaluation.mab_eta_helper
---------------------------------------
Wrapper de run_strategy_mab que mide tiempo sample-por-sample y reporta
ETA progresivo basado en la mediana de samples ya procesados.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median

from memory_arena.benchmarks.memory_agent_bench import MABSample
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


def run_strategy_mab_with_eta(
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

    sample_durations: list[float] = []
    total_questions = 0

    print(
        f"  [eta] arrancando {len(samples)} samples — primer ETA disponible tras sample 1",
        flush=True,
    )

    try:
        with open(responses_path, "w", encoding="utf-8") as f:
            for i, sample in enumerate(samples, 1):
                t_sample = time.perf_counter()
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
                    f.flush()
                    total_questions += 1

                elapsed = time.perf_counter() - t_sample
                sample_durations.append(elapsed)

                med = median(sample_durations)
                remaining = len(samples) - i
                eta_sec = med * remaining

                print(
                    f"  [{i}/{len(samples)}] {sample.sample_id[:40]:<40} "
                    f"-> {elapsed/60:>5.1f} min  "
                    f"| med: {med/60:>5.1f} min/sample  "
                    f"| ETA: {eta_sec/60:>6.1f} min ({eta_sec/3600:.2f} hs)",
                    flush=True,
                )
    finally:
        runs_path = output_dir / "runs" / f"{metadata.run_id}.json"
        finalize_run(metadata, runs_path)

    print(
        f"  [eta] sub_dataset terminado: {len(samples)} samples, "
        f"{total_questions} questions, "
        f"total {sum(sample_durations)/60:.1f} min",
        flush=True,
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
    strategy.reset()
    strategy.store(
        Turn(
            role="document",
            content=sample.context,
            session_id=sample.sample_id,
            date=None,
        )
    )
    records: list[dict] = []
    for q_idx, (question, qid) in enumerate(
        zip(sample.questions, sample.question_ids)
    ):
        retrieved = strategy.retrieve(question, top_k=top_k)
        prompt = _build_prompt(question, retrieved)
        t0 = time.perf_counter()
        answer = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        latency_s = time.perf_counter() - t0
        records.append({
            "split": split,
            "sub_dataset": sub_dataset,
            "source": sample.source,
            "sample_id": sample.sample_id,
            "question_id": qid,
            "question_idx": q_idx,
            "question": question,
            "gold_answers": sample.answers[q_idx],
            "system_answer": answer,
            "retrieved_context": retrieved,
            "retrieved_count": len(retrieved),
            "latency_s": round(latency_s, 3),
        })
    return records


def _build_prompt(question: str, retrieved: list[str]) -> str:
    if retrieved:
        context_block = "\n\n---\n\n".join(retrieved)
        return (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely."
        )
    return f"Question: {question}\n\nAnswer concisely."
