"""
memory_arena.evaluation.judgment_runner
----------------------------------------
Fase B del pipeline de LongMemEval: lee un JSONL de respuestas (producido por
Fase A), aplica un juez y escribe los veredictos a JSONL.

Esta separacion permite correr N jueces distintos sobre el mismo set de
respuestas sin regenerar respuestas (que es la parte cara del pipeline).
Util para analisis de sensibilidad al juez.

Metadata: emite un JSON a `results/runs/<run_id_juez>.json` con timestamps,
duración, hardware y git commit — mismo patrón que `mab_judgment_runner.py` y
`runner.py`/`mab_runner.py` de Fase A.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from memory_arena.evaluation.judge import JuezBase
from memory_arena.evaluation.run_metadata import finalize_run, start_run


def run_judgment(
    responses_path: Path,
    judge: JuezBase,
    output_dir: Path = Path("results"),
    judge_name: str = "mistral",
) -> dict:
    """Corre un juez sobre un JSONL de respuestas y devuelve stats agregadas.

    Args:
        responses_path: path al JSONL generado por Fase A.
        judge: instancia concreta de JuezBase.
        output_dir: raiz de `results/`. Se escribe a `results/judgments/` y
            `results/runs/` (metadata).
        judge_name: nombre corto del juez para el filename final.

    Returns:
        Dict con:
          - run_id, judge_name, judge_model, output_path, run_metadata_path
          - total: cantidad de samples juzgados
          - total_latency_s: tiempo total wall-clock
          - avg_latency_per_item_s: latencia promedio por item
          - overall_accuracy: proporcion de labels True sobre el total
          - by_type / counts_by_type: accuracy y conteo por question_type
    """
    responses_path = Path(responses_path)
    run_id = responses_path.stem
    out_path = output_dir / "judgments" / f"{run_id}__{judge_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadata de la corrida del juez. num_samples lo actualizamos al cierre
    # (no conocemos el total sin recorrer el JSONL primero).
    metadata = start_run(
        strategy=f"judge_{judge_name}",
        benchmark=run_id,
        model="n/a",  # se sobrescribe con el judge_model del primer verdict
        num_samples=0,
    )

    per_type: dict[str, list[bool]] = {}
    total = 0
    judge_model_used: str | None = None

    t0 = time.perf_counter()
    try:
        with (
            open(responses_path, "r", encoding="utf-8") as fin,
            open(out_path, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                rec = json.loads(line)
                t_item0 = time.perf_counter()
                verdict = judge.judge(
                    question=rec["question"],
                    expected_answer=rec["expected_answer"],
                    system_answer=rec["system_answer"],
                    question_type=rec["question_type"],
                    question_id=rec["sample_id"],
                )
                item_latency = time.perf_counter() - t_item0
                judge_model_used = verdict.judge_model

                out_rec = {
                    "sample_id": rec["sample_id"],
                    "question_type": rec["question_type"],
                    "label": verdict.label,
                    "judge_model": verdict.judge_model,
                    "raw_response": verdict.raw_response,
                    "judge_latency_s": round(item_latency, 3),
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                fout.flush()

                per_type.setdefault(rec["question_type"], []).append(verdict.label)
                total += 1
    finally:
        metadata.model = judge_model_used or "n/a"
        metadata.num_samples = total
        runs_path = output_dir / "runs" / f"{metadata.run_id}.json"
        finalize_run(metadata, runs_path)

    total_latency_s = round(time.perf_counter() - t0, 3)
    avg_latency_per_item_s = round(total_latency_s / total, 3) if total else 0.0

    if total == 0:
        overall: float | None = None
    else:
        all_labels = [lab for labs in per_type.values() for lab in labs]
        overall = round(sum(all_labels) / len(all_labels), 4)

    return {
        "run_id": run_id,
        "judge_name": judge_name,
        "judge_model": judge_model_used,
        "output_path": str(out_path),
        "run_metadata_path": str(output_dir / "runs" / f"{metadata.run_id}.json"),
        "total": total,
        "total_latency_s": total_latency_s,
        "avg_latency_per_item_s": avg_latency_per_item_s,
        "overall_accuracy": overall,
        "by_type": {k: round(sum(v) / len(v), 4) for k, v in per_type.items()},
        "counts_by_type": {k: len(v) for k, v in per_type.items()},
    }
