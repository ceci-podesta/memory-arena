"""
memory_arena.evaluation.judgment_runner
----------------------------------------
Fase B del pipeline: lee un JSONL de respuestas (producido por Fase A),
aplica un juez y escribe los veredictos a JSONL.

Esta separacion permite correr N jueces distintos sobre el mismo set de
respuestas sin regenerar respuestas (que es la parte cara del pipeline).
Util para analisis de sensibilidad al juez.
"""

import json
from pathlib import Path

from memory_arena.evaluation.judge import JuezBase


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
        output_dir: raiz de `results/`. Se escribe a `results/judgments/`.
        judge_name: nombre corto del juez para el filename final.

    Returns:
        Dict con:
          - run_id, judge_name, judge_model, output_path
          - total: cantidad de samples juzgados
          - overall_accuracy: proporcion de labels True sobre el total
          - by_type / counts_by_type: accuracy y conteo por question_type
    """
    run_id = responses_path.stem
    out_path = output_dir / "judgments" / f"{run_id}__{judge_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_type: dict[str, list[bool]] = {}
    total = 0
    judge_model_used: str | None = None

    with (
        open(responses_path, "r", encoding="utf-8") as fin,
        open(out_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            rec = json.loads(line)
            verdict = judge.judge(
                question=rec["question"],
                expected_answer=rec["expected_answer"],
                system_answer=rec["system_answer"],
                question_type=rec["question_type"],
                question_id=rec["sample_id"],
            )
            judge_model_used = verdict.judge_model

            out_rec = {
                "sample_id": rec["sample_id"],
                "question_type": rec["question_type"],
                "label": verdict.label,
                "judge_model": verdict.judge_model,
                "raw_response": verdict.raw_response,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

            per_type.setdefault(rec["question_type"], []).append(verdict.label)
            total += 1

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
        "total": total,
        "overall_accuracy": overall,
        "by_type": {k: round(sum(v) / len(v), 4) for k, v in per_type.items()},
        "counts_by_type": {k: len(v) for k, v in per_type.items()},
    }
