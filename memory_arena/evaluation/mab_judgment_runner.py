"""
memory_arena.evaluation.mab_judgment_runner
--------------------------------------------
Fase B de MAB: lee un JSONL de respuestas producido por mab_runner,
aplica el juez correspondiente según el sub_dataset, y escribe los verdicts.

Diseño equivalente a `judgment_runner.py` (para LongMemEval), pero:
  - El JSONL de MAB tiene schema distinto (claves `gold_answers` / `sub_dataset`
    en lugar de `expected_answer` / `question_type`).
  - El dispatch del juez depende del `sub_dataset`, no del `question_type`.
  - Para `infbench_sum_eng_shots2` se enriquece el gold con keypoints +
    expert_summary cargados desde HF (el JSONL de MAB solo tiene el expert
    summary en `gold_answers[0]`; los keypoints viven en metadata.keypoints
    de HF y no se persisten en Fase A para mantener el JSONL chico).

Los verdicts se escriben a `results/judgments/<run_id>__<judge_name>.jsonl`
con un record por pregunta (cada record preserva el `sub_dataset` y el
`sample_id` originales para agregación posterior).

Metadata: la corrida emite un JSON a `results/runs/<run_id>.json` con
timestamps, duración, hardware, y git commit — igual que Fase A — para
permitir estimar tiempos y trazar ejecuciones.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from memory_arena.evaluation.judge import (
    JuezBase,
    MABAnswerMatchingJudge,
    MABSummarizationJudge,
)
from memory_arena.evaluation.run_metadata import finalize_run, start_run


# Mapeo sub_dataset -> clase de juez por default. Si se pasa `judges_by_sub`
# explícito, se ignora este mapping y se usan las instancias pasadas.
_DEFAULT_JUDGE_FOR_SUB: dict[str, type[JuezBase]] = {
    "detective_qa": MABAnswerMatchingJudge,
    "infbench_sum_eng_shots2": MABSummarizationJudge,
}

# Cache module-level para referencias de infbench. Evita recargar HF en cada
# llamada dentro del mismo proceso. Single-process; no hace falta locking.
_INFBENCH_REF_CACHE: dict[str, dict] | None = None


def run_mab_judgment(
    responses_path: Path,
    output_dir: Path = Path("results"),
    judge_name: str = "mistral",
    judges_by_sub: dict[str, JuezBase] | None = None,
    infbench_reference_loader: Callable[[], dict[str, dict]] | None = None,
) -> dict:
    """Corre los jueces sobre un JSONL de MAB y devuelve stats agregadas.

    Args:
        responses_path: path al JSONL de Fase A (producido por mab_runner).
        output_dir: raíz de `results/`. Se escribe a `results/judgments/` y
            `results/runs/` (metadata de la corrida del juez).
        judge_name: sufijo del filename final.
        judges_by_sub: dict opcional {sub_dataset: JuezBase} con instancias
            concretas. Si es None, se instancian las default una sola vez y
            se cachean por sub_dataset.
        infbench_reference_loader: callable que devuelve
            {qa_pair_id: {"keypoints": [...], "expert_summary": "..."}}.
            Si es None y el JSONL contiene registros de infbench, se usa
            `load_infbench_references_from_hf` por default.

    Returns:
        Dict con:
          - run_id, judge_name, judges_used, output_path
          - run_metadata_path: path al JSON con timestamps + hardware
          - total: cantidad de records juzgados
          - total_latency_s: tiempo total de la corrida (wall-clock)
          - avg_latency_per_item_s: latencia promedio por record juzgado
          - overall_accuracy: si todos los jueces son booleanos
          - by_sub_dataset: agregados por sub_dataset
              - booleanos -> {kind: "boolean", accuracy, n}
              - estructurados -> {kind: "structured", mean_fluency, mean_recall,
                                  mean_precision, mean_f1, n, out_of_range_count}
    """
    responses_path = Path(responses_path)
    run_id = responses_path.stem
    out_path = output_dir / "judgments" / f"{run_id}__{judge_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadata de la corrida del juez. Usamos el run_id del input enriquecido
    # con el nombre del juez para que quede trazable.
    # La convención de naming de start_run genera:
    #   <timestamp>_<strategy>_<benchmark>
    # Acá usamos strategy=f"judge_{judge_name}" y benchmark=<run_id base>.
    # Contamos preguntas contándolas durante la corrida (no podemos saber de
    # antemano cuántas líneas tiene el JSONL sin leerlo dos veces).
    # Dejamos num_samples en 0 y lo actualizamos al final.
    metadata = start_run(
        strategy=f"judge_{judge_name}",
        benchmark=run_id,
        model="n/a",  # se sobrescribe al primer verdict
        num_samples=0,
    )

    judges: dict[str, JuezBase] = dict(judges_by_sub) if judges_by_sub else {}

    per_sub_bool: dict[str, list[bool]] = {}
    per_sub_scores: dict[str, list[dict[str, float]]] = {}
    judges_used: dict[str, str] = {}
    total = 0

    t0 = time.perf_counter()
    try:
        with (
            open(responses_path, "r", encoding="utf-8") as fin,
            open(out_path, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sub = rec.get("sub_dataset") or ""

                # Resolver juez (lazy instancing).
                if sub not in judges:
                    judge_cls = _DEFAULT_JUDGE_FOR_SUB.get(sub)
                    if judge_cls is None:
                        raise ValueError(
                            f"No hay juez configurado para sub_dataset={sub!r}. "
                            f"Pasalo en judges_by_sub o agrega un mapping en "
                            f"_DEFAULT_JUDGE_FOR_SUB."
                        )
                    judges[sub] = judge_cls()

                judge = judges[sub]

                expected_answer = _build_expected_answer(
                    rec=rec,
                    sub_dataset=sub,
                    infbench_loader=infbench_reference_loader,
                )

                t_item0 = time.perf_counter()
                verdict = judge.judge(
                    question=rec.get("question", "") or "",
                    expected_answer=expected_answer,
                    system_answer=rec.get("system_answer", "") or "",
                    question_type=sub,
                    question_id=rec.get("question_id", "") or "",
                )
                item_latency = time.perf_counter() - t_item0
                judges_used[sub] = verdict.judge_model

                out_rec = {
                    "sample_id": rec.get("sample_id"),
                    "question_id": rec.get("question_id"),
                    "sub_dataset": sub,
                    "label": verdict.label,
                    "scores": verdict.scores,
                    "judge_model": verdict.judge_model,
                    "raw_response": verdict.raw_response,
                    "judge_latency_s": round(item_latency, 3),
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                fout.flush()

                if verdict.label is not None:
                    per_sub_bool.setdefault(sub, []).append(verdict.label)
                if verdict.scores is not None:
                    per_sub_scores.setdefault(sub, []).append(verdict.scores)
                total += 1
    finally:
        # Persistimos metadata SIEMPRE, incluso si la corrida se cortó.
        # Esto nos deja rastro de tiempos parciales.
        metadata.model = "+".join(sorted(set(judges_used.values()))) or "n/a"
        metadata.num_samples = total
        runs_path = output_dir / "runs" / f"{metadata.run_id}.json"
        finalize_run(metadata, runs_path)

    total_latency_s = round(time.perf_counter() - t0, 3)
    avg_latency_per_item_s = round(total_latency_s / total, 3) if total else 0.0

    # Agregados por sub_dataset.
    by_sub: dict[str, dict] = {}
    for sub, labels in per_sub_bool.items():
        n = len(labels)
        by_sub[sub] = {
            "kind": "boolean",
            "n": n,
            "accuracy": round(sum(labels) / n, 4) if n else None,
        }
    for sub, scores_list in per_sub_scores.items():
        n = len(scores_list)
        if n == 0:
            continue
        means = {
            f"mean_{k}": round(sum(s.get(k, 0.0) for s in scores_list) / n, 4)
            for k in ("fluency", "recall", "precision", "f1")
        }
        # Si el juez de este sub es un MABSummarizationJudge, levantamos
        # su contador de out-of-range (bugs del juez local).
        oor_count = 0
        judge_for_sub = judges.get(sub)
        if judge_for_sub is not None and hasattr(judge_for_sub, "out_of_range_count"):
            oor_count = int(judge_for_sub.out_of_range_count)
        by_sub[sub] = {
            "kind": "structured",
            "n": n,
            "out_of_range_count": oor_count,
            **means,
        }

    # Overall accuracy solo si todos los jueces fueron booleanos.
    all_bool = [lab for labs in per_sub_bool.values() for lab in labs]
    any_structured = bool(per_sub_scores)
    overall: float | None
    if any_structured or not all_bool:
        overall = None
    else:
        overall = round(sum(all_bool) / len(all_bool), 4)

    return {
        "run_id": run_id,
        "judge_name": judge_name,
        "judges_used": judges_used,
        "output_path": str(out_path),
        "run_metadata_path": str(output_dir / "runs" / f"{metadata.run_id}.json"),
        "total": total,
        "total_latency_s": total_latency_s,
        "avg_latency_per_item_s": avg_latency_per_item_s,
        "overall_accuracy": overall,
        "by_sub_dataset": by_sub,
    }


# ----------------------------------------------------------------------------
# Construcción de expected_answer según sub_dataset
# ----------------------------------------------------------------------------


def _build_expected_answer(
    rec: dict,
    sub_dataset: str,
    infbench_loader: Callable[[], dict[str, dict]] | None,
) -> str:
    """Arma el `expected_answer` que recibe el juez, según el sub_dataset.

    - detective_qa (MCQA): concatena gold_answers con " / " (por si hay varios
      aceptables). El juez ve la pregunta completa (con las 4 opciones).
    - infbench_sum_eng_shots2: devuelve JSON `{"keypoints": [...],
      "expert_summary": "..."}` — formato que parsea MABSummarizationJudge.
    - default: string plano con gold_answers concatenados.
    """
    global _INFBENCH_REF_CACHE

    golds = rec.get("gold_answers") or []
    if isinstance(golds, str):
        golds = [golds]

    if sub_dataset == "infbench_sum_eng_shots2":
        if _INFBENCH_REF_CACHE is None:
            loader = infbench_loader or load_infbench_references_from_hf
            _INFBENCH_REF_CACHE = loader()
        refs = _INFBENCH_REF_CACHE

        qid = rec.get("question_id") or ""
        sample_id = rec.get("sample_id") or ""

        entry = refs.get(qid) or refs.get(str(qid)) or refs.get(sample_id)
        if entry is None:
            return json.dumps({
                "keypoints": [],
                "expert_summary": golds[0] if golds else "",
            })
        return json.dumps({
            "keypoints": entry.get("keypoints") or [],
            "expert_summary": (
                entry.get("expert_summary")
                or (golds[0] if golds else "")
            ),
        })

    # MCQA y resto: concatenar golds con " / " como separador.
    if not golds:
        return ""
    return " / ".join(str(g) for g in golds)


# ----------------------------------------------------------------------------
# Reference loading para infbench_sum_eng_shots2 (keypoints + expert_summary)
# ----------------------------------------------------------------------------


def load_infbench_references_from_hf(
    huggingface_dataset_name: str = "ai-hyz/MemoryAgentBench",
    source_dataset_name: str = "infbench_sum_eng_shots2",
) -> dict[str, dict]:
    """Carga keypoints + expert_summary del dataset HF para `infbench_sum_eng_shots2`.

    Replica la lógica de `summarization_evaluate.load_data_from_huggingface` del
    repo oficial: recorre todos los splits, filtra por source, y arma un dict
    {qa_pair_id: {"keypoints", "expert_summary"}}. Se ejecuta una sola vez
    gracias al cache en `_INFBENCH_REF_CACHE`.

    Por qué HF en lugar de local: los keypoints no se persisten en el JSONL
    de Fase A (decisión de diseño: mantener el JSONL chico). Si en el futuro
    queremos ahorrar esta re-descarga, persistir keypoints en el JSONL.

    Returns:
        Dict del tipo:
            {
                "infbench_001": {
                    "keypoints": ["...", "..."],
                    "expert_summary": "..."
                },
                ...
            }
    """
    from datasets import load_dataset  # import diferido

    full = load_dataset(huggingface_dataset_name, revision="main")
    refs: dict[str, dict] = {}
    for split_name in full.keys():
        filtered = full[split_name].filter(
            lambda ex: (ex.get("metadata") or {}).get("source", "") == source_dataset_name
        )
        for entry in filtered:
            metadata = entry.get("metadata") or {}
            qa_ids = metadata.get("qa_pair_ids") or []
            keypoints = (
                metadata.get("keypoints")
                or metadata.get("summary/short_keypoints")
                or []
            )
            answers = entry.get("answers") or [[]]
            if answers and isinstance(answers[0], list) and answers[0]:
                expert_summary = answers[0][0]
            elif answers:
                expert_summary = answers[0]
            else:
                expert_summary = ""
            if not isinstance(expert_summary, str):
                expert_summary = str(expert_summary)
            doc_id = qa_ids[0] if qa_ids else ""
            refs[str(doc_id)] = {
                "keypoints": [str(kp) for kp in keypoints],
                "expert_summary": expert_summary,
            }
    return refs
