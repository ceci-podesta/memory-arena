"""
memory_arena.evaluation.mab_scoring
------------------------------------
Scoring para MemoryAgentBench, replicando fielmente el scorer oficial
del paper (Hu et al., ICLR 2026).

Referencia: MemoryAgentBench/utils/eval_other_utils.py

Qué implementa (default del paper, aplicable a AR + CR + LRU[autom.] + TTL/icl):
  - exact_match, substring_exact_match, f1                  (sin deps extras)
  - rougeL_f1, rougeL_recall, rougeLsum_f1, rougeLsum_recall (rouge-score)

Ramas específicas por sub-dataset (dispatcher idéntico al oficial):
  - eventqa_*      -> agrega eventqa_recall (fracción de gold elements en predicción)
  - icl_*          -> aplica parse_output explícito antes del default
  - recsys_*       -> delega en recsys_scorer.score_recsys_response (Bloque F):
                      devuelve recsys_recall@1/5/10 + n_gold + n_predicted.
                      NO hace fallback al default porque las métricas léxicas no
                      aplican a recomendación por IDs.
  - default        -> default (usa system_answer directo + parsed; toma max por métrica)

Qué NO implementa (fuera de scope):
  - Juez LLM para LRU              -> vive en judge.py + mab_judgment_runner.py
  - choice_eng variant             -> ninguno de nuestros sub-datasets es choice
  - ruler_niah                     -> usamos ruler_qa*, no niah

API pública:
    calculate_default_metrics(prediction, gold_answers) -> dict[str, float]
    score_response(system_answer, gold_answers, sub_dataset) -> dict[str, float]
    score_jsonl(jsonl_path) -> dict
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from rouge_score import rouge_scorer


# ----------------------------------------------------------------------------
# Normalización (idéntica al scorer oficial)
# ----------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """Normalización estándar DrQA: lowercase + sin puntuación + sin artículos + ws collapsed.

    Es la misma normalización que usa el repo oficial de MAB y que viene de
    DensePhrases/DrQA. La mantenemos idéntica para poder comparar scores.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


# ----------------------------------------------------------------------------
# Métricas atómicas (por ground_truth individual)
# ----------------------------------------------------------------------------


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """1.0 si prediction == ground_truth tras normalizar, 0.0 si no."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def substring_exact_match_score(prediction: str, ground_truth: str) -> float:
    """1.0 si el ground_truth normalizado aparece como substring en la predicción normalizada.

    Esta es la métrica más importante para AR/CR: el modelo puede envolver la
    respuesta correcta en relleno ("La respuesta es X porque..."), y el
    substring match captura ese caso donde EM falla.
    """
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))


_SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


def f1_token_score(prediction: str, ground_truth: str) -> float:
    """F1 por overlap de tokens (DrQA-style). Devuelve solo el F1.

    Regla especial del scorer oficial: si uno es yes/no/noanswer y el otro no,
    devuelve 0 automáticamente. Evita que "yes, la respuesta es que no" dé F1
    alto contra un gold "no".
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    if (
        norm_pred in _SPECIAL_ANSWERS or norm_gt in _SPECIAL_ANSWERS
    ) and norm_pred != norm_gt:
        return 0.0

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0 or not pred_tokens or not gt_tokens:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ----------------------------------------------------------------------------
# Helpers para múltiples ground truths
# ----------------------------------------------------------------------------


def _flatten_golds(gold_answers: str | list) -> list[str]:
    """Aplana gold_answers a una lista plana de strings.

    MAB a veces trae las respuestas como string, como list[str] o como list[list[str]]
    (paralela a questions). El scorer oficial flattenea todo antes de scorear.
    """
    if isinstance(gold_answers, str):
        return [gold_answers]
    if not gold_answers:
        return []
    if isinstance(gold_answers[0], list):
        return [gt for sub in gold_answers for gt in sub]
    return [str(gt) for gt in gold_answers]


def _max_over_golds(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    gold_answers: Iterable[str],
) -> float:
    """Devuelve el máximo score de prediction contra cada ground_truth."""
    golds = list(gold_answers)
    if not golds:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in golds)


# ----------------------------------------------------------------------------
# ROUGE
# ----------------------------------------------------------------------------

# Scorer compartido (stemmer Porter). use_stemmer=True es lo que usa el oficial.
_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"], use_stemmer=True)


def _rouge_scores(prediction: str, golds_flat: list[str]) -> dict[str, float]:
    """Calcula rougeL y rougeLsum (f1 y recall), tomando max sobre golds.

    Convención del oficial: el f1 = fmeasure de la librería; el recall = recall.
    """
    if not golds_flat:
        return {
            "rougeL_f1": 0.0,
            "rougeL_recall": 0.0,
            "rougeLsum_f1": 0.0,
            "rougeLsum_recall": 0.0,
        }
    per_gold = [_ROUGE_SCORER.score(target=gt, prediction=prediction) for gt in golds_flat]
    return {
        "rougeL_f1": max(s["rougeL"].fmeasure for s in per_gold),
        "rougeL_recall": max(s["rougeL"].recall for s in per_gold),
        "rougeLsum_f1": max(s["rougeLsum"].fmeasure for s in per_gold),
        "rougeLsum_recall": max(s["rougeLsum"].recall for s in per_gold),
    }


# ----------------------------------------------------------------------------
# parse_output (extrae "Answer: ..." si el modelo lo pone explícito)
# ----------------------------------------------------------------------------


def parse_output(output_text: str, answer_prefix: str = "Answer:") -> str | None:
    """Intenta extraer la parte de la respuesta después de un prefijo tipo 'Answer:'.

    Si no encuentra el prefijo, toma la primera línea. Devuelve None si el input
    está vacío.
    """
    if not output_text:
        return None

    patterns = [
        re.compile(f"(?:{re.escape(answer_prefix)})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pat in patterns:
        m = pat.search(output_text)
        if m:
            extracted = m.group(1).strip()
            # Sacar el prefijo si quedó repetido.
            cleaned = re.sub(
                f"^{re.escape(answer_prefix)}",
                "",
                extracted,
                flags=re.IGNORECASE,
            ).strip()
            return cleaned
    return None


# ----------------------------------------------------------------------------
# Default metrics (7 métricas del paper, multi-gold)
# ----------------------------------------------------------------------------


def calculate_default_metrics(
    prediction: str, gold_answers: str | list
) -> dict[str, float]:
    """Las 7 métricas default del paper, con max sobre múltiples ground truths.

    Args:
        prediction: el texto generado por el modelo.
        gold_answers: string, list[str] o list[list[str]] con las respuestas
            aceptables. Se flattenea internamente.

    Returns:
        Dict con exact_match, substring_exact_match, f1, rougeL_f1, rougeL_recall,
        rougeLsum_f1, rougeLsum_recall. Todos son floats en [0, 1].
    """
    golds_flat = _flatten_golds(gold_answers)

    metrics: dict[str, float] = {
        "exact_match": _max_over_golds(exact_match_score, prediction, golds_flat),
        "substring_exact_match": _max_over_golds(
            substring_exact_match_score, prediction, golds_flat
        ),
        "f1": _max_over_golds(f1_token_score, prediction, golds_flat),
    }
    metrics.update(_rouge_scores(prediction, golds_flat))
    return metrics


# ----------------------------------------------------------------------------
# Ramas específicas por sub-dataset
# ----------------------------------------------------------------------------


def _eventqa_recall_metrics(
    prediction: str, gold_answers: str | list
) -> dict[str, float]:
    """eventqa_recall: fracción de gold elements (case-insensitive) presentes en predicción.

    En el scorer oficial hay dos variantes: la fracción ("eventqa_recall" float
    en [0,1]) y el binario "1 si todos aparecen". Acá devolvemos la fracción,
    que es más informativa y se puede umbrar post-hoc si se quiere binario.
    """
    golds_flat = _flatten_golds(gold_answers)
    if not golds_flat:
        return {"eventqa_recall": 0.0}
    pred_lower = prediction.lower()
    hits = sum(1 for g in golds_flat if g.lower() in pred_lower)
    return {"eventqa_recall": hits / len(golds_flat)}


def _default_branch(
    system_answer: str, gold_answers: str | list
) -> dict[str, float]:
    """Scoring default: calcula metrics sobre system_answer directo y sobre parse_output(system_answer),
    toma el máximo de cada métrica. Replica default_post_process del oficial.
    """
    direct = calculate_default_metrics(system_answer, gold_answers)
    parsed = parse_output(system_answer)
    if parsed is None:
        return direct
    parsed_metrics = calculate_default_metrics(parsed, gold_answers)
    return {k: max(direct[k], parsed_metrics[k]) for k in direct}


def _icl_branch(
    system_answer: str, gold_answers: str | list
) -> dict[str, float]:
    """Scoring icl_*: aplica parse_output y scorea sobre la respuesta extraída.

    A diferencia del default (que scorea ambos y toma max), el branch icl
    del oficial scorea solo la parsed. Lo replico idéntico.
    """
    parsed = parse_output(system_answer)
    parsed_text = parsed if parsed is not None else system_answer
    return calculate_default_metrics(parsed_text, gold_answers)


def _recsys_branch(
    system_answer: str, gold_answers: str | list
) -> dict:
    """Scoring recsys_*: delega al scorer Bloque F (entity2id + fuzzy matching).

    Import diferido para no requerir `editdistance` si el pipeline no corre
    sub_datasets de recsys (útil para testing aislado del core scorer).

    Devuelve keys: recsys_recall@1/5/10, n_gold, n_predicted + _parsed_output
    y _gt_movies (debug, prefijados con _ para que _compute_aggregates los
    ignore en la agregación).

    Si el mapping entity2id no está disponible (FileNotFoundError), loggea
    y devuelve recall=0 para no romper el pipeline; se reporta al agregar.
    """
    try:
        from memory_arena.evaluation.recsys_scorer import score_recsys_response
        return score_recsys_response(system_answer, gold_answers)
    except FileNotFoundError as e:
        # El caller puede leer esto; imprimimos una sola vez en stderr para no spamear.
        import sys
        print(
            f"[mab_scoring._recsys_branch] entity2id no disponible: {e}. "
            f"Devolviendo recall=0 para este sample.",
            file=sys.stderr,
        )
        return {
            "recsys_recall@1": 0.0,
            "recsys_recall@5": 0.0,
            "recsys_recall@10": 0.0,
            "n_gold": 0,
            "n_predicted": 0,
        }


# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------


def score_response(
    system_answer: str,
    gold_answers: str | list,
    sub_dataset: str,
) -> dict[str, float]:
    """Dispatcher que aplica el scoring correcto según el sub_dataset.

    Args:
        system_answer: respuesta generada por el modelo.
        gold_answers: respuestas aceptables (string, list[str] o list[list[str]]).
        sub_dataset: identificador del sub-dataset (valor de metadata.source en HF).

    Returns:
        Dict con las métricas aplicables:
        - Default (AR, CR, longmemeval_s*, factconsolidation_*, ruler_qa*,
          detective_qa, infbench_sum_*): 7 métricas default.
        - eventqa_*: default + eventqa_recall.
        - icl_*: default sobre parse_output.
        - recsys_*: recsys_recall@1/5/10 + n_gold + n_predicted (NO default).
    """
    name = sub_dataset.lower()

    if "eventqa" in name:
        metrics = _default_branch(system_answer, gold_answers)
        metrics.update(_eventqa_recall_metrics(system_answer, gold_answers))
        return metrics

    if "icl" in name:
        return _icl_branch(system_answer, gold_answers)

    if "recsys" in name:
        return _recsys_branch(system_answer, gold_answers)

    # Default: longmemeval_s*, factconsolidation_*, ruler_qa*, detective_qa,
    # infbench_sum_*, etc.
    return _default_branch(system_answer, gold_answers)


# ----------------------------------------------------------------------------
# Scoring sobre JSONL producido por mab_runner
# ----------------------------------------------------------------------------


def score_jsonl(jsonl_path: Path) -> dict:
    """Lee un JSONL generado por mab_runner.run_strategy_mab y devuelve métricas.

    Cada línea del JSONL tiene las claves (ver mab_runner._process_sample):
        sample_id, question_id, question, gold_answers, system_answer,
        sub_dataset, source, ...

    Returns:
        {
            "per_question": [{...record original..., "metrics": {...}}, ...],
            "aggregates": {
                "n_total": int,
                "by_metric": {
                    "exact_match": {"mean": float, "n": int},
                    ...
                },
                "by_sub_dataset": {
                    "<sub_dataset>": {
                        "n": int,
                        "exact_match": {"mean": float, "n": int},
                        ...
                    }
                }
            }
        }
    """
    path = Path(jsonl_path)
    per_question: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metrics = score_response(
                system_answer=record.get("system_answer", "") or "",
                gold_answers=record.get("gold_answers", []),
                sub_dataset=record.get("sub_dataset", ""),
            )
            per_question.append({**record, "metrics": metrics})

    aggregates = _compute_aggregates(per_question)
    return {"per_question": per_question, "aggregates": aggregates}


def _is_aggregatable_metric(key: str, value) -> bool:
    """True si la key/value es una métrica numérica agregable.

    Skipea keys que empiezan con `_` (convención: debug info, ej. _parsed_output
    y _gt_movies del scorer recsys) y valores no numéricos.
    """
    if key.startswith("_"):
        return False
    return isinstance(value, (int, float))


def _compute_aggregates(per_question: list[dict]) -> dict:
    """Agrega métricas across preguntas, globales y por sub_dataset.

    Keys con prefijo `_` (ej. `_parsed_output`, `_gt_movies` del scorer recsys)
    se ignoran en la agregación — son debug info que viaja con el record pero
    no es una métrica numérica.
    """
    n_total = len(per_question)

    # Agregado global: por cada métrica, promedio solo sobre records donde aparece.
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for q in per_question:
        for k, v in q.get("metrics", {}).items():
            if not _is_aggregatable_metric(k, v):
                continue
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            metric_counts[k] = metric_counts.get(k, 0) + 1

    by_metric = {
        k: {"mean": metric_sums[k] / metric_counts[k], "n": metric_counts[k]}
        for k in metric_sums
    }

    # Agregado por sub_dataset.
    by_sub: dict[str, dict] = {}
    for q in per_question:
        sub = q.get("sub_dataset", "")
        if sub not in by_sub:
            by_sub[sub] = {"_sums": {}, "_counts": {}}
        for k, v in q.get("metrics", {}).items():
            if not _is_aggregatable_metric(k, v):
                continue
            by_sub[sub]["_sums"][k] = by_sub[sub]["_sums"].get(k, 0.0) + float(v)
            by_sub[sub]["_counts"][k] = by_sub[sub]["_counts"].get(k, 0) + 1

    by_sub_clean: dict[str, dict] = {}
    for sub, data in by_sub.items():
        sums = data["_sums"]
        counts = data["_counts"]
        entry: dict = {"n": max(counts.values()) if counts else 0}
        for k in sums:
            entry[k] = {"mean": sums[k] / counts[k], "n": counts[k]}
        by_sub_clean[sub] = entry

    return {
        "n_total": n_total,
        "by_metric": by_metric,
        "by_sub_dataset": by_sub_clean,
    }
