"""
scripts/score_e_lru.py
-----------------------
Bloque E — Agrega y reporta los verdicts del juez LLM sobre D.LRU.

Lee los JSONL de `results/judgments/` generados por run_e_lru_judge.py y
produce:
  1. Una tabla resumen por sub_dataset (stdout).
  2. Un CSV `results/d_lru_judge_scores.csv` para análisis posterior.

Semántica de las métricas reportadas:
  - detective_qa (MCQA)          -> juez booleano, reporta `accuracy` (fracción de
                                     respuestas donde el juez dijo "yes").
  - infbench_sum_eng_shots2     -> juez estructurado, reporta mean de fluency /
                                     recall / precision / F1 compuesto.

Comparación contra el scorer default (Bloque D.LRU, sección 8 de notas-informe-tp.md):
  Se imprime también, al final, una tabla paralela con los números default (EM / F1)
  para facilitar interpretar el delta que trae el juez. Los números default salen
  de `d_lru_nomemoria_scores.csv` que genera score_d_lru.py. Ese script escribe
  el CSV a `results/runs/d_lru_nomemoria_scores.csv` (convención histórica); acá
  buscamos en ambos paths (raíz y `runs/`) para ser robustos.

Uso:
    uv run python scripts/run_e_lru_judge.py     # primero
    uv run python scripts/score_e_lru.py          # después
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
JUDGMENTS_DIR = RESULTS_DIR / "judgments"

# Paths donde puede estar el CSV de scores default de D.LRU. Los probamos en
# orden (primero el que usa score_d_lru.py, después la raíz por si alguien lo
# mueve).
DEFAULT_SCORES_CANDIDATES = [
    RESULTS_DIR / "runs" / "d_lru_nomemoria_scores.csv",
    RESULTS_DIR / "d_lru_nomemoria_scores.csv",
]

OUTPUT_CSV = RESULTS_DIR / "d_lru_judge_scores.csv"


# ----------------------------------------------------------------------------
# Carga de judgments
# ----------------------------------------------------------------------------


def find_judgment_jsonls(judge_suffix: str = "__mistral.jsonl") -> list[Path]:
    """Devuelve los JSONL de judgments del juez indicado, solo de LRU."""
    if not JUDGMENTS_DIR.exists():
        return []
    return sorted(
        [p for p in JUDGMENTS_DIR.glob(f"*{judge_suffix}")
         if "Long_Range_Understanding" in p.name]
    )


def aggregate_judgment_jsonl(path: Path) -> list[dict]:
    """Agrega un JSONL de judgments en métricas por sub_dataset.

    Returns:
        Lista de dicts (uno por sub_dataset encontrado en el JSONL). Cada dict
        contiene:
            {
                "sub_dataset": str,
                "n": int,
                "kind": "boolean" | "structured",
                "accuracy": float | None,            # solo si booleano
                "mean_fluency": float | None,        # solo si structured
                "mean_recall": float | None,
                "mean_precision": float | None,
                "mean_f1": float | None,
                "source_path": str,
            }
    """
    per_sub_bool: dict[str, list[bool]] = defaultdict(list)
    per_sub_scores: dict[str, list[dict[str, float]]] = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sub = rec.get("sub_dataset") or ""
            label = rec.get("label")
            scores = rec.get("scores")
            if label is not None:
                per_sub_bool[sub].append(bool(label))
            if scores is not None:
                per_sub_scores[sub].append(scores)

    rows: list[dict] = []
    all_subs = set(per_sub_bool.keys()) | set(per_sub_scores.keys())
    for sub in sorted(all_subs):
        if per_sub_bool.get(sub):
            labels = per_sub_bool[sub]
            rows.append({
                "sub_dataset": sub,
                "n": len(labels),
                "kind": "boolean",
                "accuracy": round(sum(labels) / len(labels), 4),
                "mean_fluency": None,
                "mean_recall": None,
                "mean_precision": None,
                "mean_f1": None,
                "source_path": str(path),
            })
        if per_sub_scores.get(sub):
            slist = per_sub_scores[sub]
            n = len(slist)
            means = {
                k: round(sum(s.get(k, 0.0) for s in slist) / n, 4)
                for k in ("fluency", "recall", "precision", "f1")
            }
            rows.append({
                "sub_dataset": sub,
                "n": n,
                "kind": "structured",
                "accuracy": None,
                "mean_fluency": means["fluency"],
                "mean_recall": means["recall"],
                "mean_precision": means["precision"],
                "mean_f1": means["f1"],
                "source_path": str(path),
            })
    return rows


# ----------------------------------------------------------------------------
# Merge con scores default (Bloque D.LRU)
# ----------------------------------------------------------------------------


def find_default_scores_csv() -> Path | None:
    """Busca el CSV de scores default de D.LRU en los paths conocidos."""
    for cand in DEFAULT_SCORES_CANDIDATES:
        if cand.exists():
            return cand
    return None


def load_default_scores_for_lru() -> tuple[dict[str, dict[str, float]], Path | None]:
    """Lee el CSV de default scores y devuelve
    ({sub_dataset: {"n": n, "exact_match": float, "f1": float}}, csv_path_usada)."""
    out: dict[str, dict[str, float]] = {}
    csv_path = find_default_scores_csv()
    if csv_path is None:
        return out, None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row.get("sub_dataset") or ""
            if not sub:
                continue
            try:
                out[sub] = {
                    "n": int(row.get("n") or 0),
                    "exact_match": float(row.get("exact_match") or 0),
                    "f1": float(row.get("f1") or 0),
                }
            except (TypeError, ValueError):
                continue
    return out, csv_path


# ----------------------------------------------------------------------------
# Reporte
# ----------------------------------------------------------------------------


def print_judge_table(all_rows: list[dict]) -> None:
    """Imprime tabla 1: resultados del juez, por sub_dataset."""
    print("\n" + "=" * 88)
    print("Bloque E — Juez LLM (Mistral 7B) sobre D.LRU NoMemoria")
    print("=" * 88)
    header = (
        f"{'sub_dataset':<32} {'n':>5}  {'kind':<11}  "
        f"{'accuracy':>9}  {'fluency':>8}  {'recall':>7}  {'precis':>7}  {'f1':>7}"
    )
    print(header)
    print("-" * 88)
    for r in all_rows:
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "   -   "
        flu = f"{r['mean_fluency']:.4f}" if r.get("mean_fluency") is not None else "   -  "
        rec = f"{r['mean_recall']:.4f}" if r.get("mean_recall") is not None else "   -  "
        pre = f"{r['mean_precision']:.4f}" if r.get("mean_precision") is not None else "   -  "
        f1v = f"{r['mean_f1']:.4f}" if r.get("mean_f1") is not None else "   -  "
        print(
            f"{r['sub_dataset']:<32} {r['n']:>5}  {r['kind']:<11}  "
            f"{acc:>9}  {flu:>8}  {rec:>7}  {pre:>7}  {f1v:>7}"
        )


def print_comparison_table(
    all_rows: list[dict],
    default_scores: dict,
    default_csv_path: Path | None,
) -> None:
    """Imprime tabla 2: comparación judge vs default por sub_dataset."""
    if not default_scores:
        print("\n(nota: no se encontró el CSV de default scores de D.LRU en "
              f"{[str(p) for p in DEFAULT_SCORES_CANDIDATES]}; "
              "omitiendo comparación con default)")
        return

    print("\n" + "=" * 88)
    print(f"Comparación: juez LLM vs. scorer default (por sub_dataset)")
    print(f"Default scores file: {default_csv_path}")
    print("=" * 88)
    header = (
        f"{'sub_dataset':<32} {'n':>5}  "
        f"{'default_EM':>11}  {'default_F1':>11}  "
        f"{'judge_metric':>13}  {'judge_value':>11}"
    )
    print(header)
    print("-" * 88)
    for r in all_rows:
        sub = r["sub_dataset"]
        d = default_scores.get(sub, {})
        default_em = f"{d.get('exact_match', 0):.4f}" if d else "   -   "
        default_f1 = f"{d.get('f1', 0):.4f}" if d else "   -   "

        if r["kind"] == "boolean":
            judge_metric = "accuracy"
            judge_value = f"{r['accuracy']:.4f}"
        else:
            judge_metric = "f1 (composite)"
            judge_value = f"{r['mean_f1']:.4f}"

        print(
            f"{sub:<32} {r['n']:>5}  "
            f"{default_em:>11}  {default_f1:>11}  "
            f"{judge_metric:>13}  {judge_value:>11}"
        )

    print("\n"
          "NOTA: para detective_qa el piso de chance es 0.25 (MCQA 4-opciones).\n"
          "      Cualquier valor < 0.25 en el juez indica performance subrandom.\n"
          "      Para infbench_sum el default F1 es léxico (ROUGE-based); el juez F1\n"
          "      es compuesto (fluency × 2·rec·prec / (rec+prec)) y NO son directamente\n"
          "      comparables — se muestran en paralelo como referencia.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    jsonls = find_judgment_jsonls()
    if not jsonls:
        print(f"No hay JSONLs de judgments en {JUDGMENTS_DIR}. Correr run_e_lru_judge.py primero.")
        return 1

    print(f"Encontrados {len(jsonls)} JSONL de judgments:")
    for p in jsonls:
        print(f"  - {p.relative_to(REPO_ROOT)}")

    all_rows: list[dict] = []
    for p in jsonls:
        rows = aggregate_judgment_jsonl(p)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("No se pudieron agregar métricas (¿JSONLs vacíos?).")
        return 1

    # Dump CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sub_dataset", "n", "kind",
                "accuracy",
                "mean_fluency", "mean_recall", "mean_precision", "mean_f1",
                "source_path",
            ],
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"\nCSV dumpeado a: {OUTPUT_CSV.relative_to(REPO_ROOT)}")

    # Tabla juez
    print_judge_table(all_rows)

    # Tabla comparación
    default_scores, default_csv_path = load_default_scores_for_lru()
    print_comparison_table(all_rows, default_scores, default_csv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
