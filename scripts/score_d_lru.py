"""
Scorea D.LRU con metricas default (piso lexico — NO es la evaluacion final).

Lee los run_ids de results/runs/d_lru_nomemoria.log y scorea cada JSONL con
las 7 metricas default del paper. Util como sanity check y como baseline
para comparar despues con el juez LLM Mistral (Bloque E).

LRU son tareas generativas de respuesta larga:
  - detective_qa: QA narrativo sobre novelas.
  - infbench_sum_eng_shots2: resumen de documentos largos.

Las metricas de overlap (EM, F1, ROUGE) sobre texto largo correlacionan mal
con calidad real. El juez es el que va a dar la medida final.
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from memory_arena.evaluation.mab_scoring import score_jsonl
LOG_PATH = Path("results/runs/d_lru_nomemoria.log")
RESPONSES_DIR = Path("results/responses")
OUTPUT_CSV = Path("results/runs/d_lru_nomemoria_scores.csv")
METRIC_KEYS = [
    "exact_match",
    "substring_exact_match",
    "f1",
    "rougeL_f1",
    "rougeL_recall",
    "rougeLsum_f1",
    "rougeLsum_recall",
]
def parse_run_ids_from_log(log_path: Path) -> list[str]:
    text = log_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s{2}(\S+):\s+(\d{8}_\d{6}_\S+)", re.MULTILINE)
    return [m.group(2) for m in pattern.finditer(text)]
def main() -> None:
    run_ids = parse_run_ids_from_log(LOG_PATH)
    if not run_ids:
        raise SystemExit(f"No encontre run_ids en {LOG_PATH}")
    print(f"Scoreando {len(run_ids)} corridas de D.LRU (default metrics — piso lexico)...\n")
    per_sub: dict[str, dict] = {}
    total_n = 0
    weighted_sums = {k: 0.0 for k in METRIC_KEYS}
    for run_id in run_ids:
        jsonl_path = RESPONSES_DIR / f"{run_id}.jsonl"
        if not jsonl_path.exists():
            print(f"  MISSING: {jsonl_path}")
            continue
        result = score_jsonl(jsonl_path)
        agg = result["aggregates"]
        (sub_name, sub_block), = agg["by_sub_dataset"].items()
        per_sub[sub_name] = sub_block
        n = sub_block["n"]
        total_n += n
        for k in METRIC_KEYS:
            weighted_sums[k] += sub_block[k]["mean"] * n
    header = f"{'sub_dataset':<32} {'n':>5}  " + "  ".join(
        f"{k:>10}" for k in METRIC_KEYS
    )
    print(header)
    print("-" * len(header))
    for sub in sorted(per_sub.keys()):
        block = per_sub[sub]
        row = f"{sub:<32} {block['n']:>5}  " + "  ".join(
            f"{block[k]['mean']:>10.4f}" for k in METRIC_KEYS
        )
        print(row)
    print("-" * len(header))
    global_row = f"{'GLOBAL (weighted)':<32} {total_n:>5}  " + "  ".join(
        f"{weighted_sums[k] / total_n:>10.4f}" for k in METRIC_KEYS
    )
    print(global_row)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sub_dataset", "n", *METRIC_KEYS])
        for sub in sorted(per_sub.keys()):
            block = per_sub[sub]
            writer.writerow(
                [sub, block["n"], *[block[k]["mean"] for k in METRIC_KEYS]]
            )
        writer.writerow(
            [
                "GLOBAL (weighted)",
                total_n,
                *[weighted_sums[k] / total_n for k in METRIC_KEYS],
            ]
        )
    print(f"\nGuardado: {OUTPUT_CSV}")
    print("\nNOTA: estos numeros son piso lexico, no la evaluacion final.")
    print("La medida correcta sale con el juez LLM Mistral (Bloque E).")
if __name__ == "__main__":
    main()
