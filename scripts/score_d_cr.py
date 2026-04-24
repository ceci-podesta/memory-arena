"""
Scorea D.CR completo — los 8 JSONL producidos por run_d_cr_nomemoria.py.

Lee los run_ids del log (results/runs/d_cr_nomemoria.log) y scorea cada JSONL.
Imprime tabla por sub_dataset + aggregate global ponderado, y persiste un CSV.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from memory_arena.evaluation.mab_scoring import score_jsonl

LOG_PATH = Path("results/runs/d_cr_nomemoria.log")
RESPONSES_DIR = Path("results/responses")
OUTPUT_CSV = Path("results/runs/d_cr_nomemoria_scores.csv")

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
    """Extrae los run_ids del resumen final del log."""
    text = log_path.read_text(encoding="utf-8")
    # El resumen tiene lineas como:
    #   factconsolidation_sh_6k: <run_id> (0.6 min)
    pattern = re.compile(r"^\s{2}(\S+):\s+(\d{8}_\d{6}_\S+)\s+\(", re.MULTILINE)
    return [m.group(2) for m in pattern.finditer(text)]


def main() -> None:
    run_ids = parse_run_ids_from_log(LOG_PATH)
    if not run_ids:
        raise SystemExit(f"No encontré run_ids en {LOG_PATH}")
    print(f"Scoreando {len(run_ids)} corridas de D.CR...\n")

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
        # El sub_dataset es el único key en by_sub_dataset
        (sub_name, sub_block), = agg["by_sub_dataset"].items()
        per_sub[sub_name] = sub_block

        n = sub_block["n"]
        total_n += n
        for k in METRIC_KEYS:
            weighted_sums[k] += sub_block[k]["mean"] * n

    # Tabla
    header = f"{'sub_dataset':<30} {'n':>5}  " + "  ".join(
        f"{k:>10}" for k in METRIC_KEYS
    )
    print(header)
    print("-" * len(header))

    for sub in sorted(per_sub.keys()):
        block = per_sub[sub]
        row = f"{sub:<30} {block['n']:>5}  " + "  ".join(
            f"{block[k]['mean']:>10.4f}" for k in METRIC_KEYS
        )
        print(row)

    # Global weighted
    print("-" * len(header))
    global_row = f"{'GLOBAL (weighted)':<30} {total_n:>5}  " + "  ".join(
        f"{weighted_sums[k] / total_n:>10.4f}" for k in METRIC_KEYS
    )
    print(global_row)

    # CSV
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


if __name__ == "__main__":
    main()
