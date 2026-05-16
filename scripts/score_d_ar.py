"""
Scorea D.AR completo — todos los JSONL de Accurate_Retrieval para una estrategia dada.
Uso: uv run python scripts/score_d_ar.py [--strategy NAME]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from memory_arena.evaluation.mab_scoring import score_jsonl

RUNS_DIR = Path("results/runs")
RESPONSES_DIR = Path("results/responses")
SPLIT = "Accurate_Retrieval"
PREFIX = "d_ar"

METRIC_KEYS = [
    "exact_match",
    "substring_exact_match",
    "f1",
    "rougeL_f1",
    "rougeL_recall",
    "rougeLsum_f1",
    "rougeLsum_recall",
]


def find_run_ids(strategy: str, split: str) -> list[str]:
    if not RUNS_DIR.exists():
        return []
    by_benchmark: dict[str, tuple[str, str]] = {}
    prefix = f"mab_{split}_"
    for json_path in RUNS_DIR.glob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("strategy") != strategy:
            continue
        benchmark = data.get("benchmark", "")
        if not benchmark.startswith(prefix):
            continue
        run_id = data.get("run_id", "")
        started_at = data.get("started_at") or ""
        if not run_id:
            continue
        if benchmark not in by_benchmark or started_at > by_benchmark[benchmark][0]:
            by_benchmark[benchmark] = (started_at, run_id)
    return [rid for _, rid in by_benchmark.values()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="no_memoria")
    args = parser.parse_args()

    output_csv = RUNS_DIR / f"{PREFIX}_{args.strategy}_scores.csv"

    run_ids = find_run_ids(args.strategy, SPLIT)
    if not run_ids:
        raise SystemExit(
            f"No encontré corridas para strategy={args.strategy!r} y "
            f"split={SPLIT!r} en {RUNS_DIR}/*.json"
        )
    print(f"Scoreando {len(run_ids)} corridas de {PREFIX} (strategy={args.strategy})...\n")

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

    header = f"{'sub_dataset':<30} {'n':>5}  " + "  ".join(f"{k:>10}" for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))
    for sub in sorted(per_sub.keys()):
        block = per_sub[sub]
        row = f"{sub:<30} {block['n']:>5}  " + "  ".join(
            f"{block[k]['mean']:>10.4f}" for k in METRIC_KEYS
        )
        print(row)
    print("-" * len(header))
    if total_n > 0:
        global_row = f"{'GLOBAL (weighted)':<30} {total_n:>5}  " + "  ".join(
            f"{weighted_sums[k] / total_n:>10.4f}" for k in METRIC_KEYS
        )
        print(global_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sub_dataset", "n", *METRIC_KEYS])
        for sub in sorted(per_sub.keys()):
            block = per_sub[sub]
            writer.writerow([sub, block["n"], *[block[k]["mean"] for k in METRIC_KEYS]])
        if total_n > 0:
            writer.writerow([
                "GLOBAL (weighted)", total_n,
                *[weighted_sums[k] / total_n for k in METRIC_KEYS],
            ])
    print(f"\nGuardado: {output_csv}")


if __name__ == "__main__":
    main()
