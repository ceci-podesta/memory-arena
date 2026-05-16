"""
Scorea recsys_redial_full con Recall@K (Bloque F).
Uso: uv run python scripts/score_d_ttl_recsys.py [--strategy NAME]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.evaluation.mab_scoring import score_jsonl
from memory_arena.evaluation.recsys_scorer import load_entity2id

RESPONSES_DIR = REPO_ROOT / "results" / "responses"
RUNS_DIR = REPO_ROOT / "results" / "runs"


def find_recsys_jsonl(strategy: str) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    target_bench = "mab_Test_Time_Learning_recsys_redial_full"
    best: tuple[str, str] | None = None
    for jp in RUNS_DIR.glob("*.json"):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("strategy") != strategy:
            continue
        if d.get("benchmark") != target_bench:
            continue
        rid = d.get("run_id", "")
        sa = d.get("started_at") or ""
        if rid and (best is None or sa > best[0]):
            best = (sa, rid)
    if not best:
        return None
    p = RESPONSES_DIR / f"{best[1]}.jsonl"
    return p if p.exists() else None


def check_coverage(jsonl_path: Path) -> dict:
    try:
        name_to_id = load_entity2id()
    except FileNotFoundError:
        return {"total_golds": 0, "found": 0, "coverage_pct": 0.0,
                "error": "entity2id.json no encontrado — correr build_entity2id.py",
                "missing_ids_sample": []}
    known = set(name_to_id.values())
    total, found, missing = 0, 0, []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            golds = rec.get("gold_answers") or []
            if isinstance(golds, str):
                golds = [golds]
            for g in golds:
                if isinstance(g, list):
                    g = g[0] if g else None
                if g is None:
                    continue
                gi = int(g) if isinstance(g, int) else (int(g.strip()) if isinstance(g, str) and g.strip().isdigit() else -1)
                total += 1
                if gi in known:
                    found += 1
                else:
                    missing.append(gi)
    cov = (found/total*100.0) if total else 0.0
    return {"total_golds": total, "found": found, "coverage_pct": round(cov, 2),
            "missing_ids_sample": missing[:10]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="no_memoria")
    args = parser.parse_args()

    output_csv = RUNS_DIR / f"d_ttl_recsys_{args.strategy}_scores.csv"

    jsonl = find_recsys_jsonl(args.strategy)
    if jsonl is None:
        print(f"No hay JSONL de recsys_redial_full para strategy={args.strategy!r}.")
        return 1

    print("="*72)
    print(f"D.TTL recsys — Fase B (scoring, strategy={args.strategy})")
    print("="*72)
    print(f"Input: {jsonl.relative_to(REPO_ROOT)}")

    print("\nCobertura entity2id vs gold IDs:")
    cov = check_coverage(jsonl)
    if "error" in cov:
        print(f"  ERROR: {cov['error']}", file=sys.stderr)
        return 1
    print(f"  Gold IDs totales: {cov['total_golds']}")
    print(f"  Encontrados: {cov['found']}  ({cov['coverage_pct']}%)")
    if cov["coverage_pct"] < 80.0:
        print("  ⚠ Cobertura < 80%: revisar entity2id (sección 10 del informe).")

    print(f"\nScoreando (Recall@1/5/10)...")
    result = score_jsonl(jsonl)
    agg = result["aggregates"]
    by_metric = agg["by_metric"]

    print("\n" + "="*72)
    print(f"Resultados globales (n={agg['n_total']}, strategy={args.strategy})")
    print("="*72)
    for k in ("recsys_recall@1","recsys_recall@5","recsys_recall@10"):
        if k in by_metric:
            print(f"  {k:<25}: {by_metric[k]['mean']:.4f}  (n={by_metric[k]['n']})")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric","mean","n"])
        for k in ("recsys_recall@1","recsys_recall@5","recsys_recall@10","n_gold","n_predicted"):
            if k in by_metric:
                writer.writerow([k, f"{by_metric[k]['mean']:.6f}", by_metric[k]["n"]])
        writer.writerow(["_entity2id_coverage_pct", f"{cov['coverage_pct']:.2f}", cov["total_golds"]])
    print(f"\nCSV: {output_csv.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
