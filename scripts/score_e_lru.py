"""
Scorea D.LRU con el juez LLM (Mistral 7B) — Bloque E.
Lee los JSONL de results/judgments/ filtrados por strategy y reporta accuracy.
Uso: uv run python scripts/score_e_lru.py [--strategy NAME]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
JUDGMENTS_DIR = REPO_ROOT / "results" / "judgments"


def find_run_ids_for(strategy: str) -> list[str]:
    if not RUNS_DIR.exists():
        return []
    by_benchmark: dict[str, tuple[str, str]] = {}
    prefix = "mab_Long_Range_Understanding_"
    for jp in RUNS_DIR.glob("*.json"):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("strategy") != strategy:
            continue
        bench = d.get("benchmark", "")
        if not bench.startswith(prefix):
            continue
        rid = d.get("run_id", "")
        sa = d.get("started_at") or ""
        if not rid:
            continue
        if bench not in by_benchmark or sa > by_benchmark[bench][0]:
            by_benchmark[bench] = (sa, rid)
    return [r for _, r in by_benchmark.values()]


def find_judgment_jsonls(run_ids: list[str], judge_suffix: str = "__mistral.jsonl") -> list[Path]:
    if not JUDGMENTS_DIR.exists():
        return []
    return [p for rid in run_ids
            if (p := JUDGMENTS_DIR / f"{rid}{judge_suffix}").exists()]


def aggregate_judgment_jsonl(path: Path) -> list[dict]:
    per_sub_bool: dict[str, list[bool]] = defaultdict(list)
    per_sub_scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sub = rec.get("sub_dataset") or ""
            if rec.get("label") is not None:
                per_sub_bool[sub].append(bool(rec["label"]))
            if rec.get("scores") is not None:
                per_sub_scores[sub].append(rec["scores"])
    rows: list[dict] = []
    for sub in sorted(set(per_sub_bool) | set(per_sub_scores)):
        if per_sub_bool.get(sub):
            labs = per_sub_bool[sub]
            rows.append({"sub_dataset": sub, "n": len(labs), "kind": "boolean",
                         "accuracy": round(sum(labs)/len(labs), 4),
                         "mean_fluency": None, "mean_recall": None,
                         "mean_precision": None, "mean_f1": None,
                         "source_path": str(path)})
        if per_sub_scores.get(sub):
            sl = per_sub_scores[sub]; n = len(sl)
            means = {k: round(sum(s.get(k, 0.0) for s in sl)/n, 4)
                     for k in ("fluency","recall","precision","f1")}
            rows.append({"sub_dataset": sub, "n": n, "kind": "structured",
                         "accuracy": None,
                         "mean_fluency": means["fluency"], "mean_recall": means["recall"],
                         "mean_precision": means["precision"], "mean_f1": means["f1"],
                         "source_path": str(path)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="no_memoria")
    args = parser.parse_args()

    output_csv = REPO_ROOT / "results" / f"d_lru_{args.strategy}_judge_scores.csv"

    run_ids = find_run_ids_for(args.strategy)
    if not run_ids:
        print(f"No hay corridas de Fase A LRU para strategy={args.strategy!r}")
        return 1

    jsonls = find_judgment_jsonls(run_ids)
    if not jsonls:
        print(f"No hay judgments en {JUDGMENTS_DIR} para strategy={args.strategy!r}.")
        print(f"Correr scripts/run_e_lru_judge.py --strategy {args.strategy} primero.")
        return 1

    print(f"Encontrados {len(jsonls)} JSONL de judgments (strategy={args.strategy}):")
    for p in jsonls:
        print(f"  - {p.relative_to(REPO_ROOT)}")

    all_rows: list[dict] = []
    for p in jsonls:
        all_rows.extend(aggregate_judgment_jsonl(p))

    if not all_rows:
        print("No se agregaron métricas (JSONLs vacíos?)")
        return 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sub_dataset","n","kind","accuracy",
            "mean_fluency","mean_recall","mean_precision","mean_f1","source_path"])
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"\nCSV: {output_csv.relative_to(REPO_ROOT)}")

    print("\n" + "="*88)
    print(f"Bloque E — Juez LLM (Mistral) sobre D.LRU (strategy={args.strategy})")
    print("="*88)
    hdr = f"{'sub_dataset':<32} {'n':>5}  {'kind':<11}  {'accuracy':>9}  {'fluency':>8}  {'recall':>7}  {'precis':>7}  {'f1':>7}"
    print(hdr); print("-"*88)
    for r in all_rows:
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "   -   "
        flu = f"{r['mean_fluency']:.4f}" if r.get("mean_fluency") is not None else "   -  "
        rec = f"{r['mean_recall']:.4f}" if r.get("mean_recall") is not None else "   -  "
        pre = f"{r['mean_precision']:.4f}" if r.get("mean_precision") is not None else "   -  "
        f1v = f"{r['mean_f1']:.4f}" if r.get("mean_f1") is not None else "   -  "
        print(f"{r['sub_dataset']:<32} {r['n']:>5}  {r['kind']:<11}  {acc:>9}  {flu:>8}  {rec:>7}  {pre:>7}  {f1v:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
