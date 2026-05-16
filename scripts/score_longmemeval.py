"""
Reporta accuracy de LongMemEval para una estrategia (lee judgments existentes).

Uso:
    uv run python scripts/score_longmemeval.py
    uv run python scripts/score_longmemeval.py --strategy graph_memory
    uv run python scripts/score_longmemeval.py --subset s

Escanea results/runs/*.json filtrando por strategy + subset, agarra el JSONL
de judgments correspondiente (results/judgments/<run_id>__mistral.jsonl),
agrega accuracy por question_type, imprime tabla y persiste CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
JUDGMENTS_DIR = REPO_ROOT / "results" / "judgments"


def find_latest_run_id(strategy: str, subset: str) -> str | None:
    """Devuelve el run_id más reciente de Fase A para strategy+subset."""
    if not RUNS_DIR.exists():
        return None
    best: tuple[str, str] | None = None
    for jp in RUNS_DIR.glob("*.json"):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("strategy") != strategy:
            continue
        if d.get("benchmark") != subset:
            continue
        rid = d.get("run_id", "")
        sa = d.get("started_at") or ""
        if rid and (best is None or sa > best[0]):
            best = (sa, rid)
    return best[1] if best else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="no_memoria")
    parser.add_argument("--subset", default="longmemeval_oracle",
                        help="Nombre completo del subset (ej longmemeval_oracle, "
                             "longmemeval_s_cleaned, longmemeval_m_cleaned)")
    args = parser.parse_args()

    output_csv = REPO_ROOT / "results" / f"longmemeval_{args.subset}_{args.strategy}_scores.csv"

    rid = find_latest_run_id(args.strategy, args.subset)
    if rid is None:
        print(f"No hay corrida de Fase A para strategy={args.strategy!r} "
              f"y subset={args.subset!r} en {RUNS_DIR}.")
        print(f"Correr primero: uv run python scripts/run_longmemeval.py "
              f"--strategy {args.strategy} --subset oracle")
        return 1

    judgment_path = JUDGMENTS_DIR / f"{rid}__mistral.jsonl"
    if not judgment_path.exists():
        print(f"No hay judgments para run_id={rid!r} en {judgment_path}.")
        print(f"El run de Fase A existe pero el juez no corrió todavía. "
              f"Re-correr con: uv run python scripts/run_longmemeval.py "
              f"--strategy {args.strategy} --subset {args.subset.replace('longmemeval_','').replace('_cleaned','')}")
        return 1

    print(f"Leyendo judgments de: {judgment_path.relative_to(REPO_ROOT)}")
    by_type: dict[str, list[bool]] = defaultdict(list)
    with open(judgment_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qt = rec.get("question_type") or "unknown"
            lab = rec.get("label")
            if isinstance(lab, bool):
                by_type[qt].append(lab)

    total = sum(len(v) for v in by_type.values())
    overall = (sum(sum(v) for v in by_type.values()) / total) if total else 0.0

    print(f"\n=== LongMemEval ({args.subset}, strategy={args.strategy}) ===")
    print(f"{'question_type':<32} {'accuracy':>10} {'n':>5}")
    print("-" * 50)
    for qt in sorted(by_type):
        labs = by_type[qt]
        acc = sum(labs) / len(labs) if labs else 0.0
        print(f"{qt:<32} {acc:>10.4f} {len(labs):>5}")
    print("-" * 50)
    print(f"{'GLOBAL':<32} {overall:>10.4f} {total:>5}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_type", "accuracy", "n"])
        for qt in sorted(by_type):
            labs = by_type[qt]
            acc = sum(labs) / len(labs) if labs else 0.0
            w.writerow([qt, f"{acc:.6f}", len(labs)])
        w.writerow(["GLOBAL", f"{overall:.6f}", total])
    print(f"\nCSV: {output_csv.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
