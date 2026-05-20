"""
Reporta accuracy de LongMemEval para una estrategia.

Lee judgments existentes en results/judgments/<run_id>__mistral.jsonl, agrega
accuracy por question_type, imprime tabla y persiste CSV.

Uso:
    uv run python scripts/score_longmemeval.py --strategy summarized
    uv run python scripts/score_longmemeval.py --strategy summarized --run-id 20260519_012652_summarized_longmemeval_oracle
    uv run python scripts/score_longmemeval.py --strategy summarized --subset longmemeval_s_cleaned
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


def find_run_ids(strategy: str, subset: str) -> list[str]:
    """Devuelve run_ids de Fase A para strategy+subset, mas recientes primero."""
    if not RUNS_DIR.exists():
        return []

    runs: list[tuple[str, str]] = []
    for json_path in RUNS_DIR.glob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if data.get("strategy") != strategy:
            continue
        if data.get("benchmark") != subset:
            continue

        run_id = data.get("run_id", "")
        started_at = data.get("started_at") or ""
        if run_id:
            runs.append((started_at, run_id))

    runs.sort(reverse=True)
    return [run_id for _, run_id in runs]


def find_latest_judged_run_id(strategy: str, subset: str) -> str | None:
    """Devuelve el run_id mas reciente que ya tenga judgment Mistral."""
    for run_id in find_run_ids(strategy, subset):
        if (JUDGMENTS_DIR / f"{run_id}__mistral.jsonl").exists():
            return run_id
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="no_memoria")
    parser.add_argument(
        "--subset",
        default="longmemeval_oracle",
        help=(
            "Nombre completo del subset, por ejemplo longmemeval_oracle, "
            "longmemeval_s_cleaned o longmemeval_m_cleaned."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Run_id especifico de Fase A. Si se omite, usa la corrida mas "
            "reciente que ya tenga judgment Mistral."
        ),
    )
    args = parser.parse_args()

    output_csv = (
        REPO_ROOT
        / "results"
        / f"longmemeval_{args.subset}_{args.strategy}_scores.csv"
    )

    run_id = args.run_id or find_latest_judged_run_id(args.strategy, args.subset)
    if run_id is None:
        run_ids = find_run_ids(args.strategy, args.subset)
        if run_ids:
            print(
                f"Hay corridas de Fase A para strategy={args.strategy!r} "
                f"y subset={args.subset!r}, pero ninguna tiene judgment Mistral."
            )
            print("Correr primero el juez sobre el JSONL deseado, por ejemplo:")
            print(
                "  uv run python scripts/run_longmemeval_judge.py "
                f"--responses results/responses/{run_ids[0]}.jsonl"
            )
        else:
            print(
                f"No hay corrida de Fase A para strategy={args.strategy!r} "
                f"y subset={args.subset!r} en {RUNS_DIR}."
            )
            print("Correr primero la generacion, por ejemplo:")
            print(
                "  uv run python scripts/run_longmemeval_summarized.py "
                "--subset oracle --limit 50"
            )
        return 1

    judgment_path = JUDGMENTS_DIR / f"{run_id}__mistral.jsonl"
    if not judgment_path.exists():
        print(f"No hay judgments para run_id={run_id!r} en {judgment_path}.")
        print("El run de Fase A existe pero el juez no corrio todavia.")
        print("Correr:")
        print(
            "  uv run python scripts/run_longmemeval_judge.py "
            f"--responses results/responses/{run_id}.jsonl"
        )
        return 1

    print(f"Leyendo judgments de: {judgment_path.relative_to(REPO_ROOT)}")
    by_type: dict[str, list[bool]] = defaultdict(list)
    with judgment_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question_type = record.get("question_type") or "unknown"
            label = record.get("label")
            if isinstance(label, bool):
                by_type[question_type].append(label)

    total = sum(len(labels) for labels in by_type.values())
    overall = (
        sum(sum(labels) for labels in by_type.values()) / total
        if total
        else 0.0
    )

    print(f"\n=== LongMemEval ({args.subset}, strategy={args.strategy}) ===")
    print(f"run_id: {run_id}")
    print(f"{'question_type':<32} {'accuracy':>10} {'n':>5}")
    print("-" * 50)
    for question_type in sorted(by_type):
        labels = by_type[question_type]
        acc = sum(labels) / len(labels) if labels else 0.0
        print(f"{question_type:<32} {acc:>10.4f} {len(labels):>5}")
    print("-" * 50)
    print(f"{'GLOBAL':<32} {overall:>10.4f} {total:>5}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_type", "accuracy", "n"])
        for question_type in sorted(by_type):
            labels = by_type[question_type]
            acc = sum(labels) / len(labels) if labels else 0.0
            writer.writerow([question_type, f"{acc:.6f}", len(labels)])
        writer.writerow(["GLOBAL", f"{overall:.6f}", total])

    print(f"\nCSV: {output_csv.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
