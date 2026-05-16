"""
Bloque E — Corrida del juez LLM (Mistral 7B) sobre los JSONL de D.LRU
de la estrategia indicada.
Uso: uv run python scripts/run_e_lru_judge.py [--strategy NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.evaluation.judge import (
    MABAnswerMatchingJudge,
    MABSummarizationJudge,
)
from memory_arena.evaluation.mab_judgment_runner import run_mab_judgment
from memory_arena.llm.ollama_client import OllamaClient

RESULTS_DIR = REPO_ROOT / "results"
RESPONSES_DIR = RESULTS_DIR / "responses"
RUNS_DIR = RESULTS_DIR / "runs"

JUDGE_MODEL = "mistral:7b"
JUDGE_NAME = "mistral"
SUB_DATASETS = ["detective_qa", "infbench_sum_eng_shots2"]


def find_run_ids_for(strategy: str) -> dict[str, str]:
    if not RUNS_DIR.exists():
        return {}
    by_sub: dict[str, tuple[str, str]] = {}
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
        sub = bench[len(prefix):]
        rid = d.get("run_id", "")
        sa = d.get("started_at") or ""
        if not rid:
            continue
        if sub not in by_sub or sa > by_sub[sub][0]:
            by_sub[sub] = (sa, rid)
    return {s: rid for s, (_, rid) in by_sub.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="no_memoria")
    args = parser.parse_args()

    print("="*72)
    print(f"Bloque E — Juez LLM sobre D.LRU (strategy={args.strategy})")
    print("="*72)

    run_ids = find_run_ids_for(args.strategy)
    if not run_ids:
        print(f"No hay corridas de Fase A LRU para strategy={args.strategy!r}")
        return 1

    shared_llm = OllamaClient(model=JUDGE_MODEL)
    judges = {
        "detective_qa": MABAnswerMatchingJudge(llm=shared_llm),
        "infbench_sum_eng_shots2": MABSummarizationJudge(llm=shared_llm),
    }

    summaries = []
    for sub in SUB_DATASETS:
        rid = run_ids.get(sub)
        if not rid:
            print(f"\n[skip] {sub}: no hay run de Fase A.")
            continue
        jsonl = RESPONSES_DIR / f"{rid}.jsonl"
        if not jsonl.exists():
            print(f"\n[skip] {sub}: JSONL no existe en {jsonl}")
            continue

        print(f"\n▶ {sub}")
        print(f"  input: {jsonl.relative_to(REPO_ROOT)}")
        result = run_mab_judgment(
            responses_path=jsonl, output_dir=RESULTS_DIR,
            judge_name=JUDGE_NAME, judges_by_sub=judges)
        print(f"  total: {result['total']}")
        summaries.append(result)

    summary_path = RESULTS_DIR / f"e_lru_judge_summary_{args.strategy}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\nSummary: {summary_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
