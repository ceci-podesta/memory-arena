"""
Bloque E — Juez LLM sobre D.LRU para VerbatimRAG.
Versión específica para verbatim_rag: busca los JSONL de detective_qa e
infbench_sum generados por VerbatimRAG y corre el juez Mistral sobre ellos.

Uso:
    uv run python scripts/run_e_lru_judge_verbatimrag.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.evaluation.judge import MABAnswerMatchingJudge, MABSummarizationJudge
from memory_arena.evaluation.mab_judgment_runner import run_mab_judgment
from memory_arena.llm.ollama_client import OllamaClient

RESULTS_DIR = REPO_ROOT / "results"
RESPONSES_DIR = RESULTS_DIR / "responses"
JUDGE_MODEL = "mistral:7b"
JUDGE_NAME = "mistral"
STRATEGY = "verbatim_rag"
SUB_DATASETS = ["detective_qa", "infbench_sum_eng_shots2"]


def find_latest_response_jsonl(sub_dataset: str) -> Path | None:
    """Busca el JSONL más reciente de verbatim_rag para el sub_dataset dado."""
    if not RESPONSES_DIR.exists():
        return None
    candidates = [
        p for p in RESPONSES_DIR.glob("*.jsonl")
        if sub_dataset in p.name and STRATEGY in p.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def main() -> None:
    print("=" * 72)
    print(f"Bloque E — Juez LLM sobre D.LRU (strategy={STRATEGY})")
    print(f"Judge model: {JUDGE_MODEL}")
    print("=" * 72)

    shared_llm = OllamaClient(model=JUDGE_MODEL)
    judges_by_sub = {
        "detective_qa": MABAnswerMatchingJudge(llm=shared_llm),
        "infbench_sum_eng_shots2": MABSummarizationJudge(llm=shared_llm),
    }

    summaries = []
    for sub in SUB_DATASETS:
        jsonl = find_latest_response_jsonl(sub)
        if jsonl is None:
            print(f"\n[skip] {sub}: no hay JSONL de verbatim_rag en {RESPONSES_DIR}.")
            continue

        print(f"\n▶ {sub}")
        print(f"  input: {jsonl.relative_to(REPO_ROOT)}")
        result = run_mab_judgment(
            responses_path=jsonl,
            output_dir=RESULTS_DIR,
            judge_name=JUDGE_NAME,
            judges_by_sub=judges_by_sub,
        )
        print(f"  output: {result['output_path']}")
        print(f"  total queries juzgadas: {result['total']}")
        print(f"  tiempo: {format_duration(result['total_latency_s'])}")

        sub_agg = result["by_sub_dataset"].get(sub) or {}
        kind = sub_agg.get("kind")
        if kind == "boolean":
            print(f"  accuracy (juez): {sub_agg.get('accuracy')}  (n={sub_agg.get('n')})")
        elif kind == "structured":
            print(
                f"  fluency={sub_agg.get('mean_fluency')}  "
                f"recall={sub_agg.get('mean_recall')}  "
                f"precision={sub_agg.get('mean_precision')}  "
                f"f1={sub_agg.get('mean_f1')}  (n={sub_agg.get('n')})"
            )
        summaries.append(result)

    summary_path = RESULTS_DIR / "e_lru_judge_verbatimrag_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\nSummary guardado en: {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
