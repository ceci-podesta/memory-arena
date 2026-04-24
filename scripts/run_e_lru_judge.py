"""
scripts/run_e_lru_judge.py
---------------------------
Bloque E — Corrida del juez LLM (Mistral 7B local) sobre los JSONL de D.LRU
generados por NoMemoria en el Bloque D.LRU.

Patrón:
    Fase A (mab_runner) -> results/responses/<run_id>.jsonl   (ya hecho en D.LRU)
    Fase B (este script) -> results/judgments/<run_id>__mistral.jsonl
    Metadata de la corrida -> results/runs/<run_id_juez>.json
                              (con timestamps, duración, hardware, commit)

Se corre una vez por sub_dataset (2 corridas totales: detective_qa + infbench_sum).
El script elige automáticamente el juez correcto según el sub_dataset (ver
_DEFAULT_JUDGE_FOR_SUB en mab_judgment_runner.py).

Uso:
    uv run python scripts/run_e_lru_judge.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ajuste de path para ejecutar desde la raíz del repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.evaluation.judge import (
    MABAnswerMatchingJudge,
    MABSummarizationJudge,
)
from memory_arena.evaluation.mab_judgment_runner import run_mab_judgment
from memory_arena.llm.ollama_client import OllamaClient


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

RESULTS_DIR = REPO_ROOT / "results"
RESPONSES_DIR = RESULTS_DIR / "responses"

# Modelo del juez. Mistral 7B local (via Ollama).
JUDGE_MODEL = "mistral:7b"

# Nombre del juez usado en el filename de output.
JUDGE_NAME = "mistral"

# Sub_datasets a evaluar. Se matcheará por substring en el nombre del JSONL.
SUB_DATASETS = ["detective_qa", "infbench_sum_eng_shots2"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def find_latest_response_jsonl(sub_dataset: str) -> Path | None:
    """Busca el JSONL más reciente en results/responses/ que matchee el sub_dataset.

    Convención de naming de mab_runner: run_id incluye el sub_dataset sanitizado.
    Matcheamos por substring del sub_dataset.
    """
    if not RESPONSES_DIR.exists():
        return None
    candidates = [
        p for p in RESPONSES_DIR.glob("*.jsonl") if sub_dataset in p.name
    ]
    if not candidates:
        return None
    # El más reciente por mtime.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def format_duration(seconds: float) -> str:
    """Formatea segundos como '12m 34s' o '45.2s' para el output."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print(f"Bloque E — Juez LLM sobre D.LRU")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Judge name:  {JUDGE_NAME}")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 72)

    # Instanciamos los jueces UNA sola vez, compartiendo el OllamaClient (que
    # reutiliza la conexión subyacente y el cache del modelo en Ollama).
    shared_llm = OllamaClient(model=JUDGE_MODEL)
    judges_by_sub = {
        "detective_qa": MABAnswerMatchingJudge(llm=shared_llm),
        "infbench_sum_eng_shots2": MABSummarizationJudge(llm=shared_llm),
    }

    summaries: list[dict] = []

    for sub in SUB_DATASETS:
        jsonl = find_latest_response_jsonl(sub)
        if jsonl is None:
            print(f"\n[skip] {sub}: no hay JSONL de respuestas en {RESPONSES_DIR}.")
            continue

        print(f"\n▶ {sub}")
        print(f"  input:  {jsonl.relative_to(REPO_ROOT)}")
        result = run_mab_judgment(
            responses_path=jsonl,
            output_dir=RESULTS_DIR,
            judge_name=JUDGE_NAME,
            judges_by_sub=judges_by_sub,
        )
        out = Path(result["output_path"])
        runs = Path(result["run_metadata_path"])
        print(f"  output: {out.relative_to(REPO_ROOT)}")
        print(f"  metadata: {runs.relative_to(REPO_ROOT)}")
        print(f"  total queries juzgadas: {result['total']}")
        print(
            f"  tiempo: {format_duration(result['total_latency_s'])} total | "
            f"{result['avg_latency_per_item_s']:.2f}s por item"
        )

        sub_agg = result["by_sub_dataset"].get(sub) or {}
        kind = sub_agg.get("kind")
        if kind == "boolean":
            print(
                f"  accuracy (juez): {sub_agg.get('accuracy')}  "
                f"(n={sub_agg.get('n')})"
            )
        elif kind == "structured":
            print(
                f"  mean fluency={sub_agg.get('mean_fluency')}  "
                f"recall={sub_agg.get('mean_recall')}  "
                f"precision={sub_agg.get('mean_precision')}  "
                f"f1={sub_agg.get('mean_f1')}  (n={sub_agg.get('n')})"
            )
            oor = sub_agg.get("out_of_range_count", 0)
            if oor:
                print(
                    f"  ⚠ out-of-range scores: {oor}/{sub_agg.get('n')} "
                    f"samples con rec/prec fuera de [0,1] capeados"
                )
        summaries.append(result)

    print("\n" + "=" * 72)
    print("Resumen global")
    print("=" * 72)
    total_time = sum(s.get("total_latency_s", 0.0) for s in summaries)
    total_items = sum(s.get("total", 0) for s in summaries)
    print(f"  total items juzgados: {total_items}")
    print(f"  tiempo total:         {format_duration(total_time)}")
    print(f"  promedio por item:    {(total_time / total_items if total_items else 0):.2f}s")
    print()
    for s in summaries:
        print(f"  {s['run_id']}: total={s['total']}  "
              f"tiempo={format_duration(s.get('total_latency_s', 0.0))}")
        print(f"    -> {s['output_path']}")
        print(f"    -> {s['run_metadata_path']}")

    # Dump de summary para que lo lea score_e_lru.py sin recorrer los JSONLs.
    summary_path = RESULTS_DIR / "e_lru_judge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\nSummary dumpeado a: {summary_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
