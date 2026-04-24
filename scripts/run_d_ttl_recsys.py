"""
scripts/run_d_ttl_recsys.py
----------------------------
Bloque D.TTL (parte recsys) — Fase A: corre NoMemoria sobre
`recsys_redial_full` del split Test_Time_Learning de MAB y escribe el JSONL de
respuestas a `results/responses/`.

Este script completa el set de corridas de D.TTL: los 5 sub-datasets ICL ya
corrieron con `scripts/run_d_ttl_nomemoria.py` (D.TTL base); este archivo es
solo para el sub-dataset recsys, que necesitaba el scorer de Bloque F para
poder ser scoreado. Fase A no depende de F — la Fase A solo genera respuestas.
El scoring (Fase B) vive en `scripts/score_d_ttl_recsys.py`.

Contexto: `recsys_redial_full` es conversational movie recommendation (dataset
ReDial, Li et al. 2018). El contexto por sample es muy largo
(~1.5M chars según el yaml del paper) — con NoMemoria el modelo no ve
ese contexto; las recomendaciones que produzca son parametric y constituyen
el piso de chance.

Uso:
    uv run python scripts/run_d_ttl_recsys.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ajuste de path para ejecutar desde la raíz del repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_TTL,
    SUB_TTL_RECSYS_REDIAL,
    load_mab,
)
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.no_memory import NoMemoria


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

EVAL_MODEL = "llama3.2:3b"
STRATEGY_NAME = "no_memoria"
SPLIT = SPLIT_TTL
SUB_DATASET = SUB_TTL_RECSYS_REDIAL  # "recsys_redial_full"

# None = correr todos los samples del sub-dataset. Si querés smoke test con 1,
# poné 1 acá.
MAX_SAMPLES: int | None = None

RESULTS_DIR = REPO_ROOT / "results"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print(f"D.TTL recsys — Fase A (NoMemoria sobre {SUB_DATASET})")
    print(f"Model: {EVAL_MODEL}")
    print(f"Split: {SPLIT}")
    print(f"Results dir: {RESULTS_DIR}")
    if MAX_SAMPLES is not None:
        print(f"⚠ max_samples={MAX_SAMPLES} (smoke test)")
    print("=" * 72)

    print(f"\nCargando samples de {SUB_DATASET}...")
    samples = load_mab(
        split=SPLIT,
        sub_dataset=SUB_DATASET,
        max_samples=MAX_SAMPLES,
    )
    print(f"  samples cargados: {len(samples)}")
    total_questions = sum(len(s.questions) for s in samples)
    print(f"  preguntas totales: {total_questions}")

    # Chequeo de sanity sobre gold: para recsys esperamos IDs enteros.
    if samples:
        first = samples[0]
        print(
            f"\nSample #0: sample_id={first.sample_id}  "
            f"n_questions={len(first.questions)}"
        )
        print(f"  gold_answers[0]: {first.answers[0]}")
        print(f"  question[0] (primeros 160 chars): {first.questions[0][:160]!r}")

    print(f"\nInstanciando NoMemoria + Ollama({EVAL_MODEL})...")
    strategy = NoMemoria()
    llm = OllamaClient(model=EVAL_MODEL)

    print(f"\nArrancando run...")
    metadata = run_strategy_mab(
        strategy=strategy,
        samples=samples,
        llm=llm,
        strategy_name=STRATEGY_NAME,
        split=SPLIT,
        sub_dataset=SUB_DATASET,
        output_dir=RESULTS_DIR,
    )
    print(f"\nRun completado.")
    print(f"  run_id: {metadata.run_id}")
    print(f"  duración: {metadata.duration_seconds}s")
    print(f"  metadata: {RESULTS_DIR / 'runs' / (metadata.run_id + '.json')}")
    print(f"  responses: {RESULTS_DIR / 'responses' / (metadata.run_id + '.jsonl')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
