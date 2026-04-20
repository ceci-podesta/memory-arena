"""
memory_arena.benchmarks.longmemeval
------------------------------------
Loader para el benchmark LongMemEval.

Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Paper:   Wu et al. 2024, "LongMemEval: Benchmarking Chat Assistants on Long-Term
         Interactive Memory".

Estructura del sample:
- Cada sample es una pregunta con una respuesta correcta.
- Cada sample tiene un "haystack" de sesiones historicas.
- La estrategia de memoria consume las sesiones del haystack (via store()) y
  despues tiene que responder la pregunta (via retrieve() + LLM).

Nota de implementacion:
    No usamos datasets.load_dataset() porque intenta construir cache Arrow de
    los 3 splits antes de devolver el pedido, y el split m_cleaned (2.7 GB)
    hace overflow en pyarrow. En su lugar, descargamos el JSON especifico con
    hf_hub_download y lo parseamos con json.load.
"""

import json
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

from memory_arena.memories.base import Turn


# ------------------------------------------------------------
# Constantes del dataset
# ------------------------------------------------------------

DATASET_NAME = "xiaowu0162/longmemeval-cleaned"

# Nombres de los splits disponibles.
SUBSET_ORACLE = "longmemeval_oracle"          # ~15 MB  — solo sesiones con evidencia
SUBSET_S_CLEANED = "longmemeval_s_cleaned"    # ~277 MB — short haystack (default del paper)
SUBSET_M_CLEANED = "longmemeval_m_cleaned"    # ~2.7 GB — medium haystack

VALID_SUBSETS = frozenset({SUBSET_ORACLE, SUBSET_S_CLEANED, SUBSET_M_CLEANED})


# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------

@dataclass
class Session:
    """Una sesion historica del haystack: conversacion pasada con fecha."""
    session_id: str
    date: str              # formato raw: "2023/05/30 (Tue) 23:40"
    turns: list[Turn]


@dataclass
class LongMemEvalSample:
    """Un sample de LongMemEval."""
    question_id: str
    question: str
    question_type: str
    question_date: str
    expected_answer: str
    haystack: list[Session]
    answer_session_ids: list[str]


# ------------------------------------------------------------
# Loader
# ------------------------------------------------------------

def load_longmemeval(
    subset: str,
    limit: int | None = None,
) -> list[LongMemEvalSample]:
    """Cargar samples de LongMemEval desde HuggingFace.

    Args:
        subset: cual split cargar (usar constantes SUBSET_*).
        limit: si se pasa, limita al primer N samples (util para smoke tests).

    Returns:
        Lista de LongMemEvalSample.

    Raises:
        ValueError: si subset no es uno de los splits validos.
    """
    if subset not in VALID_SUBSETS:
        raise ValueError(
            f"subset invalido: {subset!r}. "
            f"Opciones validas: {sorted(VALID_SUBSETS)}"
        )

    # Descargamos SOLO el archivo JSON del split pedido.
    # hf_hub_download cachea en ~/.cache/huggingface/hub/ — si ya esta
    # descargado, reusa sin re-downloadar.
    local_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=f"{subset}.json",
        repo_type="dataset",
    )

    with open(local_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples: list[LongMemEvalSample] = []
    for i, raw in enumerate(raw_data):
        if limit is not None and i >= limit:
            break
        samples.append(_parse_sample(raw))

    return samples


def _parse_sample(raw: dict) -> LongMemEvalSample:
    """Convertir un sample crudo del dataset a LongMemEvalSample."""
    sessions: list[Session] = []
    for session_id, date, turns_raw in zip(
        raw["haystack_session_ids"],
        raw["haystack_dates"],
        raw["haystack_sessions"],
        strict=True,
    ):
        turns = [Turn(role=t["role"], content=t["content"]) for t in turns_raw]
        sessions.append(Session(session_id=session_id, date=date, turns=turns))

    return LongMemEvalSample(
        question_id=raw["question_id"],
        question=raw["question"],
        question_type=raw["question_type"],
        question_date=raw["question_date"],
        expected_answer=raw["answer"],
        haystack=sessions,
        answer_session_ids=raw["answer_session_ids"],
    )

