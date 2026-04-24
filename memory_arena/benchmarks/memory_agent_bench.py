"""
Loader para MemoryAgentBench (Hu et al., ICLR 2026).

Paper: https://arxiv.org/abs/2507.05257
Repo oficial: https://github.com/HUST-AI-HYZ/MemoryAgentBench
Dataset HF: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench

Estructura del benchmark:
- 4 splits (competencias):
    - Accurate_Retrieval (AR): preguntas cortas sobre haystack largo.
        Sub-datasets: longmemeval_s, longmemeval_s_star, eventqa_*, ruler_*
    - Test_Time_Learning (TTL): el agente debe aprender del contexto (ICL).
        Sub-datasets: icl_banking77, icl_clinic150, icl_nlu, icl_trec_*, recsys_redial_*
    - Long_Range_Understanding (LRU): respuestas generativas largas.
        Sub-datasets: detective_qa, infbench_sum
    - Conflict_Resolution (CR): resolver contradicciones en el contexto.
        Sub-datasets: factconsolidation_sh_*, factconsolidation_mh_*

- Filosofía "inject once, query multiple times": un contexto largo ->
  múltiples preguntas. La estrategia de memoria ingesta el contexto una
  única vez y después responde todas las preguntas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset


# ----------------------------------------------------------------------------
# Splits (competencias)
# ----------------------------------------------------------------------------

SPLIT_AR = "Accurate_Retrieval"
SPLIT_TTL = "Test_Time_Learning"
SPLIT_LRU = "Long_Range_Understanding"
SPLIT_CR = "Conflict_Resolution"

ALL_SPLITS: frozenset[str] = frozenset({SPLIT_AR, SPLIT_TTL, SPLIT_LRU, SPLIT_CR})


# ----------------------------------------------------------------------------
# Sub-datasets más usados (atajos).
# Los nombres siguen la convención de metadata.source en el dataset HF
# ('ai-hyz/MemoryAgentBench'), que puede diferir de los yaml del repo oficial.
# Fuente: diagnóstico de Counter(metadata.source) sobre los 4 splits (22+6+110+8 rows).
# ----------------------------------------------------------------------------

# AR (Accurate_Retrieval) — 22 rows totales
SUB_AR_LONGMEMEVAL_S = "longmemeval_s*"   # 5 rows (ojo: asterisco literal)
SUB_AR_EVENTQA_FULL = "eventqa_full"
SUB_AR_EVENTQA_65K = "eventqa_65536"
SUB_AR_EVENTQA_131K = "eventqa_131072"
SUB_AR_RULER_QA1 = "ruler_qa1_197K"
SUB_AR_RULER_QA2 = "ruler_qa2_421K"

# TTL (Test_Time_Learning) — 6 rows totales
SUB_TTL_RECSYS_REDIAL = "recsys_redial_full"
SUB_TTL_ICL_BANKING77 = "icl_banking77_5900shot_balance"
SUB_TTL_ICL_CLINIC150 = "icl_clinic150_7050shot_balance"
SUB_TTL_ICL_NLU = "icl_nlu_8296shot_balance"
SUB_TTL_ICL_TREC_COARSE = "icl_trec_coarse_6600shot_balance"
SUB_TTL_ICL_TREC_FINE = "icl_trec_fine_6400shot_balance"

# LRU (Long_Range_Understanding) — 110 rows totales
SUB_LRU_DETECTIVE_QA = "detective_qa"                 # 10 rows
SUB_LRU_INFBENCH_SUM = "infbench_sum_eng_shots2"      # 100 rows

# CR (Conflict_Resolution) — 8 rows totales
# sh = single-hop, mh = multi-hop. Los números son context_max_length.
SUB_CR_FACTCONSOL_SH_6K = "factconsolidation_sh_6k"
SUB_CR_FACTCONSOL_SH_32K = "factconsolidation_sh_32k"
SUB_CR_FACTCONSOL_SH_64K = "factconsolidation_sh_64k"
SUB_CR_FACTCONSOL_SH_262K = "factconsolidation_sh_262k"
SUB_CR_FACTCONSOL_MH_6K = "factconsolidation_mh_6k"
SUB_CR_FACTCONSOL_MH_32K = "factconsolidation_mh_32k"
SUB_CR_FACTCONSOL_MH_64K = "factconsolidation_mh_64k"
SUB_CR_FACTCONSOL_MH_262K = "factconsolidation_mh_262k"


# ----------------------------------------------------------------------------
# Dataclass del sample
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MABSample:
    """Un sample de MemoryAgentBench: un contexto largo + N preguntas sobre él.

    Attributes:
        sample_id: identificador único dentro del sub-dataset. Derivado del
            índice de la fila en HF (ya que el dataset no siempre trae un
            campo ``id`` propio).
        source: el nombre del sub-dataset (ej: 'longmemeval_s_-1_500'). Igual
            al valor de metadata.source de HF, conservado para trazabilidad
            en los JSONL de salida.
        context: el texto largo a "inyectar" en la memoria de la estrategia.
        questions: lista de N preguntas sobre el contexto.
        answers: lista paralela a ``questions``. Cada elemento es una lista
            de respuestas aceptables (el benchmark a veces acepta múltiples
            formulaciones equivalentes; seguimos la convención del repo
            oficial de tomar max_score sobre todas las ground truths).
        question_ids: lista paralela a ``questions``. Si el dataset no trae
            qa_pair_ids los sintetizamos como f"{sample_id}_q{idx}".
    """

    sample_id: str
    source: str
    context: str
    questions: list[str]
    answers: list[list[str]]
    question_ids: list[str]

    def __post_init__(self) -> None:
        n = len(self.questions)
        if not (n == len(self.answers) == len(self.question_ids)):
            raise ValueError(
                "questions, answers y question_ids deben tener la misma longitud "
                f"(got {n} / {len(self.answers)} / {len(self.question_ids)})"
            )


# ----------------------------------------------------------------------------
# Loader público
# ----------------------------------------------------------------------------


def load_mab(
    split: str,
    sub_dataset: str,
    max_samples: int | None = None,
) -> list[MABSample]:
    """Carga un sub-dataset de MemoryAgentBench desde HuggingFace.

    Args:
        split: una de las 4 competencias. Debe ser una de ``ALL_SPLITS``.
        sub_dataset: el identificador del sub-dataset (valor de
            ``metadata.source`` en el dataset). Ej: ``'longmemeval_s_-1_500'``.
        max_samples: si no es None, trunca el dataset a los primeros N samples
            *después* de filtrar por source. Útil para smoke tests.

    Returns:
        Lista de ``MABSample``.

    Raises:
        ValueError: si ``split`` no es una competencia válida o si el filtro
            por ``sub_dataset`` devuelve 0 samples (típicamente por typo).
    """
    if split not in ALL_SPLITS:
        raise ValueError(
            f"split desconocido: {split!r}. Válidos: {sorted(ALL_SPLITS)}"
        )

    # load_dataset usa el cache local si está disponible; no re-descarga.
    raw = load_dataset("ai-hyz/MemoryAgentBench", split=split, revision="main")

    filtered = raw.filter(
        lambda row: (row.get("metadata") or {}).get("source", "") == sub_dataset
    )

    if len(filtered) == 0:
        raise ValueError(
            f"No hay samples con metadata.source == {sub_dataset!r} "
            f"en el split {split!r}. ¿Typo en el sub_dataset?"
        )

    if max_samples is not None:
        filtered = filtered.select(range(min(max_samples, len(filtered))))

    return [_row_to_sample(row, idx) for idx, row in enumerate(filtered)]


# ----------------------------------------------------------------------------
# Helpers privados
# ----------------------------------------------------------------------------


def _row_to_sample(row: dict, idx: int) -> MABSample:
    """Convierte una fila cruda de HF al dataclass MABSample."""
    metadata = row.get("metadata") or {}
    source = metadata.get("source", "") or "unknown"
    sample_id = f"{source}__{idx:05d}"

    questions = _as_str_list(row.get("questions"))
    answers_raw = row.get("answers") or []
    answers = [_as_str_list(a) for a in answers_raw]

    # qa_pair_ids puede venir como lista paralela, como string única, o ausente.
    qa_pair_ids = _as_str_list(metadata.get("qa_pair_ids"))
    if len(qa_pair_ids) != len(questions):
        qa_pair_ids = [f"{sample_id}_q{i}" for i in range(len(questions))]

    # Si la cantidad de respuestas no matchea la de preguntas, truncamos / paddeamos
    # con listas vacías para que el sample quede bien formado. Esto no debería
    # pasar en la práctica pero nos cubre de datasets ruidosos.
    if len(answers) < len(questions):
        answers = answers + [[] for _ in range(len(questions) - len(answers))]
    elif len(answers) > len(questions):
        answers = answers[: len(questions)]

    return MABSample(
        sample_id=sample_id,
        source=source,
        context=str(row.get("context") or ""),
        questions=questions,
        answers=answers,
        question_ids=qa_pair_ids,
    )


def _as_str_list(value: object) -> list[str]:
    """Normaliza cualquier valor escalar o lista a lista de strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    # Escalar suelto -> lista de 1
    return [str(value)]
