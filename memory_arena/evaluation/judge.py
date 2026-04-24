"""
memory_arena.evaluation.judge
------------------------------
Jueces LLM-as-judge para evaluar las respuestas generadas en Fase A.

Jueces disponibles:
    - MistralJudge            : LongMemEval (yes/no, 5 question_types + abstention).
                                Prompt copiado textual del repo oficial de LongMemEval.
    - MABAnswerMatchingJudge  : MAB LRU MCQA (detective_qa). Extensión metodológica sobre
                                el paper MAB, que no tiene LLM judge para esta tarea.
    - MABSummarizationJudge   : MAB LRU summarization (infbench_sum_eng_shots2). Replica
                                los 3 prompts (fluency, recall, precision) del repo
                                oficial de MAB, con juez local en lugar de GPT-4o.

Todos los prompts de origen externo se copian textuales y se marca su procedencia
(archivo + licencia) en el docstring de cada función `get_*_prompt`. Mantenerlos
sincronizados si los repos originales se actualizan.
"""

from __future__ import annotations

import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

from memory_arena.llm.ollama_client import OllamaClient


DEFAULT_JUDGE_MODEL = "mistral:7b"


@dataclass
class Judgment:
    """Veredicto de un juez sobre una respuesta.

    Soporta tanto jueces booleanos (LongMemEval, answer-matching) como jueces
    estructurados (summarization, que devuelven fluency/recall/precision/f1).

    Attributes:
        label: True si el juez dictamina que la respuesta es correcta. Para jueces
            estructurados (summarization) queda en None porque no tienen un
            veredicto binario.
        scores: dict de scores numéricos adicionales. Para MABSummarizationJudge
            contiene {"fluency", "recall", "precision", "f1"}. Para booleanos es None.
        judge_model: modelo que emitio el veredicto (ej: "mistral:7b").
        raw_response: texto crudo devuelto por el juez. Para summarization guarda
            una concatenación de los 3 outputs (fluency / recall / precision).
        prompt: util para auditar/debuggear. Para summarization guarda el último
            de los 3 prompts aplicados (el de precision); los otros se pierden para
            mantener el dataclass simple.
    """

    label: bool | None
    judge_model: str
    raw_response: str
    prompt: str | None = None
    scores: dict[str, float] | None = None


class JuezBase(ABC):
    """Contrato que todo juez debe cumplir.

    Mantenemos la firma de `judge()` con los mismos kwargs para que los runners
    de judgment (tanto LongMemEval como MAB) puedan tratar a cualquier juez de
    forma uniforme. Los jueces MAB ignoran `question_type` (reciben el
    sub_dataset via el runner) pero lo aceptan para no romper el contrato.
    """

    @abstractmethod
    def judge(
        self,
        question: str,
        expected_answer: str,
        system_answer: str,
        question_type: str,
        question_id: str,
    ) -> Judgment:
        """Emite un veredicto comparando `system_answer` contra `expected_answer`."""
        ...


# ----------------------------------------------------------------------------
# MistralJudge (LongMemEval) — sin cambios.
# ----------------------------------------------------------------------------


class MistralJudge(JuezBase):
    """Juez con Mistral 7B local via Ollama para LongMemEval.

    Usa el prompt oficial del paper (xiaowu0162/LongMemEval), branchea por
    question_type (single-session-user/-assistant, multi-session, temporal-reasoning,
    knowledge-update, single-session-preference) + branch de abstention.
    """

    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm if llm is not None else OllamaClient(model=DEFAULT_JUDGE_MODEL)

    def judge(
        self,
        question: str,
        expected_answer: str,
        system_answer: str,
        question_type: str,
        question_id: str,
    ) -> Judgment:
        is_abstention = "_abs" in question_id
        prompt = get_anscheck_prompt(
            task=question_type,
            question=question,
            answer=expected_answer,
            response=system_answer,
            abstention=is_abstention,
        )
        raw = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        label = "yes" in raw.lower()
        return Judgment(
            label=label,
            judge_model=self.llm.model,
            raw_response=raw.strip(),
            prompt=prompt,
        )


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    """Prompt del juez para LongMemEval.

    Copiado textualmente de xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py
    (MIT license, Wu et al. 2024). Mantener sincronizado si el repo se actualiza.
    """
    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\nQuestion: {}\n\nExplanation: "
            "{}\n\nModel Response: {}\n\nDoes the model correctly identify the "
            "question as unanswerable? Answer yes or no only."
        )
        return template.format(question, answer, response)

    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, "
            "answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
            "Response: {}\n\nIs the model response correct? Answer yes or no "
            "only."
        )
        return template.format(question, answer, response)

    if task == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, "
            "answer no. In addition, do not penalize off-by-one errors for the "
            "number of days. If the question asks for the number of "
            "days/weeks/months, etc., and the model makes off-by-one errors "
            "(e.g., predicting 19 days when the answer is 18), the model's "
            "response is still correct. \n\nQuestion: {}\n\nCorrect Answer: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer "
            "yes or no only."
        )
        return template.format(question, answer, response)

    if task == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response "
            "should be considered as correct as long as the updated answer is "
            "the required answer.\n\nQuestion: {}\n\nCorrect Answer: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer "
            "yes or no only."
        )
        return template.format(question, answer, response)

    if task == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. The "
            "model does not need to reflect all the points in the rubric. The "
            "response is correct as long as it recalls and utilizes the user's "
            "personal information correctly.\n\nQuestion: {}\n\nRubric: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer "
            "yes or no only."
        )
        return template.format(question, answer, response)

    raise NotImplementedError(f"question_type no soportado: {task!r}")


# ----------------------------------------------------------------------------
# MABAnswerMatchingJudge (detective_qa y futuras tareas MCQA/short-answer de LRU)
# ----------------------------------------------------------------------------


class MABAnswerMatchingJudge(JuezBase):
    """Juez answer-matching para MAB LRU (tipicamente MCQA).

    Extensión metodológica sobre el paper MemoryAgentBench (Hu et al., ICLR 2026),
    que para `detective_qa` no tiene LLM judge oficial y se queda con el scorer
    default (EM/substring/ROUGE) — insuficiente para MCQA 4-opciones donde el
    EM estricto queda por debajo del piso de chance (0.25).

    El prompt le pide al juez que ignore diferencias de formato y decida si el
    modelo eligió la misma opción/respuesta que el gold. Devuelve yes/no igual
    que el de LongMemEval, para mantener compatibilidad con el pipeline
    booleano.
    """

    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm if llm is not None else OllamaClient(model=DEFAULT_JUDGE_MODEL)

    def judge(
        self,
        question: str,
        expected_answer: str,
        system_answer: str,
        question_type: str,  # se ignora; se deja por compatibilidad
        question_id: str,    # idem
    ) -> Judgment:
        prompt = get_mab_answer_matching_prompt(
            question=question,
            answer=expected_answer,
            response=system_answer,
        )
        raw = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        label = "yes" in raw.lower()
        return Judgment(
            label=label,
            judge_model=self.llm.model,
            raw_response=raw.strip(),
            prompt=prompt,
        )


def get_mab_answer_matching_prompt(
    question: str,
    answer: str,
    response: str,
) -> str:
    """Prompt answer-matching para tareas MCQA/short-answer de MAB LRU.

    Diseñado a mano para `detective_qa`, inspirado en el prompt de LongMemEval
    (same yes/no contract) pero explícito sobre MCQA: el modelo gana si eligió
    la misma opción, independientemente del formato de salida.
    """
    template = (
        "I will give you a multiple-choice or short-answer question, the correct "
        "answer, and a response from a model. Please answer yes if the response "
        "selects the same answer option as the correct answer. Otherwise, answer "
        "no.\n\n"
        "Rules:\n"
        "- Ignore formatting differences. If the correct answer is 'A. Blue Hat "
        "Stranger' and the model responds 'A', 'Answer: A', 'The correct answer "
        "is A. Blue Hat Stranger', or 'Blue Hat Stranger', these all count as yes.\n"
        "- The model may include extra reasoning or explanation; only the selected "
        "option matters.\n"
        "- If the response is ambiguous (picks multiple options, refuses, or does "
        "not commit), answer no.\n"
        "- If the response picks a different option than the correct answer, "
        "answer no.\n\n"
        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
        "Did the model select the same answer as the correct answer? "
        "Answer yes or no only."
    )
    return template.format(question, answer, response)


# ----------------------------------------------------------------------------
# MABSummarizationJudge (infbench_sum_eng_shots2)
# ----------------------------------------------------------------------------


class MABSummarizationJudge(JuezBase):
    """Juez de summarization para MAB LRU (`infbench_sum_eng_shots2`).

    Replica el pipeline de 3 prompts (fluency + recall + precision) del repo
    oficial de MAB (`llm_based_eval/summarization_evaluate.py`) con Mistral 7B
    local en lugar de GPT-4o. Los prompts se copian textuales.

    Devuelve Judgment con:
        - label=None (no es juez binario)
        - scores={"fluency", "recall", "precision", "f1"} en [0, 1]
          donde f1 = fluency * 2 * rec * prec / (rec + prec) (igual al paper).

    Robustez empírica (observada en la primera corrida 2026-04-23):
        - Mistral 7B parseó JSON correctamente en 100/100 samples (0 parse fails).
        - En algunos casos devolvió valores inconsistentes (ej. `precision: X,
          sentence_count: Y` con X > Y, imposible semánticamente). Cap a [0, 1]
          antes de computar F1 y contamos los out-of-range en
          `self.out_of_range_count` para trazabilidad (log a stderr cuando > 0).

    IMPORTANTE sobre schema MAB:
        El dataclass MABSample del loader no carga keypoints; los tiene el
        metadata de HF bajo metadata.keypoints. Este juez espera que el campo
        `expected_answer` ya contenga los keypoints (como JSON con
        "keypoints" y "expert_summary"). El `mab_judgment_runner` se encarga
        de enriquecer el record desde HF antes de invocar este juez.
    """

    def __init__(self, llm: OllamaClient | None = None, max_tokens: int = 512):
        self.llm = llm if llm is not None else OllamaClient(model=DEFAULT_JUDGE_MODEL)
        # Los prompts de summarization piden JSON con scores + reasoning;
        # necesitamos más tokens que los 10 del yes/no.
        self.max_tokens = max_tokens
        # Contadores de sanity check para reportar al final del run.
        self.out_of_range_count = 0

    def judge(
        self,
        question: str,
        expected_answer: str,  # expected_answer debe contener keypoints y expert summary
        system_answer: str,
        question_type: str,    # se ignora
        question_id: str,      # se ignora
    ) -> Judgment:
        keypoints, expert_summary = _parse_summarization_reference(expected_answer)

        fluency_prompt = FLUENCY_PROMPT_BOOK.format(text=system_answer.strip())
        recall_prompt = RECALL_PROMPT_BOOK.format(
            keypoints="\n".join(f"{i+1}. {kp}" for i, kp in enumerate(keypoints)),
            summary=system_answer.strip(),
        )
        precision_prompt = PRECISION_PROMPT_BOOK.format(
            expert_summary=expert_summary,
            summary=system_answer.strip(),
        )

        fluency_raw = self.llm.chat(
            [{"role": "user", "content": fluency_prompt}], max_tokens=self.max_tokens
        )
        recall_raw = self.llm.chat(
            [{"role": "user", "content": recall_prompt}], max_tokens=self.max_tokens
        )
        precision_raw = self.llm.chat(
            [{"role": "user", "content": precision_prompt}], max_tokens=self.max_tokens
        )

        fluency_score = _parse_json_score(fluency_raw, key="fluency")
        recall_obj = _parse_json_object(recall_raw)
        precision_obj = _parse_json_object(precision_raw)

        recall_raw_n = _safe_int(recall_obj, "recall")
        n_keypoints = len(keypoints) if keypoints else 0
        rec = (recall_raw_n / n_keypoints) if n_keypoints > 0 else 0.0

        prec_raw_n = _safe_int(precision_obj, "precision")
        sent_count = _safe_int(precision_obj, "sentence_count")
        prec = (prec_raw_n / sent_count) if sent_count > 0 else 0.0

        # Cap a [0, 1] — el juez local a veces devuelve valores imposibles
        # (ej. precision > sentence_count). Loguear el caso para trazabilidad.
        rec_out_of_range = not (0.0 <= rec <= 1.0)
        prec_out_of_range = not (0.0 <= prec <= 1.0)
        if rec_out_of_range or prec_out_of_range:
            self.out_of_range_count += 1
            print(
                f"[MABSummarizationJudge] out-of-range score (qid={question_id}): "
                f"recall={rec:.3f} (raw={recall_raw_n}/{n_keypoints}) "
                f"precision={prec:.3f} (raw={prec_raw_n}/{sent_count}). "
                f"Capped to [0,1].",
                file=sys.stderr,
            )
        rec = max(0.0, min(1.0, rec))
        prec = max(0.0, min(1.0, prec))

        flu = float(fluency_score) if fluency_score is not None else 0.0
        # Fluency también: nominalmente 0 o 1, pero por las dudas lo capeamos.
        flu = max(0.0, min(1.0, flu))

        if rec + prec > 0:
            f1 = flu * 2 * (rec * prec) / (rec + prec)
        else:
            f1 = 0.0

        scores = {
            "fluency": round(flu, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
        }

        return Judgment(
            label=None,
            judge_model=self.llm.model,
            raw_response=(
                f"---fluency---\n{fluency_raw.strip()}\n"
                f"---recall---\n{recall_raw.strip()}\n"
                f"---precision---\n{precision_raw.strip()}"
            ),
            prompt=precision_prompt,  # guardamos solo el último; reasonable
            scores=scores,
        )


# ----------------------------------------------------------------------------
# Helpers: parse JSON con fallback, parse del reference de summarization
# ----------------------------------------------------------------------------


def _parse_summarization_reference(expected_answer: str) -> tuple[list[str], str]:
    """Parsea el `expected_answer` al formato (keypoints, expert_summary).

    El formato esperado (producido por el preprocesamiento que hacemos en el
    runner o por un helper de reference-loading) es JSON:
        {"keypoints": ["...", "..."], "expert_summary": "..."}

    Si no viene en ese formato, interpretamos el string completo como
    expert_summary y dejamos keypoints vacío. Esto nos deja fallback robusto
    para JSONLs históricos donde solo teníamos `gold_answers[0]`.
    """
    if not expected_answer:
        return [], ""
    try:
        obj = json.loads(expected_answer)
        if isinstance(obj, dict):
            kps = obj.get("keypoints") or []
            es = obj.get("expert_summary") or ""
            if isinstance(kps, list) and isinstance(es, str):
                return [str(k) for k in kps], es
    except (json.JSONDecodeError, TypeError):
        pass
    return [], str(expected_answer)


def _parse_json_object(text: str) -> dict | None:
    """Extrae el último objeto JSON válido del texto. None si no encuentra.

    Espejo defensivo del `parse_json` del repo oficial de MAB: busca `{...}`
    y ```json ... ``` como fallbacks.
    """
    if not text:
        return None
    # Buscar objetos JSON {...} (non-greedy pero que capturen todo)
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass
    # Fallback: ```json ... ```
    fenced = re.findall(r"```json\s*(.+?)\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced[-1])
        except json.JSONDecodeError:
            pass
    return None


def _parse_json_score(text: str, key: str) -> float | None:
    """Extrae `{key: value}` del texto. Usado para fluency (score único)."""
    obj = _parse_json_object(text)
    if obj is None:
        return None
    val = obj.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _safe_int(obj: dict | None, key: str) -> int:
    """Levanta un entero del dict; 0 si falla."""
    if obj is None:
        return 0
    val = obj.get(key)
    if isinstance(val, (int, float)):
        return int(val)
    return 0


# ----------------------------------------------------------------------------
# Prompts de summarization (copiados textuales del repo oficial de MAB)
# ----------------------------------------------------------------------------
# Fuente: HUST-AI-HYZ/MemoryAgentBench/llm_based_eval/summarization_evaluate.py
# Licencia: misma que el repo (Apache 2.0 salvo que se actualice).
# Usamos las variantes *_book porque infbench_sum_eng_shots2 es novelas.
# ----------------------------------------------------------------------------

FLUENCY_PROMPT_BOOK = """Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers. If the text is coherent, non-repetitive, and fluent, but the last sentence is truncated, it should still be given a score of 1.

Now, read the provided text, and evaluate the fluency using the rubric. Then output your score in the following json format: {{"fluency": 1}}.

Text: "{text}"
"""

RECALL_PROMPT_BOOK = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel. It should discuss the plots and characters of the story. The text should contain all the given key points.

Below is your grading rubric:
Recall:
- Evaluate the provided summary by deciding if each of the key points is present in the provided summary. A key point is considered present if its factual information is mostly-supported by the provided summary. If a key point contains multiple facts, it's still considered supported if most of the facts are present.
- Score: the number of key points mostly-supported by the provided summary.

Now, read the provided summary and key points, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"supported_key_points": [2, 4], "recall": 2}}, where "supported_key_points" contains the key points that are present in the summary and "recall" is the total number of key points present in the summary.

Key points:
{keypoints}

Summary: <start of summary>{summary}<end of summary>
"""

PRECISION_PROMPT_BOOK = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel.

Below is your grading rubric:
Precision:
- Evaluate the provided summary by deciding if each sentence in the provided summary is supported by the information provided in the expert summary. A sentence is considered supported if its major facts align with the information in the expert summary. A sentence is still considered supported even if some of its minor details, such as dates, entity names, or the location, are not explicitly mentioned in the expert summary. A sentence is not supported if its major facts are not mentioned or contradicted in the expert summary. It is also not supported if it introduces new information not present in the expert summary, such as additional analysis or commentary on the story.
- Score: the number of sentences in the provided summary that are supported by the expert summary.

Now, read the provided summary and expert summary, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"precision": 7, "sentence_count": 20}}.

Expert summary: <start of summary>{expert_summary}<end of summary>

Provided summary: <start of summary>{summary}<end of summary>
"""
