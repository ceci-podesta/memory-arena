"""
memory_arena.evaluation.judge
------------------------------
Juez LLM-as-judge para evaluar las respuestas generadas en Fase A.

El prompt del juez se copia textual del repo oficial de LongMemEval
(src/evaluation/evaluate_qa.py), licencia MIT, autores Wu et al. 2024.
Usamos los mismos 5 prompts por question_type + el prompt de abstention
para garantizar compatibilidad metodologica con el paper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from memory_arena.llm.ollama_client import OllamaClient


DEFAULT_JUDGE_MODEL = "mistral:7b"


@dataclass
class Judgment:
    """Veredicto de un juez sobre una respuesta."""

    label: bool  # True = respuesta correcta
    judge_model: str  # modelo que emitio el veredicto
    raw_response: str  # texto crudo devuelto por el juez
    prompt: str | None = None  # util para auditar/debuggear


class JuezBase(ABC):
    """Contrato que todo juez debe cumplir."""

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


class MistralJudge(JuezBase):
    """Juez con Mistral 7B local via Ollama."""

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
