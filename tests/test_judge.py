"""Tests de la logica de construccion de prompts del juez."""

import pytest

from memory_arena.evaluation.judge import get_anscheck_prompt


def test_single_session_user_contiene_pregunta_y_respuesta():
    prompt = get_anscheck_prompt(
        task="single-session-user",
        question="What car did I buy?",
        answer="A Toyota Corolla",
        response="You bought a Toyota.",
    )
    assert "What car did I buy?" in prompt
    assert "A Toyota Corolla" in prompt
    assert "You bought a Toyota." in prompt
    assert "Answer yes or no only" in prompt


def test_temporal_reasoning_menciona_off_by_one():
    prompt = get_anscheck_prompt(
        task="temporal-reasoning",
        question="how many days?",
        answer="18",
        response="19",
    )
    assert "off-by-one" in prompt


def test_knowledge_update_menciona_updated_answer():
    prompt = get_anscheck_prompt(
        task="knowledge-update",
        question="what is my phone number?",
        answer="555-1234",
        response="Previously 555-0000, now 555-1234.",
    )
    assert "updated answer" in prompt


def test_preference_menciona_rubric():
    prompt = get_anscheck_prompt(
        task="single-session-preference",
        question="recomendame una peli",
        answer="rubric: considerar que le gustan dramas",
        response="te recomiendo Oppenheimer",
    )
    assert "Rubric" in prompt
    assert "rubric: considerar que le gustan dramas" in prompt


def test_abstention_menciona_unanswerable():
    prompt = get_anscheck_prompt(
        task="single-session-user",  # el task no importa en abstention
        question="que dije la semana pasada?",
        answer="no se menciono",
        response="no tengo esa informacion",
        abstention=True,
    )
    assert "unanswerable" in prompt


def test_task_invalido_raise():
    with pytest.raises(NotImplementedError):
        get_anscheck_prompt(
            task="nope-invalid-type",
            question="q",
            answer="a",
            response="r",
        )


def test_multi_session_y_single_user_comparten_mismo_template():
    prompt_a = get_anscheck_prompt(
        "single-session-user", "q", "a", "r",
    )
    prompt_b = get_anscheck_prompt(
        "multi-session", "q", "a", "r",
    )
    assert prompt_a == prompt_b
