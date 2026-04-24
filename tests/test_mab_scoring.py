"""Tests para memory_arena.evaluation.mab_scoring.

Todos los tests son unit, sin red, sin Ollama. Deberían correr en <1s.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arena.evaluation.mab_scoring import (
    calculate_default_metrics,
    exact_match_score,
    f1_token_score,
    normalize_answer,
    parse_output,
    score_jsonl,
    score_response,
    substring_exact_match_score,
)


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("HELLO World") == "hello world"

    def test_strip_punctuation(self):
        assert normalize_answer("Hello, World!") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("The quick brown fox") == "quick brown fox"
        assert normalize_answer("a dog and an owl") == "dog and owl"

    def test_collapse_whitespace(self):
        assert normalize_answer("  too   much    space  ") == "too much space"

    def test_empty_string(self):
        assert normalize_answer("") == ""


# ---------------------------------------------------------------------------
# Métricas atómicas
# ---------------------------------------------------------------------------


class TestExactMatchScore:
    def test_exact(self):
        assert exact_match_score("Paris", "Paris") == 1.0

    def test_case_and_punctuation(self):
        assert exact_match_score("Paris!", "paris") == 1.0

    def test_articles_ignored(self):
        assert exact_match_score("The Eiffel Tower", "Eiffel Tower") == 1.0

    def test_no_match(self):
        assert exact_match_score("Paris", "London") == 0.0


class TestSubstringExactMatchScore:
    def test_gold_inside_prediction(self):
        pred = "The answer is definitely Paris, no doubt."
        assert substring_exact_match_score(pred, "Paris") == 1.0

    def test_gold_not_in_prediction(self):
        assert substring_exact_match_score("I don't know", "Paris") == 0.0

    def test_articles_normalized(self):
        # "the Eiffel Tower" normaliza a "eiffel tower", que debería estar en la pred.
        assert (
            substring_exact_match_score(
                "I think it is the Eiffel Tower in Paris.", "Eiffel Tower"
            )
            == 1.0
        )


class TestF1TokenScore:
    def test_identical(self):
        assert f1_token_score("Paris is nice", "Paris is nice") == pytest.approx(1.0)

    def test_partial_overlap(self):
        # pred tokens: {paris, nice} — gold tokens: {london, nice}
        # common = 1 ({nice}), precision = 1/2, recall = 1/2, f1 = 0.5
        assert f1_token_score("Paris nice", "London nice") == pytest.approx(0.5)

    def test_zero_overlap(self):
        assert f1_token_score("Paris", "London") == 0.0

    def test_yes_no_mismatch(self):
        # Caso especial: yes/no no se cruzan aunque compartan tokens genéricos.
        assert f1_token_score("yes", "no") == 0.0
        assert f1_token_score("yes indeed", "no") == 0.0


# ---------------------------------------------------------------------------
# Multi-gold (max over ground truths)
# ---------------------------------------------------------------------------


class TestMultiGold:
    def test_takes_max(self):
        # Prediction matches only the second gold exactly.
        metrics = calculate_default_metrics(
            "London", ["Paris", "London", "Madrid"]
        )
        assert metrics["exact_match"] == 1.0

    def test_nested_list_flattened(self):
        # Lista anidada (como puede venir de HF).
        metrics = calculate_default_metrics("Paris", [["Paris"], ["London"]])
        assert metrics["exact_match"] == 1.0

    def test_string_gold(self):
        # String suelto también debe funcionar.
        metrics = calculate_default_metrics("Paris", "Paris")
        assert metrics["exact_match"] == 1.0

    def test_empty_golds(self):
        metrics = calculate_default_metrics("Paris", [])
        assert metrics["exact_match"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["rougeL_f1"] == 0.0


# ---------------------------------------------------------------------------
# calculate_default_metrics devuelve las 7 claves
# ---------------------------------------------------------------------------


class TestDefaultMetricsKeys:
    def test_all_seven_keys_present(self):
        metrics = calculate_default_metrics("Paris", ["Paris"])
        expected = {
            "exact_match",
            "substring_exact_match",
            "f1",
            "rougeL_f1",
            "rougeL_recall",
            "rougeLsum_f1",
            "rougeLsum_recall",
        }
        assert set(metrics.keys()) == expected

    def test_all_values_in_unit_interval(self):
        metrics = calculate_default_metrics(
            "The answer is probably Paris, actually.", ["Paris"]
        )
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} fuera de [0,1]"


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------


class TestParseOutput:
    def test_with_answer_prefix(self):
        assert parse_output("Answer: Paris") == "Paris"

    def test_with_answer_prefix_and_trailing(self):
        assert parse_output("Answer: Paris\nExplanation: ...") == "Paris"

    def test_without_prefix_first_line(self):
        assert parse_output("Paris is the capital.") == "Paris is the capital."

    def test_empty(self):
        assert parse_output("") is None

    def test_repeated_prefix(self):
        # El oficial limpia prefijos repetidos.
        assert parse_output("Answer: Answer: Paris") == "Paris"


# ---------------------------------------------------------------------------
# Dispatcher por sub_dataset
# ---------------------------------------------------------------------------


class TestScoreResponseDispatcher:
    def test_default_longmemeval(self):
        metrics = score_response(
            "The answer is Paris.", ["Paris"], "longmemeval_s*"
        )
        assert metrics["substring_exact_match"] == 1.0
        assert "eventqa_recall" not in metrics

    def test_eventqa_adds_recall(self):
        metrics = score_response(
            "The meeting was about budget and quarterly reports.",
            ["budget", "quarterly reports", "launch"],
            "eventqa_full",
        )
        # 2 de 3 golds presentes.
        assert metrics["eventqa_recall"] == pytest.approx(2 / 3)

    def test_icl_uses_parsed(self):
        # Con rama icl, parse_output extrae después de 'Answer:'
        metrics = score_response(
            "Thinking... Answer: card_arrival\n",
            ["card_arrival"],
            "icl_banking77_5900shot_balance",
        )
        assert metrics["exact_match"] == 1.0

    def test_factconsolidation_uses_default(self):
        metrics = score_response(
            "India", ["India"], "factconsolidation_sh_6k"
        )
        assert metrics["exact_match"] == 1.0
        assert "eventqa_recall" not in metrics

    def test_default_takes_max_of_direct_and_parsed(self):
        # Con 'Answer:' explícito, el parsed debería dar más score que el raw
        # en ciertos casos; el default toma el max.
        metrics = score_response(
            "I think about this...\nAnswer: Paris",
            ["Paris"],
            "longmemeval_s*",
        )
        assert metrics["exact_match"] == 1.0


# ---------------------------------------------------------------------------
# score_jsonl (smoke)
# ---------------------------------------------------------------------------


class TestScoreJsonl:
    def test_smoke_three_records(self, tmp_path: Path):
        jsonl_path = tmp_path / "fake.jsonl"
        records = [
            {
                "split": "Conflict_Resolution",
                "sub_dataset": "factconsolidation_sh_6k",
                "source": "factconsolidation_sh_6k",
                "sample_id": "factconsolidation_sh_6k__00000",
                "question_id": "fc_q0",
                "question_idx": 0,
                "question": "What country does X belong to?",
                "gold_answers": ["India"],
                "system_answer": "The answer is India.",
                "retrieved_context": [],
                "retrieved_count": 0,
                "latency_s": 1.2,
            },
            {
                "split": "Conflict_Resolution",
                "sub_dataset": "factconsolidation_sh_6k",
                "source": "factconsolidation_sh_6k",
                "sample_id": "factconsolidation_sh_6k__00000",
                "question_id": "fc_q1",
                "question_idx": 1,
                "question": "What country does Y belong to?",
                "gold_answers": ["China"],
                "system_answer": "I'm not sure.",
                "retrieved_context": [],
                "retrieved_count": 0,
                "latency_s": 0.8,
            },
            {
                "split": "Conflict_Resolution",
                "sub_dataset": "factconsolidation_sh_6k",
                "source": "factconsolidation_sh_6k",
                "sample_id": "factconsolidation_sh_6k__00000",
                "question_id": "fc_q2",
                "question_idx": 2,
                "question": "What country does Z belong to?",
                "gold_answers": ["Brazil"],
                "system_answer": "Brazil.",
                "retrieved_context": [],
                "retrieved_count": 0,
                "latency_s": 0.5,
            },
        ]
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        result = score_jsonl(jsonl_path)

        assert result["aggregates"]["n_total"] == 3
        assert "per_question" in result
        assert len(result["per_question"]) == 3

        # record 0: substring EM deberíamos matchear ("The answer is India." contiene "India")
        assert result["per_question"][0]["metrics"]["substring_exact_match"] == 1.0
        # record 1: no match
        assert result["per_question"][1]["metrics"]["exact_match"] == 0.0
        # record 2: exact match (después de normalizar punctuation)
        assert result["per_question"][2]["metrics"]["exact_match"] == 1.0

        # Aggregate global: substring_em promedio = (1 + 0 + 1) / 3
        assert result["aggregates"]["by_metric"]["substring_exact_match"]["mean"] == pytest.approx(
            2 / 3
        )
        assert result["aggregates"]["by_metric"]["substring_exact_match"]["n"] == 3

        # Aggregate por sub_dataset
        assert "factconsolidation_sh_6k" in result["aggregates"]["by_sub_dataset"]
        assert result["aggregates"]["by_sub_dataset"]["factconsolidation_sh_6k"]["n"] == 3
