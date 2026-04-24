"""
Consistency tests: todos los runners y el cliente LLM deben usar los defaults
de memory_arena.experimental_config.

Si un test acá falla, es que alguien metió un default local (en un runner nuevo
o ajustando uno existente) que rompería la comparabilidad entre corridas.
"""

import inspect

from memory_arena import experimental_config as cfg
from memory_arena.evaluation.mab_runner import run_strategy_mab
from memory_arena.evaluation.runner import run_strategy
from memory_arena.llm.ollama_client import OllamaClient


def _default_of(func_or_class, param_name):
    """Devuelve el valor default de un parámetro dado, con mensaje útil si falla."""
    sig = inspect.signature(func_or_class)
    if param_name not in sig.parameters:
        raise AssertionError(
            f"{func_or_class.__name__} no expone el parámetro '{param_name}'. "
            "Si lo quitaste a propósito, actualizá también el test."
        )
    param = sig.parameters[param_name]
    if param.default is inspect.Parameter.empty:
        raise AssertionError(
            f"{func_or_class.__name__} expone '{param_name}' sin default. "
            "Poné un default importado de experimental_config."
        )
    return param.default


# --- OllamaClient -----------------------------------------------------------

def test_ollama_num_ctx_matches_config():
    assert _default_of(OllamaClient, "num_ctx") == cfg.DEFAULT_NUM_CTX


def test_ollama_temperature_matches_config():
    assert _default_of(OllamaClient, "temperature") == cfg.DEFAULT_TEMPERATURE


def test_ollama_top_p_matches_config():
    assert _default_of(OllamaClient, "top_p") == cfg.DEFAULT_TOP_P


def test_ollama_seed_matches_config():
    assert _default_of(OllamaClient, "seed") == cfg.DEFAULT_SEED


def test_ollama_max_new_tokens_matches_config():
    assert _default_of(OllamaClient, "max_new_tokens") == cfg.DEFAULT_MAX_NEW_TOKENS


# --- MAB runner -------------------------------------------------------------

def test_mab_runner_max_new_tokens_matches_config():
    assert _default_of(run_strategy_mab, "max_new_tokens") == cfg.DEFAULT_MAX_NEW_TOKENS


def test_mab_runner_top_k_matches_config():
    assert _default_of(run_strategy_mab, "top_k") == cfg.DEFAULT_RETRIEVAL_TOP_K


# --- LongMemEval runner -----------------------------------------------------

def test_longmemeval_runner_max_new_tokens_matches_config():
    assert _default_of(run_strategy, "max_new_tokens") == cfg.DEFAULT_MAX_NEW_TOKENS


def test_longmemeval_runner_top_k_matches_config():
    assert _default_of(run_strategy, "top_k") == cfg.DEFAULT_RETRIEVAL_TOP_K
