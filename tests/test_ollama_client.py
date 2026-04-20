"""
Tests para OllamaClient.

Los tests que requieren un servidor Ollama corriendo se skipean automáticamente
si no está disponible. 
"""

import pytest

from memory_arena.llm.ollama_client import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    OllamaClient,
)


def _ollama_disponible() -> bool:
    """Devuelve True si hay un servidor Ollama respondiendo en localhost."""
    try:
        import ollama
        ollama.Client(host="http://127.0.0.1:11434").list()
        return True
    except Exception:
        return False


# -------------------------------------------------------------------
# Tests unitarios (no requieren Ollama)
# -------------------------------------------------------------------

def test_defaults_son_deterministicos():
    """Verificar que los defaults elegidos coincidan con lo que documentamos."""
    assert DEFAULT_TEMPERATURE == 0.0
    assert DEFAULT_TOP_P == 1.0
    assert DEFAULT_SEED == 42
    assert DEFAULT_NUM_CTX == 16384
    assert DEFAULT_CHAT_MODEL == "llama3.2:3b"


def test_cliente_se_instancia_con_defaults():
    """OllamaClient debe instanciarse sin argumentos."""
    client = OllamaClient()
    assert client.model == DEFAULT_CHAT_MODEL
    assert client.num_ctx == DEFAULT_NUM_CTX
    assert client.temperature == DEFAULT_TEMPERATURE
    assert client.top_p == DEFAULT_TOP_P
    assert client.seed == DEFAULT_SEED


def test_cliente_acepta_overrides():
    """Se pueden pasar parametros custom al constructor."""
    client = OllamaClient(temperature=0.7, seed=99)
    assert client.temperature == 0.7
    assert client.seed == 99
    # Los que no pasamos deben seguir con el default
    assert client.num_ctx == DEFAULT_NUM_CTX


# -------------------------------------------------------------------
# Tests de integracion (requieren Ollama + llama3.2:3b descargado)
# -------------------------------------------------------------------

@pytest.mark.skipif(
    not _ollama_disponible(),
    reason="Ollama no esta corriendo en localhost",
)
def test_chat_devuelve_string_no_vacio():
    """Smoke test: llamar al modelo debe devolver texto no vacio."""
    client = OllamaClient()
    respuesta = client.chat([
        {"role": "user", "content": "Responde con una sola palabra: hola"}
    ])
    assert isinstance(respuesta, str)
    assert len(respuesta) > 0


@pytest.mark.skipif(
    not _ollama_disponible(),
    reason="Ollama no esta corriendo en localhost",
)
def test_determinismo_dos_llamadas_iguales():
    """Con temperature=0 y seed fija, dos llamadas al mismo prompt deben dar el mismo output."""
    client = OllamaClient()
    mensaje = [{"role": "user", "content": "Nombra tres frutas separadas por coma."}]
    respuesta_a = client.chat(mensaje)
    respuesta_b = client.chat(mensaje)
    assert respuesta_a == respuesta_b, (
        f"Determinismo roto: '{respuesta_a}' != '{respuesta_b}'"
    )
