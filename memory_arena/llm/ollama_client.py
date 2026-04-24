"""
memory_arena.llm.ollama_client
-------------------------------
Wrapper sobre la API de Ollama con parámetros de sampling configurables.

Los defaults experimentales (num_ctx, temperature, top_p, seed, max_new_tokens)
viven en memory_arena.experimental_config — cambiar ahí impacta uniformemente a
todos los benchmarks.
"""

from dataclasses import dataclass, field

import ollama

from memory_arena.experimental_config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_CTX,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)

# Constantes de infra (no experimentales).
DEFAULT_CHAT_MODEL = "llama3.2:3b"
DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass
class OllamaClient:
    model: str = DEFAULT_CHAT_MODEL
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    seed: int = DEFAULT_SEED
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    host: str = DEFAULT_HOST
    _client: ollama.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = ollama.Client(host=self.host)

    def chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> str:
        """Llama al modelo y devuelve el texto de la respuesta.

        Args:
            messages: lista de mensajes en formato ChatML.
            max_tokens: override opcional del máximo de tokens de salida.
                Si es None, usa self.max_new_tokens (default experimental).
                Se mapea a ``num_predict`` en Ollama.
        """
        effective_max = max_tokens if max_tokens is not None else self.max_new_tokens

        options: dict = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "num_predict": effective_max,
        }

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
        )
        return response["message"]["content"]
