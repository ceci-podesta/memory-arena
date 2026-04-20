"""
memory_arena.llm.ollama_client
-------------------------------
Wrapper sobre la API de Ollama con parametros de sampling configurables.
"""

from dataclasses import dataclass, field

import ollama

DEFAULT_NUM_CTX = 16384
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 42
DEFAULT_CHAT_MODEL = "llama3.2:3b"
DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass
class OllamaClient:
    model: str = DEFAULT_CHAT_MODEL
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    seed: int = DEFAULT_SEED
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
            max_tokens: limite duro de tokens de salida. Se mapea a
                `num_predict` en Ollama. None deja el default del modelo.
        """
        options: dict = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
        )
        return response["message"]["content"]
