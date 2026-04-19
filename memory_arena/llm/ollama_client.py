
"""
memory_arena.llm.ollama_client
-------------------------------
Wrapper sobre la API de Ollama con parametros de sampling configurables.

Todas las estrategias llaman al modelo a traves de este cliente para que la
comparacion entre ellas sea justa (misma ventana de contexto, misma seed,
misma temperatura, etc.).
"""

from dataclasses import dataclass, field

import ollama


# Ventana de contexto por default.
# Llama 3.2 3B soporta hasta 131072 tokens pero Ollama por default limita a
# 4096. Subimos a 16384 para no castrar las estrategias context-heavy.
# Ver project_context_window.md.
DEFAULT_NUM_CTX = 16384

# Parametros de sampling por default.
# Para benchmarks queremos reproducibilidad → determinismo.
# - temperature=0.0 → siempre se toma el token mas probable (argmax).
# - seed=42         → fija el RNG en caso de desempates o variantes del runtime.
# - top_p=1.0       → sin truncado por nucleus (con temp=0 no importa,
#                     pero lo dejamos explicito para no heredar el 0.9 de Ollama).
#
# Referencia defaults de Ollama (si no pasaramos estos valores):
#   temperature=0.8, top_p=0.9, top_k=40, seed=0 (random), repeat_penalty=1.1
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 42

# Modelo por default.
DEFAULT_CHAT_MODEL = "llama3.2:3b"

# Host por default: localhost. Ollama corre local, no queremos exponerlo.
DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass
class OllamaClient:
    """Cliente fino sobre Ollama con parametros de sampling fijos por instancia."""

    model: str = DEFAULT_CHAT_MODEL
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    seed: int = DEFAULT_SEED
    host: str = DEFAULT_HOST
    _client: ollama.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = ollama.Client(host=self.host)

    def chat(self, messages: list[dict]) -> str:
        """Llamar al modelo con un historial de mensajes y devolver el texto.

        `messages` sigue el formato estandar:
            [{"role": "user" | "assistant" | "system", "content": "..."}]
        """
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
            },
        )
        return response["message"]["content"]

