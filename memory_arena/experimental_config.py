"""
Configuración experimental compartida entre estrategias y benchmarks.

Todos los runners, clientes LLM y estrategias deben importar sus hiperparámetros
de acá. Cambiar un valor en este archivo afecta a TODO el sistema — documentar
el cambio en el informe como parte del setup experimental.

El test tests/test_config_consistency.py verifica que los runners y el cliente
LLM usen estos defaults. Si falla, es que alguien metió un default local que
rompe la comparabilidad entre corridas.
"""

# --- Generación del LLM ---
DEFAULT_TEMPERATURE: float = 0.0
"""Temperatura de sampling. 0.0 = determinismo intra-máquina."""

DEFAULT_TOP_P: float = 1.0
"""Nucleus sampling. 1.0 = sin filtrado."""

DEFAULT_SEED: int = 42
"""Semilla para reproducibilidad intra-máquina."""

DEFAULT_NUM_CTX: int = 16384
"""Ventana de contexto total del modelo (input + output), en tokens."""

DEFAULT_MAX_NEW_TOKENS: int = 512
"""Tokens máximos a generar en la respuesta. Suficiente para summaries (LRU)
y no limitante para respuestas cortas (CR, AR)."""

# --- Retrieval ---
DEFAULT_RETRIEVAL_TOP_K: int = 5
"""Cantidad default de chunks a devolver en retrieve()."""
