"""
memory_arena.memories.no_memory
--------------------------------
Estrategia 1: sin memoria (baseline).

El agente no recibe ningún contexto de turnos anteriores. Responde a cada
pregunta en blanco, como si fuera la primera interacción. Sirve como
punto de referencia — cualquier estrategia que NO supere a este baseline
no está agregando valor, solo complejidad.
"""

from memory_arena.memories.base import MemoriaBase, Turn


class NoMemoria(MemoriaBase):
    """Baseline sin memoria: almacena nada, recupera nada.

    Útil como punto de comparación contra el cual medir cuánto aporta
    cada estrategia más sofisticada. Si otra estrategia no supera a
    NoMemoria en un tipo de tarea, sabemos que no está ayudando ahí
    (o peor: está metiendo ruido al LLM).
    """

    def store(self, turn: Turn) -> None:
        # Intencionalmente no hace nada. "Sin memoria" = sin almacenamiento.
        pass

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        # Intencionalmente devuelve lista vacía. El agente va a responder
        # sin contexto histórico, solo con lo que "sabe" el modelo
        # (o sea, lo que fue entrenado a saber, nada más).
        return []
