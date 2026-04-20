"""
memory_arena.memories.no_memory
--------------------------------
Estrategia 1: sin memoria (baseline).
...
"""

from memory_arena.memories.base import MemoriaBase, Turn


class NoMemoria(MemoriaBase):
    """Baseline sin memoria: almacena nada, recupera nada."""

    def store(self, turn: Turn) -> None:
        pass

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        return []

    def reset(self) -> None:
        # No hay estado que limpiar.
        pass
