"""
memory_arena.memories.base
---------------------------
Interfaz base para todas las estrategias de memoria.

Todas las estrategias implementan el mismo contrato — un método `store`
que recibe un turno de conversación, y un método `retrieve` que devuelve
los fragmentos más relevantes para una pregunta dada. Gracias a esto,
el loop de evaluación puede intercambiar estrategias sin tocar nada más.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Turn:
    """Un turno de conversación: quién habló y qué dijo."""

    role: str  # "user" o "assistant"
    content: str


class MemoriaBase(ABC):
    """Contrato que toda estrategia de memoria debe cumplir.

    Las implementaciones concretas (NoMemoria, VerbatimRAG, etc.)
    deciden internamente CÓMO almacenan y recuperan información;
    el loop de evaluación solo ve esta interfaz uniforme.
    """

    @abstractmethod
    def store(self, turn: Turn) -> None:
        """Almacenar un turno de conversación en memoria."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Devolver los fragmentos más relevantes para `query`.

        Args:
            query: La pregunta o contexto de consulta.
            top_k: Máximo número de fragmentos a devolver.

        Returns:
            Lista de strings ordenados de más a menos relevante.
            Puede ser vacía si no hay contexto relevante (caso típico
            de la estrategia baseline sin memoria).
        """
        ...
