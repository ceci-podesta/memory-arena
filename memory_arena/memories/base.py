"""
memory_arena.memories.base
---------------------------
Interfaz base para todas las estrategias de memoria.

Todas las estrategias implementan el mismo contrato — `store` para persistir
un turno de conversación, `retrieve` para devolver los fragmentos más
relevantes para una pregunta dada, `reset` para limpiar memoria entre
samples del benchmark. Gracias a esto, el loop de evaluación puede
intercambiar estrategias sin tocar nada más.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Turn:
    """Un turno de conversación: quién habló y qué dijo.

    Los campos `session_id` y `date` son opcionales. Las estrategias que
    no los usan (NoMemoria, verbatim+RAG naive) los ignoran; las que
    hacen razonamiento temporal (summarized, agentic, graph) los usan.
    """

    role: str  # "user" o "assistant"
    content: str
    session_id: str | None = None
    date: str | None = None


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

    @abstractmethod
    def reset(self) -> None:
        """Limpiar todo el estado de memoria.

        Se llama entre samples del benchmark, porque cada sample es una
        conversación independiente y no queremos que una estrategia con
        estado persistente (verbatim+RAG, summarized, etc.) mezcle info
        de samples distintos.
        """
        ...
