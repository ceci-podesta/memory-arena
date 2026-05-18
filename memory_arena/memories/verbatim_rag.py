"""
memory_arena.memories.verbatim_rag
-----------------------------------
Estrategia 2: Verbatim + RAG.

Guarda cada turno textualmente y recupera los fragmentos más relevantes
por similitud semántica usando sentence-transformers + cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from memory_arena.memories.base import MemoriaBase, Turn


class VerbatimRAG(MemoriaBase):
    """Memoria verbatim con recuperación por embeddings (dense retrieval).

    Cada turno se guarda tal cual (verbatim). En retrieve, se embeddea la
    query y se devuelven los top_k chunks más similares por cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None

    def store(self, turn: Turn) -> None:
        text = turn.content
        embedding = self._model.encode(text, convert_to_numpy=True)

        self._texts.append(text)
        if self._embeddings is None:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if not self._texts:
            return []

        query_embedding = self._model.encode(query, convert_to_numpy=True)

        # Cosine similarity: (a · b) / (||a|| * ||b||)
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = self._embeddings @ query_embedding / (norms * query_norm + 1e-10)

        top_k = min(top_k, len(self._texts))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self._texts[i] for i in top_indices]

    def reset(self) -> None:
        self._texts = []
        self._embeddings = None
