"""
memory_arena.memories.verbatim_rag
-----------------------------------
Estrategia 2: Verbatim + RAG.

Guarda cada turno textualmente y recupera los fragmentos más relevantes
por similitud semántica usando sentence-transformers + cosine similarity.

Para turnos cortos (conversaciones de LongMemEval) se guarda un embedding
por turno. Para documentos largos (MAB inject-once) se chunkea el contenido
en fragmentos de CHUNK_WORDS palabras con overlap de CHUNK_OVERLAP palabras
antes de embedear, para que retrieve pueda encontrar la parte relevante.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from memory_arena.memories.base import MemoriaBase, Turn

CHUNK_WORDS = 100
CHUNK_OVERLAP = 20


class VerbatimRAG(MemoriaBase):
    """Memoria verbatim con recuperación por embeddings (dense retrieval).

    Turnos cortos se guardan como un único chunk. Documentos largos se
    dividen en chunks de CHUNK_WORDS palabras con overlap de CHUNK_OVERLAP
    para evitar cortar ideas en los bordes.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None

    def store(self, turn: Turn) -> None:
        for chunk in self._chunk(turn.content):
            embedding = self._model.encode(chunk, convert_to_numpy=True)
            self._texts.append(chunk)
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

    def _chunk(self, text: str) -> list[str]:
        """Divide text en chunks de CHUNK_WORDS palabras con overlap.

        Si el texto es más corto que CHUNK_WORDS palabras, devuelve el
        texto completo como un único chunk (caso típico de LongMemEval).
        """
        words = text.split()
        if len(words) <= CHUNK_WORDS:
            return [text]

        chunks = []
        step = CHUNK_WORDS - CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + CHUNK_WORDS])
            chunks.append(chunk)
            if i + CHUNK_WORDS >= len(words):
                break
        return chunks