"""
memory_arena.memories.graph_memory
------------------------------------
Estrategia 5: memoria basada en grafo de conocimiento.

El LLM extrae triples (sujeto, predicado, objeto) de cada turno y los
almacena como aristas en un grafo dirigido (networkx). En retrieve se
identifican nodos mencionados en la query y se expande 1-hop para
construir el contexto.
"""

import json
import re

import networkx as nx

from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.base import MemoriaBase, Turn

_EXTRACT_PROMPT = (
    "Extract factual triples (subject, predicate, object) from the text below.\n"
    "Return ONLY a JSON array of arrays, e.g.: [[\"Alice\",\"likes\",\"pizza\"]].\n"
    "If there are no clear facts, return [].\n\nText:\n"
)


class GraphMemory(MemoriaBase):
    """Memoria como grafo de conocimiento con extracción de triples via LLM."""

    def __init__(
        self,
        llm: OllamaClient | None = None,
        chunk_tokens: int = 600,
        chunk_overlap_tokens: int = 0,
    ):
        self.llm = llm or OllamaClient()
        self.graph: nx.DiGraph = nx.DiGraph()
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def store(self, turn: Turn) -> None:
        text = turn.content
        if _approx_tokens(text) <= self.chunk_tokens:
            chunks = [text]
        else:
            chunks = self._chunk_text(text)
        for chunk in chunks:
            if not chunk.strip():
                continue
            for subj, pred, obj in self._extract_triples(chunk):
                self.graph.add_edge(subj.lower(), obj.lower(), label=pred.lower())

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if self.graph.number_of_nodes() == 0:
            return []

        query_lower = query.lower()
        seeds = [n for n in self.graph.nodes if n in query_lower]

        if not seeds:
            seeds = sorted(self.graph.nodes, key=lambda n: self.graph.degree(n), reverse=True)[:5]

        seen: set[str] = set()
        result: list[str] = []

        for seed in seeds:
            for u, v, data in self.graph.edges(seed, data=True):
                triple_str = f"{u} {data.get('label', 'relacionado con')} {v}"
                if triple_str not in seen:
                    seen.add(triple_str)
                    result.append(triple_str)
            for u, v, data in self.graph.in_edges(seed, data=True):
                triple_str = f"{u} {data.get('label', 'relacionado con')} {v}"
                if triple_str not in seen:
                    seen.add(triple_str)
                    result.append(triple_str)
            if len(result) >= top_k:
                break

        return result[:top_k]

    def reset(self) -> None:
        self.graph = nx.DiGraph()

    def _extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        messages = [{"role": "user", "content": _EXTRACT_PROMPT + text}]
        raw = self.llm.chat(messages, max_tokens=256)
        # Extraer todos los triples ["s","p","o"] directamente, tolerando
        # respuestas con texto libre, múltiples arrays o bloques markdown.
        pattern = r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'
        return re.findall(pattern, raw)

    def _chunk_text(self, text: str) -> list[str]:
        units = _split_into_units(text)
        chunks: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for unit in units:
            u_tokens = _approx_tokens(unit)
            if u_tokens > self.chunk_tokens:
                if buf:
                    chunks.append(" ".join(buf))
                    buf, buf_tokens = [], 0
                chunks.extend(_split_oversized(unit, self.chunk_tokens))
                continue
            if buf_tokens + u_tokens > self.chunk_tokens and buf:
                chunks.append(" ".join(buf))
                buf, buf_tokens = [], 0
            buf.append(unit)
            buf_tokens += u_tokens
        if buf:
            chunks.append(" ".join(buf))
        return chunks


def _approx_tokens(s: str) -> int:
    return max(len(s) // 4, len(s.split()))


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_into_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
        units.extend(sentences or [paragraph])
    return units


def _split_oversized(unit: str, chunk_tokens: int) -> list[str]:
    words = unit.split()
    if not words:
        return []
    words_per_chunk = max(1, int(chunk_tokens * 0.75))
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]
