"""
memory_arena.memories.graph_memory
------------------------------------
Estrategia 5: memoria basada en grafo de conocimiento.

El LLM extrae triples (sujeto, predicado, objeto) de cada turno y los
almacena como aristas en un grafo dirigido (networkx). En retrieve se
identifican nodos mencionados en la query, se expande 1-hop y se
resuelven conflictos temporales devolviendo el hecho más reciente
(inspirado en Zep/Graphiti).
"""

import re

import networkx as nx

from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.base import MemoriaBase, Turn

_EXTRACT_PROMPT = (
    "Extract factual triples (subject, predicate, object) from the text below.\n"
    "Return ONLY a JSON array of arrays, e.g.: [[\"Alice\",\"likes\",\"pizza\"]].\n"
    "If there are no clear facts, return [].\n\nText:\n"
)

_CHUNK_SIZE = 16000
_CHUNK_OVERLAP = 1500


class GraphMemory(MemoriaBase):
    """Memoria como grafo de conocimiento con extracción de triples via LLM."""

    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._turn_index: int = 0

    def store(self, turn: Turn) -> None:
        for subj, pred, obj in self._extract_triples(turn.content):
            s, p, o = subj.lower(), pred.lower(), obj.lower()
            # Zep: si ya existe una arista activa con mismo predicado,
            # la marcamos como histórica (valid_until) en lugar de borrarla.
            existing = [
                (key, data)
                for key, data in self.graph.get_edge_data(s, o, default={}).items()
                if data.get("label") == p and data.get("valid_until") is None
            ]
            for key, _ in existing:
                self.graph[s][o][key]["valid_until"] = self._turn_index
            self.graph.add_edge(s, o, label=p, t=self._turn_index, valid_until=None)
        self._turn_index += 1

    def retrieve(self, query: str, top_k: int = 15) -> list[str]:
        if self.graph.number_of_nodes() == 0:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        seeds = [n for n in self.graph.nodes if any(word in n for word in query_words)]

        if not seeds:
            seeds = sorted(
                self.graph.nodes,
                key=lambda n: self.graph.degree(n),
                reverse=True
            )[:10]

        seeds = sorted(seeds, key=lambda n: self.graph.degree(n), reverse=True)

        seen: set[str] = set()
        result: list[tuple[int, str]] = []  # (timestamp, triple_str)

        for seed in seeds:
            for u, v, data in self.graph.edges(seed, data=True):
                label = data.get("label", "relacionado con")
                if data.get("valid_until") is None:
                    triple_str = f"{u} {label} {v}"
                else:
                    triple_str = f"{u} {label} {v} (previously, now updated)"
                if triple_str not in seen:
                    seen.add(triple_str)
                    result.append((data.get("t", 0), triple_str))
            for u, v, data in self.graph.in_edges(seed, data=True):
                label = data.get("label", "relacionado con")
                if data.get("valid_until") is None:
                    triple_str = f"{u} {label} {v}"
                else:
                    triple_str = f"{u} {label} {v} (previously, now updated)"
                if triple_str not in seen:
                    seen.add(triple_str)
                    result.append((data.get("t", 0), triple_str))

        def relevance(item: tuple[int, str]) -> tuple[int, int]:
            t, triple = item
            word_match = len(set(triple.lower().split()) & query_words)
            return (word_match, t)  # primero relevancia, después más reciente

        result.sort(key=relevance, reverse=True)
        return [triple for _, triple in result[:top_k]]

    def reset(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._turn_index = 0

    def _extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + _CHUNK_SIZE
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - _CHUNK_OVERLAP

        pattern = r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'
        seen: set[tuple[str, str, str]] = set()
        result: list[tuple[str, str, str]] = []
        for chunk in chunks:
            messages = [{"role": "user", "content": _EXTRACT_PROMPT + chunk}]
            raw = self.llm.chat(messages, max_tokens=8000)
            for triple in re.findall(pattern, raw):
                if triple not in seen:
                    seen.add(triple)
                    result.append(triple)
        return result