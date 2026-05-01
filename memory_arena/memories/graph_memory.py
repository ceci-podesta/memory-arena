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

    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()
        self.graph: nx.DiGraph = nx.DiGraph()

    def store(self, turn: Turn) -> None:
        triples = self._extract_triples(turn.content)
        for subj, pred, obj in triples:
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
