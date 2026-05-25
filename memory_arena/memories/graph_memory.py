"""
memory_arena.memories.graph_memory
------------------------------------
Estrategia 5: memoria basada en grafo de conocimiento.

El LLM extrae triples (sujeto, predicado, objeto) de cada turno y los
almacena como aristas en un grafo dirigido (networkx). En retrieve se
identifican entidades de la query via LLM, se expande k-hop
bidireccional y se resuelven conflictos temporales devolviendo el
hecho más reciente (inspirado en Zep/Graphiti).
"""

import re

import networkx as nx
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.base import MemoriaBase, Turn

_EXTRACT_PROMPT = (
    "Extract factual triples (subject, predicate, object) from the text below.\n"
    "Return ONLY a JSON array of arrays, e.g.: [[\"Alice\",\"likes\",\"pizza\"]].\n"
    "If there are no clear facts, return [].\n\nText:\n"
)

_CHUNK_SIZE = 16000
_CHUNK_OVERLAP = 1500
_K_HOP = 2
_SIM_THRESHOLD = 0.3
_SEED_K = 10


class GraphMemory(MemoriaBase):
    """Memoria como grafo de conocimiento con extracción de triples via LLM."""

    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._turn_index: int = 0
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._node_embeddings: dict[str, np.ndarray] = {}
        self._node_embs_matrix: np.ndarray | None = None
        self._node_embs_index: list[str] = []
        self._undirected_cache: nx.Graph | None = None
        self._nlp = spacy.load("en_core_web_sm")

    def _invalidate_cache(self) -> None:
        self._undirected_cache = None
        self._node_embs_matrix = None
        self._node_embs_index = []

    def store(self, turn: Turn) -> None:
        triples = self._extract_triples(turn.content)
        for subj, pred, obj in triples:
            s, p, o = subj.lower(), pred.lower(), obj.lower()
            # Batch encoding de nodos nuevos
            new_nodes = [n for n in (s, o) if n not in self._node_embeddings]
            if new_nodes:
                embs = self._embedder.encode(
                    new_nodes, normalize_embeddings=True
                )
                self._node_embeddings.update(zip(new_nodes, embs))
            # Zep: marcar arista activa existente como histórica
            existing = [
                (key, data)
                for key, data in self.graph.get_edge_data(s, o, default={}).items()
                if data.get("label") == p and data.get("valid_until") is None
            ]
            for key, _ in existing:
                self.graph[s][o][key]["valid_until"] = self._turn_index
            self.graph.add_edge(s, o, label=p, t=self._turn_index, valid_until=None)
        if triples:
            self._invalidate_cache()
        self._turn_index += 1

    def _extract_entities(self, query: str) -> list[str]:
        doc = self._nlp(query)
        return [ent.text.lower().strip() for ent in doc.ents if ent.text.strip()]

    def retrieve(self, query: str, top_k: int = 15) -> list[str]:
        if self.graph.number_of_nodes() == 0:
            return []

        query_lower = query.lower()

        # Encodear query una sola vez al inicio
        query_emb = self._embedder.encode(
            query_lower, normalize_embeddings=True
        )

        # Extraer entidades de la query via spaCy NER
        entities = self._extract_entities(query)

        # Buscar seeds por intersección de palabras
        seeds: list[str] = []
        if entities:
            entity_word_sets = [set(e.split()) for e in entities]
            for node in self.graph.nodes:
                node_words = set(node.split())
                if any(ew & node_words for ew in entity_word_sets):
                    seeds.append(node)

        # Fallback semántico si el LLM no encontró entidades útiles
        if not seeds and self._node_embeddings:
            if self._node_embs_matrix is None:
                self._node_embs_index = list(self._node_embeddings.keys())
                self._node_embs_matrix = np.stack(
                    [self._node_embeddings[n] for n in self._node_embs_index]
                )
            sims = self._node_embs_matrix @ query_emb
            top_indices = np.argsort(sims)[-_SEED_K:][::-1]
            seeds = [self._node_embs_index[i] for i in top_indices
                     if sims[i] >= _SIM_THRESHOLD]

        # Fallback por grado si todo lo anterior falla
        if not seeds:
            seeds = sorted(
                self.graph.nodes,
                key=lambda n: self.graph.degree(n),
                reverse=True
            )[:_SEED_K]

        # Expansión k-hop bidireccional con cache
        if self._undirected_cache is None:
            self._undirected_cache = self.graph.to_undirected()

        expanded: set[str] = set()
        for seed in seeds:
            try:
                neighbors = nx.single_source_shortest_path_length(
                    self._undirected_cache, seed, cutoff=_K_HOP
                )
                expanded.update(neighbors.keys())
            except nx.NodeNotFound:
                continue

        # Recolectar triples separando vigentes de históricos
        seen: set[str] = set()
        current: list[tuple[int, str]] = []
        historical: list[tuple[int, str]] = []

        for node in expanded:
            all_edges = (
                list(self.graph.edges(node, data=True)) +
                list(self.graph.in_edges(node, data=True))
            )
            for u, v, data in all_edges:
                label = data.get("label", "relacionado con")
                if data.get("valid_until") is None:
                    triple_str = f"{u} {label} {v}"
                    bucket = current
                else:
                    triple_str = f"{u} {label} {v} (previously, now updated)"
                    bucket = historical
                if triple_str not in seen:
                    seen.add(triple_str)
                    bucket.append((data.get("t", 0), triple_str))

        # Rankear por similitud semántica del triple completo
        def rank_by_similarity(
            items: list[tuple[int, str]]
        ) -> list[tuple[float, int, str]]:
            if not items:
                return []
            texts = [triple for _, triple in items]
            embs = self._embedder.encode(texts, normalize_embeddings=True)
            sims = embs @ query_emb
            return [
                (float(sims[i]), t, triple)
                for i, (t, triple) in enumerate(items)
            ]

        current_ranked = rank_by_similarity(current)
        historical_ranked = rank_by_similarity(historical)

        current_ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        historical_ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

        combined = current_ranked + historical_ranked
        return [triple for _, _, triple in combined[:top_k]]

    def reset(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._node_embeddings = {}
        self._invalidate_cache()
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