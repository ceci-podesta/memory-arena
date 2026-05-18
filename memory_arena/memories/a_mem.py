"""
memory_arena.memories.a_mem
----------------------------
Estrategia 4: Agentic Memory (A-MEM).

Implementación del paper "A-MEM: Agentic Memory for LLM Agents"
(Xu et al., NeurIPS 2025, arXiv:2502.12110). Sigue el pipeline ROBUSTO
del repo oficial (memory_layer_robust.py + llm_text_parsers.py),
diseñado para modelos chicos como llama3.2:3b.

Pipeline en store(turn) -> por cada contenido atómico:
  1. analyze_content        (LLM 1)  -> K, X, G de la nota.
  2. embedding sobre concat(content, K, X, G).
  3. Si hay vecinos en memoria:
     a. top-k cosine -> neighbor_ids.
     b. evolution_decision (LLM 2)  -> NO_EVOLUTION | STRENGTHEN |
                                       UPDATE_NEIGHBOR | STRENGTHEN_AND_UPDATE.
     c. si decision incluye STRENGTHEN:
        strengthen_details (LLM 3)  -> links + tags de la nota nueva.
     d. si decision incluye UPDATE_NEIGHBOR:
        update_neighbors    (LLM 4)  -> context/tags por vecino + re-embed.

retrieve(query, top_k): cosine puro contra todas las notas. Sin LLM.

Detección por turn.role:
  - 'document' (MAB inject-once): chunkear contenido, crear N notas.
  - 'user' | 'assistant' (LongMemEval): 1 nota directa.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.a_mem_prompts import (
    build_analyze_content_prompt,
    build_evolution_decision_prompt,
    build_strengthen_details_prompt,
    build_update_neighbors_prompt,
    parse_analyze_content,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
)
from memory_arena.memories.base import MemoriaBase, Turn

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------

DEFAULT_LINK_TOP_K: int = 5
"""Top-k vecinos en evolution. Default del repo oficial."""

DEFAULT_CHUNK_SIZE: int = 1500
"""Chars máximos por chunk en Turn role='document' (MAB)."""

DEFAULT_CHUNK_OVERLAP: int = 100
"""Solapamiento entre chunks consecutivos en MAB."""

DEFAULT_EMBEDDER_MODEL: str = "all-MiniLM-L6-v2"

DEFAULT_EMBEDDER_DEVICE: str = "cpu"
"""CPU para no pelear VRAM con Ollama. Latencia ~50ms/embedding, despreciable."""


# ----------------------------------------------------------------------------
# Note dataclass
# ----------------------------------------------------------------------------


@dataclass
class Note:
    """Nota A-MEM (m_i del paper, sin t_i)."""

    note_id: int
    content: str  # c_i
    keywords: list[str] = field(default_factory=list)  # K_i
    tags: list[str] = field(default_factory=list)  # G_i
    context_desc: str = ""  # X_i
    embedding: np.ndarray | None = None  # e_i
    linked_ids: list[int] = field(default_factory=list)  # L_i


# ----------------------------------------------------------------------------
# AgenticMemory
# ----------------------------------------------------------------------------


class AgenticMemory(MemoriaBase):
    """Estrategia A-MEM: escritura agéntica, lectura por similitud."""

    def __init__(
        self,
        llm: OllamaClient | None = None,
        embedder: "SentenceTransformer | None" = None,
        link_top_k: int = DEFAULT_LINK_TOP_K,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        verbose: bool = False,
    ) -> None:
        if llm is None:
            llm = OllamaClient()
        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(
                DEFAULT_EMBEDDER_MODEL, device=DEFAULT_EMBEDDER_DEVICE
            )
        self._llm = llm
        self._embedder = embedder
        self._link_top_k = link_top_k
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._verbose = verbose

        self._notes: list[Note] = []
        self._next_id: int = 0

    # ----- API pública (contrato MemoriaBase) -----

    def store(self, turn: Turn) -> None:
        if turn.role == "document":
            chunks = self._chunk_document(turn.content)
        else:
            chunks = [turn.content]
        for chunk in chunks:
            if chunk.strip():
                self._ingest_one(chunk)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if not self._notes:
            return []
        query_emb = self._embed(query)
        scored = [
            (i, self._cosine(query_emb, n.embedding))
            for i, n in enumerate(self._notes)
            if n.embedding is not None
        ]
        scored.sort(key=lambda x: -x[1])
        return [self._notes[i].content for i, _ in scored[:top_k]]

    def reset(self) -> None:
        self._notes = []
        self._next_id = 0

    # ----- Pipeline interno -----

    def _ingest_one(self, content: str) -> None:
        new_note = self._create_note(content)
        self._notes.append(new_note)
        if len(self._notes) > 1:
            self._maybe_evolve(new_note)

    def _create_note(self, content: str) -> Note:
        prompt = build_analyze_content_prompt(content)
        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            if self._verbose:
                print(f"[a-mem] analyze_content fail: {e}")
            raw = ""

        analysis = parse_analyze_content(raw, content=content)
        note = Note(
            note_id=self._next_id,
            content=content,
            keywords=analysis["keywords"],
            tags=analysis["tags"],
            context_desc=analysis["context"],
        )
        note.embedding = self._embed_note(note)
        self._next_id += 1
        return note

    def _maybe_evolve(self, new_note: Note) -> None:
        neighbor_ids = self._cosine_top_k(
            new_note.embedding,
            k=self._link_top_k,
            exclude_id=new_note.note_id,
        )
        if not neighbor_ids:
            return

        neighbor_block = self._build_neighbor_block(neighbor_ids)

        dec_prompt = build_evolution_decision_prompt(
            context=new_note.context_desc,
            content=new_note.content,
            keywords=new_note.keywords,
            nearest_neighbors_memories=neighbor_block,
        )
        try:
            dec_raw = self._llm.chat([{"role": "user", "content": dec_prompt}])
        except Exception as e:
            if self._verbose:
                print(f"[a-mem] evolution_decision fail: {e}")
            return

        decision = parse_evolution_decision(dec_raw)["decision"]
        if self._verbose:
            print(f"[a-mem] note {new_note.note_id} -> decision: {decision}")

        if decision == "NO_EVOLUTION":
            return

        do_strengthen = decision in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
        do_update = decision in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

        if do_strengthen:
            self._apply_strengthen(new_note, neighbor_ids, neighbor_block)
        if do_update:
            self._apply_update_neighbors(new_note, neighbor_ids, neighbor_block)

    def _apply_strengthen(
        self,
        new_note: Note,
        neighbor_ids: list[int],
        neighbor_block: str,
    ) -> None:
        prompt = build_strengthen_details_prompt(
            content=new_note.content,
            keywords=new_note.keywords,
            nearest_neighbors_memories=neighbor_block,
        )
        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            if self._verbose:
                print(f"[a-mem] strengthen_details fail: {e}")
            return

        details = parse_strengthen_details(raw)
        # connections vienen como índices relativos al neighbor_block (0..k-1).
        for rel_idx in details["connections"]:
            if 0 <= rel_idx < len(neighbor_ids):
                real_id = neighbor_ids[rel_idx]
                if real_id not in new_note.linked_ids:
                    new_note.linked_ids.append(real_id)

        if details["tags"]:
            new_note.tags = details["tags"]
            new_note.embedding = self._embed_note(new_note)

    def _apply_update_neighbors(
        self,
        new_note: Note,
        neighbor_ids: list[int],
        neighbor_block: str,
    ) -> None:
        prompt = build_update_neighbors_prompt(
            content=new_note.content,
            context=new_note.context_desc,
            nearest_neighbors_memories=neighbor_block,
            neighbor_count=len(neighbor_ids),
        )
        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            if self._verbose:
                print(f"[a-mem] update_neighbors fail: {e}")
            return

        updates = parse_update_neighbors(raw, num_neighbors=len(neighbor_ids))
        for rel_idx, upd in enumerate(updates):
            if not upd["context"] and not upd["tags"]:
                continue
            if rel_idx >= len(neighbor_ids):
                break
            neighbor = self._get_note_by_id(neighbor_ids[rel_idx])
            if neighbor is None:
                continue

            changed = False
            if upd["context"]:
                neighbor.context_desc = upd["context"]
                changed = True
            if upd["tags"]:
                neighbor.tags = upd["tags"]
                changed = True

            # Decisión #4 cerrada: recomputar embedding al cambiar K/G/X.
            if changed:
                neighbor.embedding = self._embed_note(neighbor)

    # ----- Chunking (Turn role='document' / MAB) -----

    def _chunk_document(self, content: str) -> list[str]:
        text = self._flatten_if_python_literal(content)
        return self._recursive_split(text)

    def _flatten_if_python_literal(self, content: str) -> str:
        stripped = content.strip()
        if not stripped.startswith("["):
            return content
        try:
            obj = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return content
        return self._flatten_chat_structure(obj)

    def _flatten_chat_structure(self, obj: object) -> str:
        parts: list[str] = []
        self._collect_flatten(obj, parts)
        return "\n\n".join(p for p in parts if p)

    def _collect_flatten(self, obj: object, parts: list[str]) -> None:
        if isinstance(obj, dict):
            role = obj.get("role")
            content = obj.get("content")
            if role and content:
                parts.append(f"{role}: {content}")
            else:
                for v in obj.values():
                    self._collect_flatten(v, parts)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_flatten(item, parts)
        elif isinstance(obj, str):
            parts.append(obj)

    def _recursive_split(self, text: str) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        for sep in ("\n\n", "\n"):
            if sep in text:
                pieces = text.split(sep)
                chunks: list[str] = []
                buffer = ""
                for piece in pieces:
                    candidate = buffer + (sep if buffer else "") + piece
                    if len(candidate) <= self._chunk_size:
                        buffer = candidate
                    else:
                        if buffer:
                            chunks.append(buffer)
                        if len(piece) <= self._chunk_size:
                            buffer = piece
                        else:
                            chunks.extend(self._char_split_with_overlap(piece))
                            buffer = ""
                if buffer:
                    chunks.append(buffer)
                return [c for c in chunks if c.strip()]

        return self._char_split_with_overlap(text)

    def _char_split_with_overlap(self, text: str) -> list[str]:
        chunks: list[str] = []
        step = max(1, self._chunk_size - self._chunk_overlap)
        for start in range(0, len(text), step):
            chunk = text[start : start + self._chunk_size]
            if chunk.strip():
                chunks.append(chunk)
            if start + self._chunk_size >= len(text):
                break
        return chunks

    # ----- Embedding + similitud -----

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(text, convert_to_numpy=True)

    def _embed_note(self, note: Note) -> np.ndarray:
        """Embedding sobre concat(content, K, X, G). Eq. 3 del paper."""
        parts = [
            note.content,
            ", ".join(note.keywords),
            note.context_desc,
            ", ".join(note.tags),
        ]
        return self._embed(" | ".join(p for p in parts if p))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _cosine_top_k(
        self,
        query_emb: np.ndarray,
        k: int,
        exclude_id: int | None = None,
    ) -> list[int]:
        scored: list[tuple[int, float]] = []
        for note in self._notes:
            if note.embedding is None:
                continue
            if exclude_id is not None and note.note_id == exclude_id:
                continue
            scored.append(
                (note.note_id, self._cosine(query_emb, note.embedding))
            )
        scored.sort(key=lambda x: -x[1])
        return [nid for nid, _ in scored[:k]]

    # ----- Utilidades internas -----

    def _get_note_by_id(self, note_id: int) -> Note | None:
        for note in self._notes:
            if note.note_id == note_id:
                return note
        return None

    def _build_neighbor_block(self, neighbor_ids: list[int]) -> str:
        """Formatea vecinos para los prompts, usando índices RELATIVOS (0..k-1)."""
        lines: list[str] = []
        for rel_idx, nid in enumerate(neighbor_ids):
            note = self._get_note_by_id(nid)
            if note is None:
                continue
            lines.append(
                f"[{rel_idx}] content: {note.content} "
                f"| context: {note.context_desc} "
                f"| keywords: {', '.join(note.keywords)} "
                f"| tags: {', '.join(note.tags)}"
            )
        return "\n".join(lines)
