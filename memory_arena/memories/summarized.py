"""
memory_arena.memories.summarized
---------------------------------
Estrategia 3: memoria resumida.

Mantiene un resumen acumulativo del historial como memoria de largo plazo y
una ventana corta de turnos recientes todavia no consolidados. En LongMemEval
recibe muchos turnos conversacionales; en MAB recibe un documento largo como
un Turn sintetico con role="document".
"""

from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.base import MemoriaBase, Turn


class SummarizedMemory(MemoriaBase):
    """Memoria basada en un resumen rolling generado por LLM.

    La estrategia no hace retrieval semantico: retrieve() devuelve una vista
    compacta global del historial (summary) y, si existen, los ultimos turnos
    que todavia no fueron incorporados al resumen.
    """

    def __init__(
        self,
        llm: OllamaClient | None = None,
        summarize_every: int = 8,
        keep_recent: int = 3,
        summary_max_tokens: int = 384,
        document_chunk_chars: int = 30000,
        max_document_chunks: int | None = None,
    ) -> None:
        if summarize_every <= 0:
            raise ValueError("summarize_every debe ser mayor a 0")
        if keep_recent < 0:
            raise ValueError("keep_recent no puede ser negativo")
        if summary_max_tokens <= 0:
            raise ValueError("summary_max_tokens debe ser mayor a 0")
        if document_chunk_chars <= 0:
            raise ValueError("document_chunk_chars debe ser mayor a 0")
        if max_document_chunks is not None and max_document_chunks <= 0:
            raise ValueError("max_document_chunks debe ser mayor a 0 o None")

        self.llm = llm if llm is not None else OllamaClient()
        self.summarize_every = summarize_every
        self.keep_recent = keep_recent
        self.summary_max_tokens = summary_max_tokens
        self.document_chunk_chars = document_chunk_chars
        self.max_document_chunks = max_document_chunks
        self.summary = ""
        self.recent_turns: list[Turn] = []
        self.turn_count = 0

    def store(self, turn: Turn) -> None:
        """Guardar un turno y actualizar el resumen cuando corresponde."""
        if not turn.content.strip():
            return

        self.turn_count += 1

        if turn.role == "document":
            self.summary = self._summarize_document(turn)
            self.recent_turns = []
            return

        self.recent_turns.append(turn)
        if len(self.recent_turns) >= self.summarize_every:
            self.summary = self._summarize(
                previous_summary=self.summary,
                new_content=self._format_turns(self.recent_turns),
            )
            self.recent_turns = []

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Consolidar pendientes y devolver el resumen vigente.

        `query` se acepta por contrato, pero esta estrategia no lo usa para
        seleccionar contexto: siempre devuelve la memoria comprimida global.
        """
        if top_k <= 0:
            return []

        self._flush_recent_turns()

        context: list[str] = []

        if self.summary.strip():
            context.append(f"Memory summary:\n{self.summary.strip()}")

        recent = self.recent_turns[-self.keep_recent:] if self.keep_recent else []
        if recent:
            context.append(
                "Recent unsummarized turns:\n" + self._format_turns(recent)
            )

        return context[:top_k]

    def reset(self) -> None:
        """Limpiar la memoria entre samples independientes del benchmark."""
        self.summary = ""
        self.recent_turns = []
        self.turn_count = 0

    def _summarize(self, previous_summary: str, new_content: str) -> str:
        prompt = _build_summary_prompt(previous_summary, new_content)
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=self.summary_max_tokens,
        ).strip()

    def _summarize_document(self, turn: Turn) -> str:
        chunks = _split_text(turn.content.strip(), self.document_chunk_chars)
        if self.max_document_chunks is not None:
            chunks = chunks[: self.max_document_chunks]

        if len(chunks) <= 1:
            return self._summarize(
                previous_summary=self.summary,
                new_content=self._format_turn(turn),
            )

        partial_summaries: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_turn = Turn(
                role=turn.role,
                content=(
                    f"Document chunk {idx}/{len(chunks)} "
                    f"from {turn.session_id or 'unknown session'}:\n{chunk}"
                ),
                session_id=turn.session_id,
                date=turn.date,
            )
            partial_summaries.append(
                self._summarize(
                    previous_summary="",
                    new_content=self._format_turn(chunk_turn),
                )
            )

        combined = "\n\n".join(
            f"Chunk {idx} summary:\n{summary}"
            for idx, summary in enumerate(partial_summaries, start=1)
        )
        return self._summarize(
            previous_summary=self.summary,
            new_content=combined,
        )

    def _flush_recent_turns(self) -> None:
        if not self.recent_turns:
            return
        self.summary = self._summarize(
            previous_summary=self.summary,
            new_content=self._format_turns(self.recent_turns),
        )
        self.recent_turns = []

    def _format_turns(self, turns: list[Turn]) -> str:
        return "\n".join(self._format_turn(turn) for turn in turns)

    def _format_turn(self, turn: Turn) -> str:
        metadata = []
        if turn.date:
            metadata.append(f"date={turn.date}")
        if turn.session_id:
            metadata.append(f"session={turn.session_id}")
        metadata.append(f"role={turn.role}")

        meta_block = " | ".join(metadata)
        return f"[{meta_block}] {turn.content.strip()}"


def _build_summary_prompt(previous_summary: str, new_content: str) -> str:
    previous = previous_summary.strip() or "(empty)"
    return (
        "You are maintaining a compact long-term memory for an LLM agent.\n\n"
        "Update the existing memory summary using the new content.\n\n"
        "Rules:\n"
        "- Preserve facts that may help answer future questions.\n"
        "- Preserve names, dates, preferences, decisions, entities, goals, and constraints.\n"
        "- Preserve temporal information when available.\n"
        "- Preserve contradictions or updates. If newer information changes older "
        "information, keep the newest version and mention the update.\n"
        "- Do not include irrelevant small talk.\n"
        "- Do not invent facts.\n"
        "- Be concise but specific.\n"
        "- Write only the updated memory summary.\n\n"
        f"Existing memory summary:\n{previous}\n\n"
        f"New content:\n{new_content.strip()}\n\n"
        "Updated memory summary:"
    )


def _split_text(text: str, chunk_chars: int) -> list[str]:
    """Divide texto en chunks por caracteres, intentando cortar en whitespace."""
    stripped = text.strip()
    if not stripped:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(stripped)

    while start < text_len:
        end = min(start + chunk_chars, text_len)
        if end < text_len:
            split_at = stripped.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

        while start < text_len and stripped[start].isspace():
            start += 1

    return chunks
