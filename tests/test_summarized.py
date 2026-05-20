"""Tests unitarios para SummarizedMemory."""

import pytest

from memory_arena.memories.base import Turn
from memory_arena.memories.summarized import SummarizedMemory


class FakeLLM:
    def __init__(self, response: str = "Resumen actualizado") -> None:
        self.response = response
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> str:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.response


def test_summarized_inicia_vacia():
    mem = SummarizedMemory(llm=FakeLLM())

    assert mem.summary == ""
    assert mem.recent_turns == []
    assert mem.turn_count == 0
    assert mem.retrieve("algo") == []


def test_store_turno_conversacional_queda_en_recent_turns():
    llm = FakeLLM()
    mem = SummarizedMemory(llm=llm, summarize_every=3)

    turn = Turn(role="user", content="Me gustan los thrillers.", session_id="s1")
    mem.store(turn)

    assert mem.summary == ""
    assert mem.recent_turns == [turn]
    assert mem.turn_count == 1
    assert llm.calls == []


def test_retrieve_consolida_turnos_pendientes_si_no_hay_summary():
    llm = FakeLLM(response="El usuario se llama David.")
    mem = SummarizedMemory(llm=llm, keep_recent=2)
    mem.store(Turn(role="user", content="Me llamo David.", session_id="s1"))
    mem.store(Turn(role="assistant", content="Mucho gusto.", session_id="s1"))

    context = mem.retrieve("Como se llama el usuario?")

    assert len(context) == 1
    assert context == ["Memory summary:\nEl usuario se llama David."]
    assert mem.recent_turns == []
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "Me llamo David." in prompt
    assert "Mucho gusto." in prompt


def test_store_resume_al_llegar_a_summarize_every():
    llm = FakeLLM(response="El usuario se llama David y trabaja en NLP.")
    mem = SummarizedMemory(llm=llm, summarize_every=2, summary_max_tokens=123)

    mem.store(Turn(role="user", content="Me llamo David."))
    mem.store(Turn(role="user", content="Trabajo en NLP."))

    assert mem.summary == "El usuario se llama David y trabaja en NLP."
    assert mem.recent_turns == []
    assert mem.turn_count == 2
    assert len(llm.calls) == 1
    assert llm.calls[0]["max_tokens"] == 123
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "Existing memory summary:" in prompt
    assert "Me llamo David." in prompt
    assert "Trabajo en NLP." in prompt


def test_document_se_resume_inmediatamente():
    llm = FakeLLM(response="Resumen del documento.")
    mem = SummarizedMemory(llm=llm, summarize_every=8)

    mem.store(Turn(role="document", content="Documento largo.", session_id="doc1"))

    assert mem.summary == "Resumen del documento."
    assert mem.recent_turns == []
    assert mem.turn_count == 1
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "[session=doc1 | role=document] Documento largo." in prompt


def test_documento_largo_se_resume_por_chunks():
    llm = FakeLLM(response="Resumen parcial")
    mem = SummarizedMemory(
        llm=llm,
        document_chunk_chars=10,
        summary_max_tokens=77,
    )

    mem.store(Turn(role="document", content="uno dos tres cuatro", session_id="doc1"))

    # 3 chunks parciales + 1 reduce final.
    assert len(llm.calls) == 4
    assert mem.summary == "Resumen parcial"
    assert mem.recent_turns == []
    assert all(call["max_tokens"] == 77 for call in llm.calls)
    assert "Document chunk 1/3" in llm.calls[0]["messages"][0]["content"]
    assert "Document chunk 2/3" in llm.calls[1]["messages"][0]["content"]
    assert "Document chunk 3/3" in llm.calls[2]["messages"][0]["content"]
    assert "Chunk 1 summary" in llm.calls[3]["messages"][0]["content"]
    assert "Chunk 2 summary" in llm.calls[3]["messages"][0]["content"]
    assert "Chunk 3 summary" in llm.calls[3]["messages"][0]["content"]


def test_max_document_chunks_limita_chunks_procesados():
    llm = FakeLLM(response="Resumen limitado")
    mem = SummarizedMemory(
        llm=llm,
        document_chunk_chars=5,
        max_document_chunks=2,
    )

    mem.store(Turn(role="document", content="aaaaa bbbbb ccccc ddddd", session_id="doc1"))

    # max_document_chunks=2 => 2 chunks parciales + 1 reduce final.
    assert len(llm.calls) == 3
    assert "Document chunk 1/2" in llm.calls[0]["messages"][0]["content"]
    assert "Document chunk 2/2" in llm.calls[1]["messages"][0]["content"]
    assert "ccccc" not in llm.calls[2]["messages"][0]["content"]


def test_retrieve_consolida_summary_y_recientes():
    llm = FakeLLM(response="Resumen combinado.")
    mem = SummarizedMemory(llm=llm, keep_recent=1)
    mem.summary = "El usuario prefiere respuestas breves."
    mem.store(Turn(role="user", content="Tambien quiere ejemplos simples."))

    context = mem.retrieve("Que prefiere el usuario?", top_k=5)

    assert context == ["Memory summary:\nResumen combinado."]
    assert mem.recent_turns == []
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "El usuario prefiere respuestas breves." in prompt
    assert "Tambien quiere ejemplos simples." in prompt


def test_retrieve_respeta_top_k():
    mem = SummarizedMemory(llm=FakeLLM(response="Resumen combinado."), keep_recent=1)
    mem.summary = "Resumen existente."
    mem.store(Turn(role="user", content="Turno reciente."))

    context = mem.retrieve("query", top_k=1)

    assert context == ["Memory summary:\nResumen combinado."]


def test_reset_limpia_estado():
    mem = SummarizedMemory(llm=FakeLLM(), keep_recent=1)
    mem.summary = "Resumen existente."
    mem.store(Turn(role="user", content="Turno reciente."))

    mem.reset()

    assert mem.summary == ""
    assert mem.recent_turns == []
    assert mem.turn_count == 0
    assert mem.retrieve("query") == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"summarize_every": 0}, "summarize_every"),
        ({"keep_recent": -1}, "keep_recent"),
        ({"summary_max_tokens": 0}, "summary_max_tokens"),
        ({"document_chunk_chars": 0}, "document_chunk_chars"),
        ({"max_document_chunks": 0}, "max_document_chunks"),
    ],
)
def test_parametros_invalidos_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SummarizedMemory(llm=FakeLLM(), **kwargs)
