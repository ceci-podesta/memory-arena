"""
Tests para GraphMemory (Estrategia 5: knowledge graph).

El LLM se mockea para no depender de Ollama en CI.
"""

from unittest.mock import MagicMock

import pytest

from memory_arena.memories.base import MemoriaBase, Turn
from memory_arena.memories.graph_memory import GraphMemory


def make_graph_memory(triples_response: str = '[]') -> GraphMemory:
    """GraphMemory con LLM mockeado que devuelve triples_response."""
    llm = MagicMock()
    llm.chat.return_value = triples_response
    return GraphMemory(llm=llm)


def test_graph_memory_es_memoria_base():
    mem = make_graph_memory()
    assert isinstance(mem, MemoriaBase)


def test_store_sin_triples_no_agrega_nodos():
    mem = make_graph_memory(triples_response='[]')
    mem.store(Turn(role="user", content="nada relevante"))
    assert mem.graph.number_of_nodes() == 0


def test_store_agrega_triples_al_grafo():
    mem = make_graph_memory(triples_response='[["alice","likes","pizza"]]')
    mem.store(Turn(role="user", content="Alice likes pizza"))
    assert mem.graph.has_edge("alice", "pizza")
    assert mem.graph["alice"]["pizza"]["label"] == "likes"


def test_store_multiples_triples():
    mem = make_graph_memory(
        triples_response='[["alice","likes","pizza"],["bob","lives_in","berlin"]]'
    )
    mem.store(Turn(role="user", content="..."))
    assert mem.graph.number_of_edges() == 2


def test_retrieve_vacio_si_grafo_vacio():
    mem = make_graph_memory()
    assert mem.retrieve("cualquier query") == []


def test_retrieve_devuelve_lista_de_strings():
    mem = make_graph_memory(triples_response='[["alice","likes","pizza"]]')
    mem.store(Turn(role="user", content="..."))
    result = mem.retrieve("alice", top_k=5)
    assert isinstance(result, list)
    assert all(isinstance(r, str) for r in result)


def test_retrieve_match_por_nodo_en_query():
    mem = make_graph_memory(triples_response='[["alice","likes","pizza"]]')
    mem.store(Turn(role="user", content="..."))
    result = mem.retrieve("what does alice like?", top_k=5)
    assert any("alice" in r for r in result)


def test_retrieve_respeta_top_k():
    mem = make_graph_memory()
    mem.llm.chat.return_value = '[["alice","r","b"],["alice","r","c"],["alice","r","d"],["alice","r","e"]]'
    mem.store(Turn(role="user", content="..."))
    result = mem.retrieve("alice", top_k=2)
    assert len(result) <= 2


def test_inject_once_query_many():
    """El grafo no se resetea entre queries del mismo sample (patrón MAB)."""
    mem = make_graph_memory(triples_response='[["alice","likes","pizza"]]')
    mem.store(Turn(role="document", content="...", session_id="s1"))
    r1 = mem.retrieve("alice?", top_k=5)
    r2 = mem.retrieve("pizza?", top_k=5)
    assert len(r1) > 0
    assert len(r2) > 0


def test_store_tolera_respuesta_llm_con_texto_libre():
    mem = make_graph_memory(
        triples_response='Here are the triples:\n```\n[["alice","likes","pizza"]]\n```'
    )
    mem.store(Turn(role="user", content="..."))
    assert mem.graph.has_edge("alice", "pizza")


def test_reset_vacia_el_grafo():
    mem = make_graph_memory(triples_response='[["alice","likes","pizza"]]')
    mem.store(Turn(role="user", content="..."))
    assert mem.graph.number_of_nodes() > 0
    mem.reset()
    assert mem.graph.number_of_nodes() == 0
    assert mem.retrieve("alice") == []


def test_store_normaliza_a_minusculas():
    mem = make_graph_memory(triples_response='[["Alice","Likes","Pizza"]]')
    mem.store(Turn(role="user", content="..."))
    assert mem.graph.has_edge("alice", "pizza")
