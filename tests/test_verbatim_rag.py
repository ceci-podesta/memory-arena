"""
Tests para la estrategia VerbatimRAG (Estrategia 2).
"""

import pytest

from memory_arena.memories.base import Turn
from memory_arena.memories.verbatim_rag import VerbatimRAG


@pytest.fixture
def mem():
    return VerbatimRAG()


def test_verbatim_rag_se_instancia():
    """VerbatimRAG debe instanciarse sin error."""
    m = VerbatimRAG()
    assert m is not None


def test_retrieve_vacio_sin_store(mem):
    """Sin ningún store previo, retrieve devuelve lista vacía."""
    assert mem.retrieve("cualquier query") == []


def test_store_no_falla_con_contenido_vario(mem):
    """store() no debe lanzar excepciones con distintos tipos de contenido."""
    mem.store(Turn(role="user", content="hola"))
    mem.store(Turn(role="assistant", content="que tal"))
    mem.store(Turn(role="user", content=""))  # contenido vacío


def test_retrieve_devuelve_lista_de_strings(mem):
    """retrieve() siempre devuelve una lista de strings."""
    mem.store(Turn(role="user", content="me gustan las pizzas"))
    result = mem.retrieve("comida favorita")
    assert isinstance(result, list)
    assert all(isinstance(r, str) for r in result)


def test_retrieve_respeta_top_k(mem):
    """retrieve() no devuelve más elementos que top_k."""
    for i in range(10):
        mem.store(Turn(role="user", content=f"turno número {i}"))
    assert len(mem.retrieve("algo", top_k=3)) <= 3
    assert len(mem.retrieve("algo", top_k=1)) <= 1


def test_retrieve_top_k_mayor_que_stored(mem):
    """Si top_k > cantidad de turnos guardados, devuelve todos los que hay."""
    mem.store(Turn(role="user", content="solo hay un turno"))
    result = mem.retrieve("query", top_k=10)
    assert len(result) == 1


def test_retrieve_semanticamente_relevante(mem):
    """El turno más similar semánticamente debe estar primero en el resultado.

    Usamos inglés porque all-MiniLM-L6-v2 está entrenado principalmente en
    inglés y la similitud semántica es más confiable en ese idioma.
    """
    mem.store(Turn(role="user", content="I love eating pizza with mozzarella"))
    mem.store(Turn(role="user", content="I live in Buenos Aires"))
    mem.store(Turn(role="user", content="I work as a software engineer"))

    result = mem.retrieve("what food does the user like?", top_k=1)

    assert len(result) == 1
    assert "pizza" in result[0].lower()


def test_reset_vacia_el_indice(mem):
    """Después de reset(), retrieve() vuelve a devolver lista vacía."""
    mem.store(Turn(role="user", content="dato importante"))
    mem.store(Turn(role="user", content="otro dato"))
    assert len(mem.retrieve("algo")) > 0

    mem.reset()

    assert mem.retrieve("algo") == []


def test_reset_permite_reusar_la_instancia(mem):
    """Después de reset(), se puede volver a usar store + retrieve normalmente."""
    mem.store(Turn(role="user", content="primera conversación"))
    mem.reset()

    mem.store(Turn(role="user", content="segunda conversación sobre gatos"))
    result = mem.retrieve("mascotas", top_k=1)

    assert len(result) == 1
    assert "gatos" in result[0].lower()


def test_store_preserva_texto_exacto(mem):
    """El texto devuelto por retrieve es exactamente el que se guardó con store."""
    texto = "Este es el texto exacto que guardamos."
    mem.store(Turn(role="user", content=texto))
    result = mem.retrieve("texto guardado", top_k=1)
    assert result[0] == texto


def test_multiples_stores_acumulan(mem):
    """Cada store agrega al índice sin reemplazar los anteriores."""
    mem.store(Turn(role="user", content="turno uno"))
    mem.store(Turn(role="user", content="turno dos"))
    mem.store(Turn(role="user", content="turno tres"))

    result = mem.retrieve("algo", top_k=5)
    assert len(result) == 3
