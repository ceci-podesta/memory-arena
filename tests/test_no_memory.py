"""
Tests para la estrategia NoMemoria (baseline).
"""

import pytest

from memory_arena.memories.base import MemoriaBase, Turn
from memory_arena.memories.no_memory import NoMemoria


def test_memoria_base_no_se_puede_instanciar():
    """MemoriaBase es abstracta: instanciarla debe fallar."""
    with pytest.raises(TypeError):
        MemoriaBase()


def test_no_memoria_se_instancia():
    """NoMemoria es concreta: debe instanciarse sin error."""
    mem = NoMemoria()
    assert mem is not None


def test_no_memoria_store_no_falla():
    """store() no debe lanzar excepciones, pase lo que pase."""
    mem = NoMemoria()
    mem.store(Turn(role="user", content="hola"))
    mem.store(Turn(role="assistant", content="que tal"))
    mem.store(Turn(role="user", content=""))  # contenido vacio


def test_no_memoria_retrieve_siempre_vacio():
    """retrieve() siempre devuelve lista vacia, sin importar query o top_k."""
    mem = NoMemoria()
    assert mem.retrieve("cualquier cosa") == []
    assert mem.retrieve("otra query", top_k=10) == []
    assert mem.retrieve("", top_k=100) == []


def test_no_memoria_retrieve_vacio_despues_de_store():
    """Aunque llamemos a store(), retrieve() sigue devolviendo []."""
    mem = NoMemoria()
    mem.store(Turn(role="user", content="dato importante"))
    mem.store(Turn(role="assistant", content="respuesta util"))
    assert mem.retrieve("dato importante") == []
