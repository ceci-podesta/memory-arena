# GraphMemory — estado actual (backup pre-refactor)

> Snapshot tomado el 2026-05-24 antes de refactorizar `graph_memory.py`.

---

## Estructura de datos

- `nx.MultiDiGraph`: grafo dirigido que permite múltiples aristas entre el mismo par de nodos.
- Cada arista guarda: `label` (predicado), `t` (turn index al momento de inserción), `valid_until` (turn index en que fue reemplazada, o `None` si sigue vigente).
- `_turn_index`: contador global que avanza con cada `store()`.

**¿Por qué dirigido?** Para triples `(sujeto, predicado, objeto)` la dirección importa — "Alice trabaja en Google" ≠ "Google trabaja en Alice". El retrieve compensa buscando tanto aristas salientes como entrantes desde cada seed, así que no se pierde contexto por la dirección.

---

## `store(turn)`

1. Extrae triples `(sujeto, predicado, objeto)` del texto via LLM.
2. Todo en minúsculas.
3. **Conflicto temporal (estilo Zep/Graphiti):** si ya existe una arista activa con el mismo predicado entre los mismos nodos, le pone `valid_until = turn_index` (la "invalida" sin borrarla) y agrega la nueva arista como vigente.

---

## `retrieve(query, top_k=15)`

1. Tokeniza la query y busca nodos que contengan alguna de esas palabras → **seed nodes**.
2. Fallback: si no hay seeds, toma los 10 nodos con mayor degree.
3. Expande **1-hop** (aristas salientes + entrantes) desde cada seed.
4. Ordena por `(word_match, timestamp)` descendente — primero más relevante, después más reciente.
5. Devuelve hasta `top_k` strings del tipo `"sujeto predicado objeto"` (o `"... (previously, now updated)"` si está invalidada).

---

## `_extract_triples(text)`

- Divide el texto en chunks de 16.000 chars con overlap de 1.500.
- Por cada chunk hace un `llm.chat()` con un prompt que pide JSON array de arrays.
- Parsea con regex: `["s","p","o"]`.
- Deduplica antes de retornar.

---

## Limitaciones / decisiones de diseño conocidas

- El retrieve **incluye triples invalidadas** (históricas) en el resultado, sólo les agrega el tag `(previously, now updated)`. No las filtra.
- El matching de seeds es **substring por palabra**, lo que puede ser ruidoso.
- **No hay embeddings**: toda la relevancia es léxica.
- El grafo **vive en memoria** — no persiste entre sesiones.
