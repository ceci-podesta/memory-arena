# GraphMemory — Estrategia basada en Grafo de Conocimiento

---

## Descripción General

GraphMemory representa la memoria de un agente como un **grafo de conocimiento dirigido**. Cada turno de conversación se convierte en triples `(sujeto, predicado, objeto)` que se almacenan como aristas. Al recuperar información, se navega el grafo partiendo de las entidades mencionadas en la query.

El resultado es una memoria que no solo recuerda hechos, sino las **relaciones entre entidades** y su **evolución temporal**.

---

## Fundamento Teórico

La idea central viene de los **knowledge graphs** usados en NLP: representar el conocimiento como una red de entidades y relaciones permite razonar sobre conexiones que no son explícitas en el texto original.

**LightRAG (Gu et al., 2024)**
Gu et al. (2024) proponen un sistema de retrieval que primero extrae entidades y relaciones del corpus usando un LLM, las almacena en un grafo de conocimiento, y luego navega ese grafo para responder queries. La ventaja frente a RAG clásico es que captura dependencias entre conceptos que en el texto original están separados por varios párrafos.
En nuestra implementación adoptamos directamente esta idea: el LLM extrae triples de cada turno y los insertamos como aristas. El retrieve navega el grafo en lugar de hacer búsqueda vectorial sobre chunks crudos.

**Zep/Graphiti (Rasmussen et al., 2025)**
Rasmussen et al. (2025) extienden la idea del knowledge graph al dominio conversacional, donde los hechos no son estáticos: una persona puede cambiar de trabajo, de ciudad, de opinión. Proponen marcar las aristas con timestamps y, cuando un hecho es contradicho, marcar la versión vieja como histórica en lugar de borrarla.
En nuestra implementación adoptamos este mecanismo íntegramente: cada arista tiene `t` (turno de inserción) y `valid_until` (turno en que fue superada). Esto permite que el LLM distinga el estado actual del histórico al leer el contexto recuperado.

**Microsoft GraphRAG (Edge et al., 2024)**
Edge et al. (2024) proponen construir un grafo de conocimiento sobre documentos largos y usar comunidades de nodos para responder preguntas globales que no pueden resolverse recuperando un solo chunk. La navegación del grafo permite sintetizar información dispersa en el documento.
En nuestra implementación aplicamos la misma intuición al patrón inject-once-query-many de MAB: el documento largo se convierte en grafo una sola vez, y cada query navega el subgrafo relevante sin releer el documento original.

**Mem0 (Chhikara et al., 2024)**
Chhikara et al. (2024) proponen una capa de memoria persistente para agentes LLM que combina búsqueda vectorial con un grafo de conocimiento. El grafo actúa como índice estructural que guía qué memorias vectoriales recuperar, combinando lo mejor de ambas representaciones.
En nuestra implementación optamos por el grafo solo, sin embeddings, dado el alcance del TP. Sin embargo, Mem0 señala la dirección natural de mejora: agregar embeddings de nodos para que el matching en retrieve sea semántico y no solo léxico.

---

## Implementación

### Store — Construcción del grafo

1. El texto del turno se divide en **chunks de 16.000 caracteres** (con overlap de 1.500) para respetar la ventana de contexto del LLM extractor.
2. Por cada chunk, el LLM extrae triples factuales en formato JSON:
   ```
   "Alice trabaja en Acme Corp" → ["alice", "trabaja en", "acme corp"]
   ```
3. Cada triple se inserta como arista en un `MultiDiGraph` de NetworkX con su timestamp `t`.
4. Si ya existe una arista activa con el mismo predicado entre los mismos nodos, se la marca como histórica (`valid_until = t_actual`) antes de insertar la nueva:
   ```
   alice --[trabaja en]--> old_company  (t=3, valid_until=7)   ← histórico
   alice --[trabaja en]--> acme corp    (t=7, valid_until=None) ← activo
   ```

### Retrieve — Expansión 1-hop

1. Se identifican **seed nodes**: nodos del grafo cuyo nombre aparece en alguna palabra del query.
2. Si no hay match, se usan los 5 nodos con mayor grado (los más conectados al resto del grafo).
3. Se recuperan todas las aristas de entrada y salida de cada seed (expansión 1-hop).
4. Los triples activos se devuelven como `"alice trabaja en acme corp"`. Los históricos se etiquetan: `"alice trabaja en old company (previously, now updated)"`.
5. Se ordenan por **relevancia léxica primero, recencia después**, y se devuelven los top-15.

---

## Comportamiento según Benchmark

### LongMemEval

Preguntas sobre sesiones largas donde los hechos evolucionan (cambios de trabajo, relaciones, preferencias). El marcado temporal de aristas permite al LLM distinguir el estado actual del histórico, atacando directamente las categorías **knowledge-update** y **temporal-reasoning**.

### MemoryAgentBench (MAB)

Patrón inject-once-query-many: un documento largo se inyecta una vez y se consulta N veces sin reiniciar la memoria. El chunking garantiza que el grafo se construya completo desde el primer `store`, antes de cualquier `retrieve`. Las N consultas navegan el mismo grafo sin costo adicional de LLM.

---

## Decisiones de Diseño

**Chunks de 16.000 caracteres en lugar de 32.000**
La ventana de contexto del modelo es de 16.384 tokens (input + output). Un chunk de 32.000 caracteres representa aproximadamente 8.000 tokens solo de input; sumado al prompt del extractor y a los tokens de respuesta solicitados, el total excede el límite. Ollama no lanza error: trunca el input silenciosamente. Con chunks de 16.000 caracteres (~4.000 tokens) el input entra cómodo y la extracción es más precisa.

**Marcar aristas como históricas (`valid_until`) en lugar de borrarlas**
Borrar el hecho viejo es la opción más simple, pero destruye la posibilidad de razonar sobre cambios temporales. Con `valid_until`, el retrieve puede entregar al LLM tanto el estado actual como el histórico etiquetado, atacando directamente las categorías *knowledge-update* y *temporal-reasoning* de LongMemEval.

**top_k = 15 en lugar de 5**
El grafo puede contener muchos triples relevantes para una query, especialmente en conversaciones largas. Con top_k = 5 se descartaba información útil. Subir a 15 no tiene costo adicional en velocidad ya que el retrieve es puramente local, sin llamadas al LLM.

---

## Limitaciones

- **Store es costoso**: cada turno requiere al menos una llamada al LLM extractor. Para textos muy largos (MAB), el número de chunks puede ser alto.
- **Matching léxico en retrieve**: los seed nodes se identifican por substring. Si la query usa sinónimos o paráfrasis, puede no encontrar el nodo correcto y caer al fallback de grado.
- **Expansión 1-hop**: cadenas de razonamiento largas (A→B→C→D) no se capturan si no hay match directo con A o D.
- **Calidad del extractor**: si el LLM extrae triples incorrectos o muy genéricos, el grafo acumula ruido que se propaga al retrieve.
- **Sin embeddings de nodos**: el matching es puramente léxico. Una mejora natural sería usar embeddings para encontrar nodos semánticamente similares a la query.

---

## Referencias

Gu, D., et al. (2024). LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779.

Edge, D., et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.

Rasmussen, P., et al. (2025). Graphiti: A Temporally-Aware Knowledge Graph for Agentic Applications. arXiv:2501.13956.

Chhikara, P., et al. (2024). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.

Wu, D., et al. (2024). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. arXiv:2410.10813.

Hu, W., et al. (2026). MemoryAgentBench: A Comprehensive Benchmark for Memory in Language Agents. arXiv:2507.05257.
