# memory-arena — Onboarding del equipo

Bienvenidos. Este documento es **para humanos**. Hay un complemento para pegar a tu
LLM en `docs/LLM_CONTEXT.md` con contexto técnico denso.

---

## 1. De qué va esto

**memory-arena** es un sistema comparativo de **5 estrategias de memoria para agentes
LLM** aplicadas a dos benchmarks de memoria de largo contexto: **LongMemEval** (Wu
et al. 2024) y **MemoryAgentBench** (Hu et al., ICLR 2026).

El objetivo del TP es **medir y comparar** el impacto de cada estrategia de memoria
frente a un baseline sin memoria, corriendo todo localmente con Ollama + modelos
chicos (Llama 3.2 3B como modelo evaluado, Mistral 7B como juez).

Entrega: **2026-05-26**.

Ya está hecho (v0.1 del pipeline):

- Infra completa: loader, runner, scoring default, juez LLM, scorer recsys.
- Baseline **NoMemoria** corrida sobre las 4 competencias de MAB + LongMemEval oracle
  (~4200 queries evaluadas), con métricas apropiadas para cada tipo de tarea.
- Documentación metodológica acumulada en `notas-informe-tp.md` (fuera del repo —
  es el documento vivo del informe).

Lo que falta: implementar las **4 estrategias con memoria** (una por persona) y
compararlas contra NoMemoria.

---

## 2. Setup paso a paso

### 2.1. Pre-requisitos del sistema

- **Linux o WSL2 con Ubuntu 22.04+**. Si tenés Windows, usá WSL (tutorial en el paso 2.2).
- **Python 3.10+** (instalado automáticamente por `uv` si no lo tenés).
- **Git + SSH key configurada con GitHub**.
- **16 GB RAM mínimo recomendado** (el modelo base Llama 3.2 3B pesa ~2 GB en RAM).

### 2.2. Instalar WSL (solo si estás en Windows)

En PowerShell como administrador:

```powershell
wsl --install -d Ubuntu-22.04
```

Reiniciá. Abrí Ubuntu desde el menú, creá tu usuario. Después:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl git
```

### 2.3. Clonar el repo y levantar el entorno

```bash
cd ~
mkdir -p projects && cd projects
git clone git@github.com:ceci-podesta/memory-arena.git
cd memory-arena
```

Instalar `uv` (manager de deps, más rápido que pip):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # o cerrar y abrir la terminal
```

Instalar deps del repo:

```bash
uv sync
```

Con eso `uv` crea `.venv/` y baja todas las deps (incluyendo `editdistance`,
`datasets`, `rouge-score`, `ollama`, etc.). Toda ejecución posterior va con
`uv run python ...`, nunca `python` pelado.

### 2.4. Instalar Ollama y bajar los modelos

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Arrancar Ollama como daemon (en segundo plano)
ollama serve &

# Bajar los dos modelos que usa el pipeline
ollama pull llama3.2:3b     # modelo evaluado (~2 GB)
ollama pull mistral:7b      # juez LLM para LRU (~4 GB)

# Verificar
ollama list
```

### 2.5. Construir el entity2id para recsys

El scorer de `recsys_redial_full` necesita un mapping de películas. Generalo una vez:

```bash
uv run python scripts/build_entity2id.py
```

Esperás ver algo tipo:

```
Generado: data/recsys_redial/entity2id.json
  Películas indexadas: 64228
  Fuente: data/recsys_redial/movies_merged.csv
  Rango de index: 0 a 64339
```

### 2.6. Smoke test: corré NoMemoria sobre un sub-dataset chico

```bash
uv run python scripts/run_d_cr_nomemoria.py
```

Si anda, vas a ver progreso de corridas y al final JSONL en `results/responses/`.
Si se rompe acá, cortá y avisá al canal del equipo — mejor debuggear antes que
seguir.

**Tip**: para smoke test rápido, en el script podés poner `MAX_SAMPLES = 1` antes
de correr y te hace solo una muestra en vez de todo.

---

## 3. Estructura del repo

```
memory-arena/
├── memory_arena/                 # El paquete Python principal
│   ├── benchmarks/
│   │   ├── longmemeval.py       # Loader de LongMemEval
│   │   └── memory_agent_bench.py  # Loader de MAB (4 competencias)
│   ├── memories/                # ← ACÁ VAN LAS ESTRATEGIAS
│   │   ├── base.py              # MemoriaBase (ABC), Turn dataclass
│   │   └── no_memory.py         # Baseline — referencia mínima para heredar
│   ├── llm/
│   │   └── ollama_client.py     # Wrapper de Ollama (chat API)
│   ├── evaluation/
│   │   ├── runner.py            # Fase A para LongMemEval
│   │   ├── mab_runner.py        # Fase A para MAB (inject-once-query-many)
│   │   ├── judgment_runner.py   # Fase B para LongMemEval (juez)
│   │   ├── mab_judgment_runner.py  # Fase B para MAB
│   │   ├── judge.py             # MistralJudge + MAB juezas
│   │   ├── mab_scoring.py       # Scoring default MAB (7 métricas + ramas)
│   │   ├── recsys_scorer.py     # Scoring recsys (Recall@K con fuzzy match)
│   │   └── run_metadata.py      # Trazabilidad: timestamps, hardware, commit
│   └── experimental_config.py   # Constantes compartidas (max_tokens, top_k, etc.)
├── scripts/                      # Puntos de entrada — uno por corrida/tarea
│   ├── build_entity2id.py       # Genera el mapping de recsys (se corre una vez)
│   ├── match_movies.py          # ReDial × MovieLens (solo si regenerás desde cero)
│   ├── run_d_ar_nomemoria.py    # Corre NoMemoria sobre Accurate_Retrieval
│   ├── run_d_cr_nomemoria.py    # ... Conflict_Resolution
│   ├── run_d_lru_nomemoria.py   # ... Long_Range_Understanding
│   ├── run_d_ttl_nomemoria.py   # ... Test_Time_Learning (5 ICL)
│   ├── run_d_ttl_recsys.py      # ... recsys_redial_full
│   ├── run_e_lru_judge.py       # Juez Mistral sobre JSONL de LRU
│   ├── score_d_*.py             # Scoring por competencia
│   └── score_e_lru.py           # Scoring con juez LLM
├── tests/                        # Unitests (pytest)
│   ├── test_memory_agent_bench.py
│   ├── test_mab_runner.py
│   ├── test_mab_scoring.py
│   └── test_config_consistency.py
├── data/                         # Datasets locales
│   ├── recsys_redial/
│   │   ├── movies_with_mentions.csv   # De ReDial original
│   │   └── movies_merged.csv          # ReDial × MovieLens (reindex global)
│   └── movielens/ml-25m/movies.csv    # Mapping de películas
├── results/                      # Outputs (ignorado en git)
│   ├── responses/               # JSONL por corrida (Fase A)
│   ├── judgments/               # JSONL post-juez (Fase B)
│   └── runs/                    # Metadata de cada corrida
├── docs/
│   ├── ONBOARDING.md            # Este archivo
│   └── LLM_CONTEXT.md           # Contexto para pegar a tu LLM
└── pyproject.toml               # Deps + config
```

---

## 4. Cómo funciona el pipeline

**Dos fases separadas** para poder re-usar respuestas con distintos jueces:

**Fase A — Generación** (costosa):
1. Cargás los samples del benchmark (`load_longmemeval` o `load_mab`).
2. Por cada sample: `memoria.reset()` → `memoria.store(turn)` por cada turno o
   documento → `memoria.retrieve(question)` para obtener contexto relevante.
3. Pasás el contexto al LLM evaluado (Llama 3.2 3B) que genera la respuesta.
4. Escribís todo a un JSONL en `results/responses/<run_id>.jsonl`.
5. Metadata de la corrida va a `results/runs/<run_id>.json` (timestamps, hardware,
   git commit, duración).

**Fase B — Scoring** (barato si es default, medio si es juez LLM):
1. Leés el JSONL de Fase A.
2. Scoreás cada respuesta según el sub-dataset:
   - **AR / CR / TTL-icl**: métricas léxicas (EM, F1, ROUGE).
   - **LRU**: juez LLM con Mistral (answer-matching para detective_qa,
     summarization-style para infbench_sum).
   - **TTL-recsys**: Recall@K con fuzzy matching por edit distance.
3. Agregás resultados y escribís CSV en `results/runs/` y a stdout.

**Inject-once-query-many (MAB)**: en MAB un solo "sample" tiene un contexto largo
+ N preguntas sobre ese contexto. El runner inyecta el contexto UNA VEZ (con
`store`), y después hace N retrieves sin resetear. Este patrón es lo que tu
estrategia tiene que soportar.

---

## 5. La interfaz que tenés que implementar

**Todas las estrategias heredan de `MemoriaBase`** (en `memory_arena/memories/base.py`):

```python
class MemoriaBase(ABC):
    @abstractmethod
    def store(self, turn: Turn) -> None:
        """Guarda un turno o documento en la memoria."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Devuelve top_k fragmentos relevantes para la query."""

    @abstractmethod
    def reset(self) -> None:
        """Limpia la memoria (se llama entre samples)."""
```

El dataclass `Turn` tiene: `role` (user / assistant / document), `content` (string),
`session_id` (trazabilidad), `date` (opcional, solo LongMemEval).

**`NoMemoria`** (`memory_arena/memories/no_memory.py`) es la implementación mínima
de referencia: `store()` no guarda nada, `retrieve()` devuelve `[]`, `reset()` es
no-op. Miralo antes de arrancar — 20 líneas, se entiende en 2 minutos.

Tu estrategia hace lo mismo pero con lógica adentro. **No tocás el runner**, solo
implementás estas 3 funciones.

---

## 6. Asignación por persona

Cada uno es **especialista** de su estrategia. Te preparás a fondo, corrés la
evaluación, y después tenemos una charla de equipo donde cada uno cuenta la suya.

> **Filosofía de trabajo**: desarrollen y testeen local con dataset chico
> (`max_samples=5` o `=10`). Pushean cuando pase el smoke test + los unitests.
> Las corridas pesadas reales del benchmark completo las coordinamos entre los
> 4 según disponibilidad de compute (Gonza y Ceci tienen máquinas más potentes;
> pueden correr las que sean más costosas).

### 6.1. Sol — Estrategia 2: Verbatim + RAG

**Qué hace**: guardar cada turno (o cada documento) verbatim en una memoria
vectorial. Cuando llega una query, sacar los top-K chunks más similares por
embedding cosine y devolverlos como contexto.

**Qué construís**:
- `memory_arena/memories/verbatim_rag.py` — clase `VerbatimRAG(MemoriaBase)`.
- `store(turn)` — computa embedding del `turn.content` y lo agrega a un índice
  (podés usar FAISS in-memory, Chroma, o un wrapper simple con NumPy cosine).
- `retrieve(query, top_k)` — embedea la query, cosine contra el índice, devuelve
  los top_k chunks como lista de strings.
- `reset()` — vaciá el índice.

**Stack sugerido**:
- `sentence-transformers` (`all-MiniLM-L6-v2` si querés velocidad, `mpnet-base-v2`
  si querés calidad). Agregalo con `uv add sentence-transformers`.
- Índice: NumPy simple al principio (cosine manual). FAISS si querés escalar.

**Papers de apoyo**:
- Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (paper original de RAG).
- Reimers & Gurevych 2019, *Sentence-BERT* (para entender por qué sentence-transformers).
- Lleves algún paper de BM25 vs dense si querés hacer baseline comparison interno.

**Consideraciones de compute**: sentence-transformers corren bien en CPU si la
compu es modesta (tardan un poco más pero no frenan). Modelos más chicos =
más rápido.

---

### 6.2. David — Estrategia 3: Summarized

**Qué hace**: mantener un **resumen rolling** de la conversación / documento. Cada
cierto número de turnos (o cuando se superan K tokens), el LLM re-resume. Cuando
llega la query, el retrieve devuelve el resumen actual (y opcionalmente los
últimos N turnos crudos).

**Qué construís**:
- `memory_arena/memories/summarized.py` — clase `SummarizedMemory(MemoriaBase)`.
- Estado interno: `self.summary` (string), `self.recent_turns` (list[Turn] con
  los últimos N sin resumir todavía), `self.turn_count`.
- `store(turn)` — agregás al buffer de recientes. Si el buffer supera N turnos,
  llamás al LLM para generar un nuevo `summary` que integre el anterior + los
  recientes, y vaciás el buffer.
- `retrieve(query, top_k)` — devolvés `[self.summary] + últimos M turnos`. No hay
  retrieval semántico, es "siempre devolver el summary".
- `reset()` — vaciá summary + buffer.

**Stack**: podés reutilizar el `OllamaClient` para generar los resúmenes con
Llama 3.2 3B. Prompt simple de "resumí este contexto".

**Papers de apoyo**:
- Zhong et al. 2024, *MemoryBank: Enhancing LLMs with Long-Term Memory* (decay + summarization).
- Chen et al., *Summarize Before You Forget* (HippoRAG o similar).
- ChatGPT Memory (OpenAI blog, 2024): producto comercial más cercano a esta estrategia.

**Consideraciones de compute**: el costo está en las llamadas al LLM para
resumir. En smoke test con `max_samples=5` es despreciable. En corridas reales
puede agregar latencia — estimá cuántas veces vas a re-resumir por sample.

---

### 6.3. Ceci — Estrategia 4: A-MEM / Agentic

**Qué hace**: un **agente** (loop de LLM con tool calling) que decide qué guardar
y cuándo actualizar la memoria. En store, el agente analiza el turno y puede
elegir: "esto es importante, lo agrego como fact", "esto modifica un fact
existente, lo actualizo", "esto no aporta, lo ignoro". En retrieve, hace varias
queries internas al índice y razona qué devolver.

**Qué construís**:
- `memory_arena/memories/a_mem.py` — clase `AgenticMemory(MemoriaBase)`.
- Estado: un store de memory notes (cada una con metadata: tags, importancia,
  links a otras memorias).
- `store(turn)` — invocás al LLM con herramientas (`add_note`, `update_note`,
  `link_notes`, `skip`). Multi-turn hasta que el agente decida que terminó.
- `retrieve(query, top_k)` — el agente puede hacer búsquedas en el store,
  componer fragmentos, etc.
- `reset()` — vaciás el store.

**Stack**:
- `OllamaClient` con Llama 3.2 3B (para el agente) o mistral:7b si necesitás
  más capacidad de reasoning — trade-off de velocidad.
- Podés implementar el tool calling manual (parsear JSON output del modelo) o
  usar una librería tipo `instructor` / `pydantic-ai` si querés menos
  plumbing.

**Papers de apoyo**:
- Zhong et al. 2025, *A-MEM: Agentic Memory for LLM Agents* (paper homónimo).
- Packer et al. 2023, *MemGPT: Towards LLMs as Operating Systems* (inspiración — OS de memoria virtual para LLMs).
- Proyecto **Letta** (ex-MemGPT): https://github.com/letta-ai/letta — referencia de producción.

**Consideraciones de compute**: este es el más costoso por sample (el agente
hace múltiples llamadas al LLM por cada store). Testeá muy chico al principio
(`max_samples=1`). Para las corridas reales del benchmark completo probablemente
Gonza corre en su Mac.

---

### 6.4. Gonza — Estrategia 5: Graph-based

**Qué hace**: extraer entidades y relaciones de cada turno con el LLM, construir
un **knowledge graph** (nodos = entidades, edges = relaciones con peso/tipo),
y en retrieve hacer **graph traversal** + embeddings sobre los nodos para
devolver los subgrafos relevantes.

**Qué construís**:
- `memory_arena/memories/graph_memory.py` — clase `GraphMemory(MemoriaBase)`.
- Estado: un grafo (`networkx.Graph` o similar) con nodos (entidades) y edges
  (relaciones). Cada nodo puede tener un embedding.
- `store(turn)` — llamás al LLM para extraer triples `(subject, predicate, object)`
  del turno. Agregás nodos/edges al grafo. Si un nodo ya existe, mergeás info.
- `retrieve(query, top_k)` — identificás entidades en la query (via LLM o NER),
  buscás nodos cercanos en el grafo, expandís k-hop, devolvés como contexto
  narrado.
- `reset()` — vaciás el grafo.

**Stack**:
- `networkx` para el grafo (stdlib más o menos).
- `OllamaClient` para extracción de triples (prompt con few-shot).
- Opcionalmente `sentence-transformers` para embeddings de nodos (re-uso con Sol).

**Papers de apoyo**:
- Rasmussen et al. 2025, *Zep: A Temporal Knowledge Graph Architecture for Agent Memory* (arxiv 2501.13956).
- Chhikara et al. 2024, *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*.
- Gu et al. 2024, *LightRAG: Simple and Fast Retrieval-Augmented Generation* (graph-based RAG).
- Edge et al. 2024, *From Local to Global: A Graph RAG Approach* (Microsoft GraphRAG).

**Consideraciones de compute**: construir el grafo es O(N_turnos × coste_LLM_per_extract).
Para LongMemEval con 40 sesiones puede tardar. Tu Mac banca, aprovechala.

---

### 6.5. Roles adicionales del equipo

Además de la estrategia técnica de cada uno:

- **Sol — responsable del armado final del informe**. Sol viene de la academia
  (física, master en España) y tiene mano fina para documentos oficiales.
  Cada uno le pasa sus findings, tablas, y análisis en bruto; ella los integra,
  unifica el tono, y arma el entregable final.
- **David — coordinador de la PPT**. Cada uno arma sus propios slides de su
  estrategia (el storytelling es más fluido cuando lo hace quien implementó).
  David coordina el arco narrativo global, el orden de los slides, la
  consistencia visual, y la sección de conclusiones.
- **Gonza — corridas pesadas + arquitectura más compleja**. Ya tiene un rol
  computacional grande (su Mac termina corriendo varias evaluaciones reales
  además de las suyas). Si algo pesado no entra en una compu, va a la de
  Gonza.
- **Ceci — estrategia Agentic + point of contact técnico del repo**. Si te
  trabás con la arquitectura, los loaders, el scoring, o cualquier detalle del
  pipeline, consultala. Tiene contexto completo del repo.

> *En realidad yo (Claude) lo tengo, pero preguntenle a Ceci y ella me pregunta.* 😄

---

## 7. Workflow de trabajo

### 7.1. Todos corren NoMemoria primero (side-experiment)

Antes de arrancar con tu estrategia, corré NoMemoria en tu compu. Razones:

1. Confirma que tu entorno funciona end-to-end.
2. Genera un baseline propio — vamos a tener **4 baselines independientes** (uno
   por compu). Si los números divergen entre nosotros con el mismo seed, tenemos
   evidencia empírica para la sección 3 del informe sobre
   reproducibilidad inter-máquina.

```bash
# Un smoke test chico (1 sample por sub-dataset):
# editá el script y poné MAX_SAMPLES = 1 temporalmente, después corré:

uv run python scripts/run_d_cr_nomemoria.py
uv run python scripts/run_d_ar_nomemoria.py
uv run python scripts/run_d_lru_nomemoria.py
uv run python scripts/run_d_ttl_nomemoria.py
uv run python scripts/run_d_ttl_recsys.py

# Luego scoreás:
uv run python scripts/score_d_cr.py
# ... etc.
```

Pegá los números al canal del equipo para comparar.

### 7.2. Arrancá tu estrategia

**Orden sugerido**:

1. **Leé `memory_arena/memories/no_memory.py`** y **`base.py`**. Entendé qué
   espera el runner de tu clase.
2. **Creá tu archivo** `memory_arena/memories/<tu_estrategia>.py` — copia la
   estructura de `no_memory.py` y empezá mínimo.
3. **Implementá incrementalmente**: primero `store` simple, después `retrieve`
   simple, recién después vas agregando lógica.
4. **Smoke test manual** desde Python:
   ```python
   from memory_arena.memories.tu_estrategia import TuMemoria
   from memory_arena.memories.base import Turn
   m = TuMemoria()
   m.store(Turn(role="user", content="me gustan las pizzas", session_id="test"))
   print(m.retrieve("qué comida me gusta?", top_k=3))
   ```
5. **Escribí unitests** en `tests/test_<tu_estrategia>.py`. Mirá `test_memory_agent_bench.py` como referencia de estilo.
6. **Corré tu estrategia sobre 1 sub-dataset con `max_samples=5`**. Si anda, empujás.
7. **Después** coordinás las corridas reales full-benchmark con Gonza/Ceci.

### 7.3. Reglas de commit

- Una branch por estrategia: `feature/verbatim-rag`, `feature/summarized`, etc.
- Commits chicos y mensajes explícitos.
- PR a main cuando tu smoke test + unitests pasen.
- Todos reviewan el PR del otro (aunque sea rápido) — así todos vemos las 4 estrategias antes de la charla final.

### 7.4. Tests

Usamos `pytest`:

```bash
uv run pytest tests/ -v
```

Antes de cualquier push, los tests existentes tienen que seguir pasando. Si
rompiste algo, fijalo antes de empujar.

---

## 8. Los benchmarks

Todos tenemos que manejar ambos con soltura porque vamos a comparar resultados.

### 8.1. LongMemEval (Wu et al. 2024)

**Paper**: *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*
(arxiv 2410.10813).

**Qué evalúa**: cómo un agente recuerda información específica de conversaciones
largas multi-sesión. El dataset tiene 500 preguntas; cada una viene con un
historial de conversaciones (de 40 a 500 sesiones según variante) donde en
algún lado está la respuesta.

**Métricas**: yes/no con juez LLM (nosotros usamos Mistral 7B local, el paper usa
GPT-4o). Se rankea por accuracy global + por tipo de pregunta (single-session,
multi-session, temporal-reasoning, knowledge-update, preference).

**Variantes del dataset**:
- `longmemeval_oracle`: solo la sesión relevante (fácil, piso alto).
- `longmemeval_s`: 40 sesiones, cleaned (el que vamos a usar para comparaciones reales).
- `longmemeval_s_star`: 500 sesiones (muy difícil, solo si queda tiempo).

**Nuestro status**: ya corrimos NoMemoria sobre `longmemeval_oracle` (500 samples,
accuracy 0.202 con juez Mistral). Pendiente: las 4 estrategias + eventualmente
`longmemeval_s`.

### 8.2. MemoryAgentBench / MAB (Hu et al., ICLR 2026)

**Paper**: *MemoryAgentBench* (arxiv 2507.05257).

**Qué evalúa**: memoria de agentes a través de **4 competencias distintas**, cada
una con sub-datasets específicos:

| Competencia | Sub-datasets | Qué testea |
|---|---|---|
| **AR** (Accurate Retrieval) | longmemeval_s*, eventqa_*, ruler_qa1/2 | recuperar info puntual de un haystack largo |
| **TTL** (Test-Time Learning) | icl_banking77/clinic150/nlu/trec_*, recsys_redial_full | aprender del contexto (ICL) |
| **LRU** (Long-Range Understanding) | detective_qa, infbench_sum | entender/resumir texto largo |
| **CR** (Conflict Resolution) | factconsolidation_sh/mh_* | resolver contradicciones del contexto |

**Métricas**: varían por sub-dataset.
- Default: EM, substring_EM, F1, ROUGE-L (7 métricas del paper).
- LRU: juez LLM (Mistral 7B local; el paper usa GPT-4o).
- recsys: Recall@1/5/10 con fuzzy matching + entity2id.

**Patrón particular**: "inject-once, query-many" — un contexto largo, N preguntas
sobre ese contexto. El agente ingesta una vez y después responde muchas queries
sin resetear.

**Nuestro status**: NoMemoria corrido sobre las 4 competencias + recsys — baseline
completa (~3800 queries MAB + 500 LongMemEval oracle = ~4300 queries). Hallazgos
interesantes documentados en `notas-informe-tp.md` (contaminación paramétrica en
RULER, MCQA chance baseline en detective_qa, gap del paper sobre entity2id para
recsys).

---

## 9. Milestones y acuerdos del equipo

### Deadlines

- **2026-05-26**: entrega final del TP (informe + PPT + repo).

### Milestones sugeridos

Hoy es **jueves 2026-04-23**. Entrega final: **martes 2026-05-26**.

- **Semana 1 (esta → cierra domingo 2026-04-26)**: todos corren NoMemoria local.
  Cada uno lee su estrategia y papers, esboza el skeleton de su clase.
- **Semana 2 (cierra domingo 2026-05-03)**: todos tienen smoke test pasando en
  `max_samples=5`.
- **Semana 3 (cierra domingo 2026-05-10)**: corridas reales sobre LongMemEval
  oracle + MAB. Revisiones cruzadas.
- **Semana 4 (cierra domingo 2026-05-17)**: análisis comparativo, cierre del
  informe (Sol lidera la integración), armado de la PPT (David coordina).
- **Última semana (cierra domingo 2026-05-24)**: **charla interna de equipo**
  donde cada uno cuenta su estrategia a los demás (30 min cada uno, con demo).
  Después ensayamos la presentación al profe. Ajustes finales.
- **Martes 2026-05-26**: entrega.

### Canal de comunicación

Todo lo técnico al grupo de WhatsApp. Los bugs raros / decisiones metodológicas
se capturan en `notas-informe-tp.md` (documento vivo del informe, subido al
**Drive compartido del equipo** por Ceci — pidan el link si no lo tienen).

Sobre ese documento: es el **cajón de notas sin curar** donde vamos acumulando
todo lo que vamos aprendiendo (decisiones de diseño, hallazgos empíricos,
limitaciones, caveats). Está intencionalmente en formato "borrador denso", no
informe pulido. La idea es que al final pasemos ese material por un LLM para
destilar + reorganizar en el informe oficial (Sol lidera ese paso).

> **Por qué no curarlo ahora**: cuando uno está en el medio del trabajo, lo
> importante es capturar el pensamiento crudo — incluyendo los intentos fallidos
> y las hipótesis que después cambiaron. Si tratamos de que sea prolijo desde el
> principio, perdemos matices que son valiosos para el análisis posterior.
> Ahora juntamos evidencia sin filtrar; al final curamos con cabeza fría.

### Cuando te trabás

1. Releé este `ONBOARDING.md` — muchas preguntas están acá.
2. Mirá `docs/LLM_CONTEXT.md` y pegá el contenido como system prompt de tu
   LLM preferido. Pedile ayuda sobre tu estrategia específica.
3. Revisá `no_memory.py` — es la referencia mínima más fiel.
4. Si todo lo anterior no alcanza, ping a Ceci por WhatsApp con el error
   concreto y un snippet.

---

## 10. Links útiles

- **Repo**: https://github.com/ceci-podesta/memory-arena
- **LongMemEval paper**: https://arxiv.org/abs/2410.10813
- **MemoryAgentBench paper**: https://arxiv.org/abs/2507.05257
- **Ollama**: https://ollama.com/
- **uv**: https://docs.astral.sh/uv/

---

*Documento vivo — avisen si algo no se entiende o encuentran errores. Se
actualiza a medida que aprendemos cosas.*
