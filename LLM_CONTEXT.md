# memory-arena — Contexto para LLM

**Cómo usar este documento**: copialo completo y pegalo como primer mensaje (o
system prompt) cuando arranques una sesión con tu LLM favorito (ChatGPT /
Claude / etc.). Después presentate ("soy Sol, trabajo en verbatim+RAG") y el
LLM ya tiene todo el contexto para ayudarte sin volver a preguntar qué es esto.

> **Al LLM que lee esto**: el usuario está trabajando en un TP universitario
> sobre estrategias de memoria para agentes LLM. Es un proyecto Python con
> Ollama local. Este documento es la fuente de verdad sobre la estructura y
> convenciones del repo — respetalas estrictamente. Preferí preguntar antes
> que inventar nombres de archivos, firmas o comportamientos que no estén
> documentados acá. Si el usuario te pide algo que contradice este documento,
> señalalo antes de avanzar.

---

## 1. Identidad del proyecto

- **Repo**: `memory-arena` (https://github.com/ceci-podesta/memory-arena)
- **Qué es**: sistema comparativo de 5 estrategias de memoria para agentes LLM
  evaluadas contra **LongMemEval** (Wu et al. 2024, arxiv 2410.10813) y
  **MemoryAgentBench/MAB** (Hu et al., ICLR 2026, arxiv 2507.05257).
- **Modelos locales** (via Ollama): `llama3.2:3b` como modelo evaluado,
  `mistral:7b` como juez LLM.
- **Deadline del TP**: martes 2026-05-26.
- **Equipo** (4 personas): Sol (verbatim+RAG + informe final), David
  (summarized + coordinación PPT), Ceci (agentic + point of contact técnico),
  Gonza (graph-based + corridas pesadas en Mac).

---

## 2. Mapa del codebase

```
memory-arena/
├── memory_arena/                 # Paquete principal
│   ├── benchmarks/
│   │   ├── longmemeval.py
│   │   └── memory_agent_bench.py
│   ├── memories/                 # ACÁ viven las 5 estrategias
│   │   ├── base.py               # MemoriaBase (ABC), Turn (dataclass)
│   │   └── no_memory.py          # Baseline — referencia para heredar
│   ├── llm/
│   │   └── ollama_client.py      # Wrapper de Ollama
│   ├── evaluation/
│   │   ├── runner.py             # Fase A LongMemEval
│   │   ├── mab_runner.py         # Fase A MAB (inject-once-query-many)
│   │   ├── judgment_runner.py    # Fase B LongMemEval
│   │   ├── mab_judgment_runner.py # Fase B MAB
│   │   ├── judge.py              # MistralJudge + MAB jueces
│   │   ├── mab_scoring.py        # Scoring default MAB + dispatcher
│   │   ├── recsys_scorer.py      # Scoring recsys (Recall@K + fuzzy)
│   │   └── run_metadata.py       # Metadata: timestamps, hardware, commit
│   └── experimental_config.py    # Constantes compartidas
├── scripts/                      # Puntos de entrada (uno por corrida/tarea)
├── tests/
├── data/
│   ├── recsys_redial/            # movies_with_mentions, movies_merged
│   └── movielens/ml-25m/         # movies.csv (solo ese se commitea)
├── results/                      # Outputs LOCALES (gitignored)
│   ├── responses/                # Fase A
│   ├── judgments/                # Fase B
│   └── runs/                     # Metadata
├── docs/
│   ├── ONBOARDING.md             # Para humanos
│   └── LLM_CONTEXT.md            # Este archivo
└── pyproject.toml
```

---

## 3. Abstracciones clave (firmas exactas)

### 3.1. `Turn` (dataclass)

```python
# memory_arena/memories/base.py
@dataclass
class Turn:
    role: str                       # "user" | "assistant" | "document"
    content: str
    session_id: str | None = None
    date: str | None = None          # solo usado en LongMemEval
```

### 3.2. `MemoriaBase` (ABC)

```python
# memory_arena/memories/base.py
class MemoriaBase(ABC):
    @abstractmethod
    def store(self, turn: Turn) -> None:
        """Guarda un turno o documento en la memoria."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Devuelve hasta top_k fragmentos de contexto relevantes para la query.
        Cada fragmento es un string. El caller los concatena con separador '---'."""

    @abstractmethod
    def reset(self) -> None:
        """Limpia la memoria (se llama entre samples)."""
```

**Reglas importantes**:
- `store` puede llamarse múltiples veces antes de un `retrieve`.
- `retrieve` puede llamarse múltiples veces sin `reset` (patrón inject-once-query-many).
- `reset` se llama entre samples (el runner lo llama automáticamente).
- Si no hay contexto para devolver, `retrieve` devuelve `[]`.

### 3.3. `OllamaClient`

```python
# memory_arena/llm/ollama_client.py
class OllamaClient:
    def __init__(self, model: str = "llama3.2:3b", ...): ...

    def chat(
        self,
        messages: list[dict],         # formato ChatML: [{"role": "...", "content": "..."}]
        max_tokens: int | None = None
    ) -> str:
        """Devuelve el texto de la respuesta (no un objeto estructurado)."""
```

- **NO usar `.generate()`** — no existe. Solo `.chat()`.
- Parámetros fijos del pipeline: `temperature=0`, `seed=42`, `num_ctx=16384`.
  Están encapsulados en el cliente; no se tocan por estrategia.

### 3.4. `MABSample` (del loader de MAB)

```python
# memory_arena/benchmarks/memory_agent_bench.py
@dataclass(frozen=True)
class MABSample:
    sample_id: str
    source: str                     # "detective_qa", "ruler_qa1_197K", etc.
    context: str                    # texto largo a inyectar una vez
    questions: list[str]            # N preguntas sobre el context
    answers: list[list[str]]        # lista paralela; cada elem es lista de golds
    question_ids: list[str]         # lista paralela de ids
```

### 3.5. `run_metadata` (Fase A y Fase B)

```python
# memory_arena/evaluation/run_metadata.py
start_run(strategy: str, benchmark: str, model: str, num_samples: int) -> RunMetadata
finalize_run(metadata: RunMetadata, output_path: Path) -> None
```

Captura automáticamente: timestamps UTC, duración, hardware (CPU, GPU via
nvidia-smi), git commit SHA. Se escribe a `results/runs/<run_id>.json`.

---

## 4. Convenciones e invariantes del repo

### 4.1. Ejecución

- **Siempre `uv run python ...`**. Nunca `python` pelado — el venv no está activado.
- **Nunca llamar a `pip install` directo**. Deps nuevas: `uv add <paquete>`.
- **Scripts entran por `scripts/`**, uno por corrida/tarea. No por `-m`.

### 4.2. Reproducibilidad

- **`temperature=0`, `seed=42`, `top_p=1.0`** para todas las llamadas al LLM.
- **`num_ctx=16384`** (ventana de contexto de Ollama). Subir si la estrategia
  lo necesita; documentar en el informe.
- Cada corrida emite metadata en `results/runs/` — no modificar ese patrón.

### 4.3. Paths

- Los scripts añaden `REPO_ROOT` al `sys.path` al inicio para imports absolutos.
- Los CSVs de `data/` se acceden via `Path(__file__).parent.parent / "data" / ...`.
- **Nunca hardcodear paths absolutos** tipo `/home/cecilia/...` — usar `Path`.

### 4.4. Gitignore

- `results/` entero: ignorado (outputs locales por dev).
- `data/movielens/ml-25m/movies.csv`: **se commitea** (es el único del dataset).
- `data/movielens/ml-25m/ratings.csv`, `genome-*`, etc.: ignorados.
- `data/recsys_redial/entity2id.json`: ignorado (regenerable con script builder).
- `.venv/`, `__pycache__/`, `*.pyc`: ignorados.

### 4.5. Estilo de código

- **Type hints estilo PEP 604**: `str | None` en vez de `Optional[str]`.
- **Docstrings en castellano** (el equipo trabaja en castellano; los prompts
  copiados del paper quedan en inglés textual con atribución).
- **Usar `pathlib.Path`** para archivos. Nunca `os.path.join`.
- **Imports absolutos** desde `memory_arena.*`.

---

## 5. Modelo mental del pipeline

### 5.1. Dos fases

**Fase A (generación, costosa)**:
```
samples → [store → retrieve → LLM] × N_questions → JSONL en results/responses/
```
La estrategia de memoria vive acá. El LLM genera las respuestas.

**Fase B (scoring, barato/medio)**:
```
JSONL responses → dispatcher por sub_dataset → [default scorer | juez LLM | recsys scorer]
              → JSONL judgments / CSV scores en results/
```
La estrategia NO participa. Son métricas post-hoc.

### 5.2. Inject-once-query-many (MAB específicamente)

En MAB cada "sample" = 1 contexto largo + N preguntas. El runner hace:

```
for sample in samples:
    memoria.reset()
    memoria.store(Turn(role="document", content=sample.context, ...))  # UNA vez
    for question in sample.questions:                                   # N veces
        context = memoria.retrieve(question, top_k=5)
        response = llm.chat(build_prompt(context, question))
        write_jsonl(response)
```

Tu estrategia **no se reinicia entre preguntas del mismo sample**. Tiene que
soportar N retrieves consecutivos con el mismo estado de memoria.

### 5.3. Scoring por sub-dataset (MAB)

El dispatcher `score_response(answer, gold, sub_dataset)` enruta:

| Sub-dataset contiene | Scorer | Métricas |
|---|---|---|
| `eventqa_*` | default + eventqa_recall | 7 default + eventqa_recall |
| `icl_*` | parse_output + default | 7 default sobre parsed |
| `recsys_*` | `recsys_scorer.score_recsys_response` | Recall@1/5/10 + n_gold + n_predicted |
| (resto) | default_branch | 7 default (EM, substring_EM, F1, ROUGE-L × 4) |

Para LRU (`detective_qa`, `infbench_sum_eng_shots2`) usamos juez LLM aparte,
no el default. Corre en Fase B via `mab_judgment_runner`.

---

## 6. Guía por estrategia

Si el usuario se presenta como una de estas personas, ya sabés qué está haciendo:

### 6.1. Sol → `VerbatimRAG` (estrategia 2)

- **Archivo a crear**: `memory_arena/memories/verbatim_rag.py`.
- **Clase**: `VerbatimRAG(MemoriaBase)`.
- **Idea**: embedding de cada `turn.content` al `store`; cosine similarity contra todos en `retrieve`; devuelve top_k.
- **Stack sugerido**: `sentence-transformers` (modelo `all-MiniLM-L6-v2` para velocidad), NumPy cosine manual. Posible FAISS si quiere escalar.
- **Agregar dep**: `uv add sentence-transformers`.
- **Papers**: Lewis et al. 2020 (RAG original), Reimers & Gurevych 2019 (SBERT).
- **Compute**: barato. Embeddings corren bien en CPU.

### 6.2. David → `SummarizedMemory` (estrategia 3)

- **Archivo**: `memory_arena/memories/summarized.py`.
- **Clase**: `SummarizedMemory(MemoriaBase)`.
- **Idea**: mantener `self.summary` (string) + buffer de turnos recientes. Cuando el buffer supera N turnos, invocar LLM para re-resumir incorporando summary anterior + buffer. `retrieve` devuelve `[summary, ultimos_M_turnos]`. No hay retrieval semántico.
- **Stack**: reutiliza `OllamaClient` para generar resúmenes. Prompt simple "resumí este contexto conservando hechos clave".
- **Papers**: Zhong et al. (MemoryBank), ChatGPT Memory (OpenAI).
- **Compute**: medio. Costo en llamadas al LLM para resumir.

### 6.3. Ceci → `AgenticMemory` (estrategia 4)

- **Archivo**: `memory_arena/memories/a_mem.py`.
- **Clase**: `AgenticMemory(MemoriaBase)`.
- **Idea**: agente con tool calling (`add_note`, `update_note`, `link_notes`, `skip`) decide cómo almacenar. En retrieve, puede hacer múltiples búsquedas internas y componer contexto.
- **Stack**: `OllamaClient` con `llama3.2:3b` o `mistral:7b`. Tool calling manual parseando JSON output (o usar `instructor`/`pydantic-ai`).
- **Papers**: Zhong et al. 2025 (A-MEM), Packer et al. 2023 (MemGPT), proyecto Letta.
- **Compute**: alto. Múltiples LLM calls por sample.

### 6.4. Gonza → `GraphMemory` (estrategia 5)

- **Archivo**: `memory_arena/memories/graph_memory.py`.
- **Clase**: `GraphMemory(MemoriaBase)`.
- **Idea**: LLM extrae triples `(subject, predicate, object)` de cada turno. `networkx` para el grafo. En retrieve, identificar entidades en la query, expansión k-hop, devolver subgrafo narrado.
- **Stack**: `networkx` (`uv add networkx`), `OllamaClient` para extracción, opcionalmente `sentence-transformers` para embeddings de nodos.
- **Papers**: Rasmussen et al. 2025 (Zep), Chhikara et al. 2024 (Mem0), Gu et al. 2024 (LightRAG), Edge et al. 2024 (Microsoft GraphRAG).
- **Compute**: alto en construcción del grafo. Gonza tiene Mac potente.

---

## 7. Tareas comunes

### 7.1. Cómo heredar de NoMemoria

Patrón mínimo:

```python
# memory_arena/memories/tu_estrategia.py
from memory_arena.memories.base import MemoriaBase, Turn

class TuMemoria(MemoriaBase):
    def __init__(self, ...):
        # init state
        pass

    def store(self, turn: Turn) -> None:
        # guardá algo
        pass

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        # devolvé lista de strings (los fragmentos relevantes)
        return []

    def reset(self) -> None:
        # limpiá estado
        pass
```

### 7.2. Smoke test manual

```python
from memory_arena.memories.tu_estrategia import TuMemoria
from memory_arena.memories.base import Turn

m = TuMemoria()
m.store(Turn(role="user", content="me encantan las pizzas con ananá", session_id="s1"))
m.store(Turn(role="user", content="odio los jueves", session_id="s1"))
print(m.retrieve("qué comida le gusta al usuario?", top_k=2))
m.reset()
assert m.retrieve("cualquier cosa", top_k=1) == []
```

### 7.3. Correr sobre el benchmark

No hace falta modificar el runner. Hay que escribir un script nuevo en
`scripts/` que importe la estrategia y llame a `run_strategy_mab` o `run_strategy_longmemeval`. Usar los `scripts/run_d_*` existentes como template.

Ejemplo (pseudo):
```python
from memory_arena.memories.tu_estrategia import TuMemoria
from memory_arena.evaluation.mab_runner import run_strategy_mab
# ... cargar samples con load_mab, instanciar TuMemoria, llamar al runner
```

### 7.4. Scorear

Los scripts `score_d_*` son agnósticos de la estrategia — solo procesan el
JSONL. Se re-usan sin modificar.

### 7.5. Tests

Agregar `tests/test_<tu_estrategia>.py` con `pytest`. Mirar `tests/test_memory_agent_bench.py` como referencia de estilo.

---

## 8. Qué NO hacer

- **No modificar** `memory_arena/memories/base.py` (es contrato estable).
- **No modificar** los runners (`runner.py`, `mab_runner.py`, `judgment_runner.py`, `mab_judgment_runner.py`) sin avisar a Ceci — son infra compartida.
- **No llamar a `pip install`** para agregar deps. Usar `uv add`.
- **No commitear** JSONLs de resultados, entity2id.json, ni archivos pesados de MovieLens. El gitignore está configurado — si algo nuevo debería ignorarse, agregalo explícito.
- **No cambiar** `temperature`, `seed`, `top_p` o `num_ctx` por capricho. Si tu estrategia necesita cambios, documentalos en `notas-informe-tp.md` como limitación metodológica.
- **No inventar** firmas de `MemoriaBase`, `OllamaClient` o `MABSample`. Si no estás segura/o, pedile al usuario que pegue el archivo real.

---

## 9. Protocolo de ayuda

Cuando el usuario te pide ayuda:

1. **Preguntá en qué estrategia trabaja** si no lo dice.
2. **Pedí ver código actual** (pegar el archivo o un snippet) antes de sugerir cambios. No inventes el estado actual.
3. **Respetá los invariantes** de la sección 4 (ejecución, reproducibilidad, paths, estilo).
4. **Sugerí incrementos chicos testables**, no refactorings grandes.
5. **Si el usuario quiere cambiar algo de `base.py` o los runners**, avisale que eso toca infra compartida y sugiere que consulte con Ceci antes.

---

## 10. Recursos

- **ONBOARDING humano**: `docs/ONBOARDING.md` del repo.
- **Notas del informe**: `notas-informe-tp.md` en el Drive compartido del equipo (no está en el repo — es documento vivo de decisiones y findings).
- **Papers**:
  - LongMemEval: https://arxiv.org/abs/2410.10813
  - MemoryAgentBench: https://arxiv.org/abs/2507.05257
- **Ollama**: https://ollama.com/
- **uv**: https://docs.astral.sh/uv/

---

*Fin del contexto. Ahora esperá que el usuario se presente con su nombre + estrategia y ayudalo de acuerdo a la sección 6 correspondiente.*
