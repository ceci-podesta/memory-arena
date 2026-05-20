# memory-arena

Evaluation of different memory strategies for LLM agents, benchmarked on LongMemEval and MemoryAgentBench.

NLP TP - UdeSA 2026.

## SummarizedMemory

Esta rama agrega la estrategia `SummarizedMemory`, implementada en:

```text
memory_arena/memories/summarized.py
```

La idea de la estrategia es mantener una memoria compacta en forma de resumen acumulativo. Cada vez que entran nuevos turnos o documentos, la memoria actualiza un `summary` usando un LLM local via Ollama. Luego, en `retrieve`, devuelve ese resumen como contexto para que el modelo responda.

Decisiones principales:

- La estrategia usa `OllamaClient` con `llama3.2:3b` para resumir y responder.
- Para conversaciones, acumula turnos recientes y los consolida cada `summarize_every`.
- En `retrieve`, consolida los turnos pendientes antes de devolver contexto. Esto evita que una corrida con `summarize_every` alto pierda informacion que todavia no habia llegado al resumen.
- Para documentos largos de MAB, aplica chunking map-reduce.
- El tamano default de chunk es `30000` caracteres.
- `max_document_chunks` permite limitar cuantos chunks procesar en pruebas rapidas. Para corridas completas, dejarlo sin definir.

## Chunking

MAB puede traer documentos muy largos, incluso mayores al contexto efectivo que Ollama puede procesar en una unica llamada. Por eso `SummarizedMemory` no manda todo el documento completo al LLM de una sola vez.

Flujo usado para `Turn(role="document")`:

1. Divide el documento en chunks de hasta `document_chunk_chars`.
2. Resume cada chunk por separado.
3. Hace una reduccion final combinando los resumenes parciales.
4. Guarda el resultado en `summary`.

Parametro default:

```text
document_chunk_chars = 30000
```

Este valor busca un equilibrio: chunks suficientemente grandes para no multiplicar demasiado el costo computacional, pero evitando mandar documentos gigantes en una sola llamada.

Para pruebas rapidas se puede limitar:

```powershell
--max-document-chunks 1
```

Para corridas finales no usar ese parametro.

## Requisitos

Instalar dependencias con `uv`:

```powershell
uv sync
```

Verificar Ollama:

```powershell
ollama --version
ollama list
```

Modelos usados:

```powershell
ollama pull llama3.2:3b
ollama pull mistral:7b
```

`llama3.2:3b` se usa para generar respuestas. `mistral:7b` se usa como juez LLM en LongMemEval y LRU.

## Tests

Tests unitarios de `SummarizedMemory`:

```powershell
uv run pytest tests/test_summarized.py -v
```

Resultado validado:

```text
15 passed
```

Tests generales sin integracion:

```powershell
uv run pytest tests/ -v -m "not integration"
```

Resultado validado:

```text
84 passed, 4 deselected
```

## Smoke Tests

Smoke test chico de MAB con `SummarizedMemory`:

```powershell
uv run python scripts/run_summarized_smoke.py --max-samples 1
```

Version con resumen mas largo:

```powershell
uv run python scripts/run_summarized_smoke.py --max-samples 1 --summary-max-tokens 1024
```

Score del smoke CR:

```powershell
uv run python scripts/score_d_cr.py --strategy summarized
```

Resultados observados en CR `factconsolidation_sh_6k`, 1 sample / 100 preguntas:

```text
summary_max_tokens=384:
exact_match=0.0400, substring_exact_match=0.0500, f1=0.0415

summary_max_tokens=1024:
exact_match=0.0600, substring_exact_match=0.0900, f1=0.0753
```

## MAB - Generacion Fase A

Scripts oficiales para correr MAB con `SummarizedMemory`:

```powershell
uv run python scripts/run_d_cr_summarized.py
uv run python scripts/run_d_ar_summarized.py
uv run python scripts/run_d_ttl_summarized.py
uv run python scripts/run_d_ttl_recsys_summarized.py
uv run python scripts/run_d_lru_summarized.py
```

Para pruebas chicas:

```powershell
uv run python scripts/run_d_cr_summarized.py --max-subdatasets 1 --max-samples 1
uv run python scripts/run_d_ar_summarized.py --max-subdatasets 1 --max-samples 1
uv run python scripts/run_d_ttl_summarized.py --max-subdatasets 1 --max-samples 1
uv run python scripts/run_d_ttl_recsys_summarized.py --max-samples 1
uv run python scripts/run_d_lru_summarized.py --max-subdatasets 1 --max-samples 1
```

Parametros comunes:

```powershell
--summarize-every 100
--keep-recent 3
--summary-max-tokens 512
--document-chunk-chars 30000
--answer-max-tokens 64
```

Para una corrida de mayor calidad, especialmente en PC mas potente:

```powershell
--summary-max-tokens 1024 --answer-max-tokens 128
```

Para LRU, el default de respuesta es mas alto:

```powershell
--answer-max-tokens 256
```

## MAB - Scores Lexicos

Despues de generar respuestas:

```powershell
uv run python scripts/score_d_cr.py --strategy summarized
uv run python scripts/score_d_ar.py --strategy summarized
uv run python scripts/score_d_ttl.py --strategy summarized
uv run python scripts/score_d_ttl_recsys.py --strategy summarized
uv run python scripts/score_d_lru.py --strategy summarized
```

## MAB - Juez LRU

LRU requiere juez LLM para la evaluacion principal:

```powershell
uv run python scripts/run_e_lru_judge.py --strategy summarized
uv run python scripts/score_e_lru.py --strategy summarized
```

El juez usa `mistral:7b`.

## LongMemEval

El acuerdo de equipo es correr LongMemEval sobre `oracle` para reducir tiempo de computo.

Corrida chica:

```powershell
uv run python scripts/run_longmemeval_summarized.py --subset oracle --limit 5 --summary-max-tokens 1024 --summarize-every 50 --answer-max-tokens 128
```

Corrida sugerida para escalar:

```powershell
uv run python scripts/run_longmemeval_summarized.py --subset oracle --limit 50 --summarize-every 100 --summary-max-tokens 512 --answer-max-tokens 64
```

Corrida completa oracle:

```powershell
uv run python scripts/run_longmemeval_summarized.py --subset oracle --limit 500 --summarize-every 100 --summary-max-tokens 512 --answer-max-tokens 64
```

Si se quiere priorizar calidad sobre velocidad:

```powershell
uv run python scripts/run_longmemeval_summarized.py --subset oracle --limit 500 --summarize-every 50 --summary-max-tokens 1024 --answer-max-tokens 128
```

## LongMemEval - Juez y Score

Primero generar judgment con Mistral:

```powershell
uv run python scripts/run_longmemeval_judge.py --responses results/responses/<RUN_ID>.jsonl
```

Luego score:

```powershell
uv run python scripts/score_longmemeval.py --strategy summarized --subset longmemeval_oracle
```

Tambien se puede scorear una corrida especifica:

```powershell
uv run python scripts/score_longmemeval.py --strategy summarized --run-id <RUN_ID>
```

Resultado observado en `oracle limit=5` despues de agregar flush en `retrieve`:

```text
overall_accuracy=0.4
```

## Flujo Recomendado

1. Correr tests unitarios.
2. Correr smoke tests con `--max-samples 1`.
3. Correr MAB Fase A para CR, AR, TTL, TTL Recsys y LRU.
4. Correr scores lexicos MAB.
5. Correr juez y score de LRU.
6. Correr LongMemEval oracle.
7. Correr juez y score de LongMemEval.

Para corridas finales en PC potente, evitar `--max-samples`, `--max-subdatasets` y `--max-document-chunks`.
