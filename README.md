# Memory Arena

Comparative evaluation of memory strategies for LLM agents, benchmarked on
LongMemEval and MemoryAgentBench.

Project for Procesamiento de Lenguaje Natural - Maestria en Inteligencia
Artificial, UdeSA, 2026.

## Overview

Large language models have a finite context window. In long-running
conversations or agentic workflows, information mentioned earlier can fall out
of context, making it hard for the model to remember facts, preferences, events
or contradictions across sessions.

This project evaluates external memory mechanisms for LLM agents under a common
interface:

- `store(turn)`: ingest information into memory.
- `retrieve(query, top_k)`: recover relevant memory for a question.
- `reset()`: clear state between independent benchmark samples.

The repository is organized around two benchmarks:

- **LongMemEval**: multi-session conversational memory.
- **MemoryAgentBench (MAB)**: document-style agent memory with four dimensions:
  Accurate Retrieval (D.AR), Conflict Resolution (D.CR), Test-Time Learning
  (D.TTL), and Long-Range Understanding (D.LRU).

## Repository State

The final implementation was developed across multiple branches. To avoid
risking regressions close to submission, the strategy branches were intentionally
not merged into `main`.

`main` contains the shared evaluation infrastructure and the baseline
`NoMemory` / `NoMemoria` strategy.

| Branch | Strategy |
| --- | --- |
| `main` | Baseline: `NoMemory` / `NoMemoria` |
| `verbatim-rag` | Verbatim + RAG |
| `summarized-dt` | Summarized Memory |
| `feature/agentic_memory` | A-MEM / Agentic Memory |
| `feature/graph-memory` | Graph Memory |

Each strategy branch follows the same high-level interface and evaluation
protocol, so results are comparable even though the branches remain separate.

## Project Structure

```text
memory_arena/
  benchmarks/       Dataset loaders for LongMemEval and MemoryAgentBench
  evaluation/       Runners, LLM judges, scoring and run metadata
  llm/              Ollama client wrapper
  memories/         Memory strategy interface and baseline NoMemory

scripts/            Experiment entrypoints and scoring scripts
tests/              Unit and smoke tests
```

## Setup

The project uses Python 3.12 and `uv`.

```bash
uv sync
```

The local LLM backend is Ollama. The experiments use:

```bash
ollama pull llama3.2:3b
ollama pull mistral:7b
```

`llama3.2:3b` is the evaluated model. `mistral:7b` is used as an LLM judge for
LongMemEval and selected MAB LRU tasks.

Some commands download benchmark data from Hugging Face on first run.

## Running Tests

```bash
uv run pytest
```

Tests that require Ollama are skipped automatically when the local Ollama server
is not available.

## Running the Baseline in `main`

LongMemEval:

```bash
uv run python scripts/run_longmemeval.py --strategy no_memoria --subset oracle
uv run python scripts/score_longmemeval.py --strategy no_memoria --subset oracle
```

MemoryAgentBench baseline runs:

```bash
uv run python scripts/run_d_ar_nomemoria.py
uv run python scripts/run_d_cr_nomemoria.py
uv run python scripts/run_d_ttl_nomemoria.py
uv run python scripts/run_d_lru_nomemoria.py
```

Note: the `recsys_redial_full` sub-dataset from MAB D.TTL was not included in
the reported results. We attempted to reconstruct the required movie/entity
mapping from available ReDial and MovieLens metadata, but could not validate
that it matched the original benchmark mapping reliably enough for a fair
comparison.

MAB scoring:

```bash
uv run python scripts/score_d_ar.py --strategy no_memoria
uv run python scripts/score_d_cr.py --strategy no_memoria
uv run python scripts/score_d_ttl.py --strategy no_memoria
uv run python scripts/run_e_lru_judge.py --strategy no_memoria
uv run python scripts/score_e_lru.py --strategy no_memoria
```

## Evaluation Protocol

The evaluation has two phases.

1. **Generation**: a strategy ingests the sample context, retrieves memory for
   each question, and the evaluated LLM produces an answer. Responses are
   written as JSONL files under `results/responses/`.
2. **Scoring / judging**: deterministic metrics or an LLM judge evaluate the
   generated responses. Metadata is written under `results/runs/`; judgments are
   written under `results/judgments/`.

For deterministic MAB tasks, the project reports metrics such as exact match,
substring exact match, F1 and ROUGE recall. For LongMemEval and selected LRU
tasks, the project uses an LLM judge.

## Consolidated Results

The table below summarizes the consolidated results reported in the final
write-up. Values are weighted averages by benchmark / dimension.

| Benchmark | Metric | NoMemory | Summarized | Verbatim + RAG | A-MEM | Graph | Best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AR | RR | 27% | 32% | 42% | 38% | 38% | Verbatim + RAG |
| AR | EM | 9% | 11% | 16% | 13% | 17% | Graph |
| CR | RR | 15% | 6% | 25% | 17% | 21% | Verbatim + RAG |
| CR | EM | 2% | 1% | 14% | 10% | 7% | Verbatim + RAG |
| TTL | RR | 0.80% | 0.40% | 3.00% | 3.00% | 3.00% | Tie |
| TTL | SEM | 3.40% | 1.40% | 4.60% | 4.60% | 5.00% | Graph |
| LRU / detective_qa | ACC | 34% | 28% | 31% | 41% | 34% | A-MEM |
| LRU / infbench_sum | Mean F1 | 0.74% | 1.12% | 0.65% | 1.92% | 0.58% | A-MEM |
| LongMemEval | ACC | 20% | 35% | 53% | 63% | 26% | A-MEM |

Metric legend:

- `EM`: exact match.
- `SEM`: substring exact match.
- `RR`: ROUGE-L recall.
- `ACC`: LLM-judge accuracy.
- `Mean F1`: summarization judge F1 for `infbench_sum`.

## Main Takeaways

- No single memory architecture dominates every task.
- Verbatim + RAG is strongest on document-style retrieval and conflict
  resolution tasks.
- A-MEM performs best on LongMemEval and MAB Long-Range Understanding, where
  incremental memory and richer note metadata are useful.
- Graph Memory is competitive on event-style retrieval and some long-context
  settings, but is less consistent overall.
- Summarized Memory is compact, but loses details that matter for exact factual
  retrieval.
- Test-Time Learning remains difficult for all strategies in this setup.

## Reproducibility Notes

- Experiments were run with local Ollama models.
- The default evaluated model is `llama3.2:3b`.
- The default judge model is `mistral:7b`.
- The default context window is configured in
  `memory_arena/experimental_config.py`.
- `recsys_redial_full` is intentionally excluded from the reported D.TTL
  results because the original movie/entity mapping could not be validated.
- Some results depend on long-running jobs and branch-specific strategy code.
  Because the final branches were not merged into `main`, reproduce a strategy
  by checking out its corresponding branch first.

## References

Benchmarks:

- Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
  Memory*, 2024.
- Hu et al., *MemoryAgentBench: Benchmarking Memory Mechanisms for LLM Agents*,
  ICLR 2026.

Memory strategies and related systems:

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks*, 2020.
- Reimers and Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks*, 2019.
- Wang et al., *Recursively Summarizing Enables Long-Term Dialogue Memory in
  Large Language Models*, 2023.
- Zhong et al., *MemoryBank: Enhancing Large Language Models with Long-Term
  Memory*, 2024.
- See, Liu and Manning, *Get To The Point: Summarization with
  Pointer-Generator Networks*, 2017.
- Xu et al., *A-MEM: Agentic Memory for LLM Agents*, 2025.
- Packer et al., *MemGPT: Towards LLMs as Operating Systems*, 2023.
- Gu et al., *LightRAG: Simple and Fast Retrieval-Augmented Generation*, 2024.
- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization*, 2024.
- Rasmussen et al., *Zep: A Temporal Knowledge Graph Architecture for Agent
  Memory*, 2025.
- Chhikara et al., *Mem0: Building Production-Ready AI Agents with Scalable
  Long-Term Memory*, 2024.
