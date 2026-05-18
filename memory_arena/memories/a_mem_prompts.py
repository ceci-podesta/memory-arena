"""
memory_arena.memories.a_mem_prompts
------------------------------------
Prompts plain-text y parsers para A-MEM, transcritos fielmente de
``llm_text_parsers.py`` del repo oficial del paper
(https://github.com/WujiangXu/A-mem), versión ROBUSTA para modelos chicos.

Por qué versión robusta y no la JSON simple (memory_layer.py): los autores
publicaron DOS versiones del pipeline. La JSON simple usa structured output
de la API de OpenAI (json_schema), que garantiza output válido pero NO
está disponible cuando corremos contra Ollama con modelos chicos como
llama3.2:3b. La robusta usa plain text con marcadores de sección
(``KEYWORDS:``, ``CONTEXT:``, ``DECISION:``, etc.), parsea con regex, y
tiene fallback heurístico sin LLM. Los autores la diseñaron para los
modelos chicos de su Tabla 1 (Llama-3.2-1B/3B, Qwen-1.5B/3B) — los mismos
que usamos nosotros.

Diferencias respecto al original:
  - Docstrings en castellano.
  - Type hints estilo PEP 604.
  - Quitamos los parsers de QA/queries (parse_plain_text_answer, etc.) que
    pertenecen al pipeline de retrieve_memory_llm — nuestro retrieve es
    cosine puro, no usa LLM.
  - Quitamos FOCUSED_KEYWORDS_PROMPT: si los keywords vienen vacíos del
    LLM, caemos directo al heurístico ``_heuristic_keywords`` (1 call menos).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable


# ----------------------------------------------------------------------------
# Utilidades de limpieza
# ----------------------------------------------------------------------------


def strip_markdown_fences(text: str) -> str:
    """Quita ```json ... ``` o ``` ... ``` que algunos modelos agregan."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_with_json_fallback(
    response: str, plain_text_parser: Callable, *parser_args: Any
) -> Any:
    """Intenta parsear JSON primero; si falla, cae al parser plain-text.

    Muchos modelos emiten JSON válido aun sin structured output, así que
    probamos eso primero (best-of-both-worlds).
    """
    try:
        cleaned = strip_markdown_fences(response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return plain_text_parser(response, *parser_args)


# ----------------------------------------------------------------------------
# Helpers de parsing de listas y secciones
# ----------------------------------------------------------------------------


def _parse_list_items(text: str) -> list[str]:
    """Parsea un texto en lista de items.

    Soporta: bullets (``-``, ``*``, numerados), comma-separated, y
    one-per-line. Quita comillas y bullet markers.
    """
    if not text or not text.strip():
        return []

    lines = text.strip().splitlines()
    items: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        if "," in line:
            for part in line.split(","):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)
    return items


def _extract_section(
    text: str, marker: str, next_markers: list[str] | None = None
) -> str:
    """Extrae el texto entre ``marker:`` y el próximo marker conocido (o fin).

    Args:
        text: respuesta del LLM completa.
        marker: header de sección a buscar (ej: ``"KEYWORDS"``).
        next_markers: lista de posibles siguientes headers.

    Returns:
        El contenido de esa sección (puede ser string vacío).
    """
    pattern = re.compile(
        rf"^\s*{re.escape(marker)}\s*:\s*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    first_line = match.group(1).strip()

    end = len(text)
    if next_markers:
        for nm in next_markers:
            nm_pattern = re.compile(
                rf"^\s*{re.escape(nm)}\s*:", re.IGNORECASE | re.MULTILINE
            )
            nm_match = nm_pattern.search(text, start)
            if nm_match and nm_match.start() < end:
                end = nm_match.start()

    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


# ----------------------------------------------------------------------------
# Prompt templates (textuales del paper, A-mem-main/llm_text_parsers.py)
# ----------------------------------------------------------------------------

ANALYZE_CONTENT_PROMPT = """Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this format (one section per header):

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...

Content for analysis:
{content}"""


EVOLUTION_DECISION_PROMPT = """You are an AI memory evolution agent. Analyze the new memory note and its nearest neighbors to decide if evolution is needed.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

Based on the relationships between the new memory and its neighbors, decide:
- NO_EVOLUTION: The memory stands alone, no changes needed.
- STRENGTHEN: The new memory should be linked to some neighbors and its tags updated.
- UPDATE_NEIGHBOR: The neighbors' context/tags should be updated based on new understanding.
- STRENGTHEN_AND_UPDATE: Both strengthen and update neighbors.

Respond using EXACTLY this format:
DECISION: <one of NO_EVOLUTION, STRENGTHEN, UPDATE_NEIGHBOR, STRENGTHEN_AND_UPDATE>
REASON: <brief explanation>"""


STRENGTHEN_DETAILS_PROMPT = """Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbor indices should the new memory connect to? What tags best describe this memory?

Respond using EXACTLY this format:
CONNECTIONS: 0, 2, 3
TAGS: tag1, tag2, tag3, ..."""


UPDATE_NEIGHBORS_PROMPT = """Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

For each neighbor (indexed 0 to {max_neighbor_idx}), provide updated context and tags. If no change is needed, repeat the original values.

Respond using EXACTLY this format (one block per neighbor):

NEIGHBOR 0:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

NEIGHBOR 1:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

(continue for all {neighbor_count} neighbors)"""


# ----------------------------------------------------------------------------
# Builders (formatean el prompt con los inputs concretos)
# ----------------------------------------------------------------------------


def build_analyze_content_prompt(content: str) -> str:
    return ANALYZE_CONTENT_PROMPT.format(content=content)


def build_evolution_decision_prompt(
    context: str,
    content: str,
    keywords: list[str],
    nearest_neighbors_memories: str,
) -> str:
    return EVOLUTION_DECISION_PROMPT.format(
        context=context,
        content=content,
        keywords=", ".join(keywords),
        nearest_neighbors_memories=nearest_neighbors_memories,
    )


def build_strengthen_details_prompt(
    content: str,
    keywords: list[str],
    nearest_neighbors_memories: str,
) -> str:
    return STRENGTHEN_DETAILS_PROMPT.format(
        content=content,
        keywords=", ".join(keywords),
        nearest_neighbors_memories=nearest_neighbors_memories,
    )


def build_update_neighbors_prompt(
    content: str,
    context: str,
    nearest_neighbors_memories: str,
    neighbor_count: int,
) -> str:
    return UPDATE_NEIGHBORS_PROMPT.format(
        content=content,
        context=context,
        nearest_neighbors_memories=nearest_neighbors_memories,
        max_neighbor_idx=max(0, neighbor_count - 1),
        neighbor_count=neighbor_count,
    )


# ----------------------------------------------------------------------------
# Parsers — uno por cada call site del pipeline
# ----------------------------------------------------------------------------


def parse_analyze_content(response: str, content: str = "") -> dict[str, Any]:
    """Parsea la respuesta de analyze_content.

    Returns:
        ``{"keywords": [...], "context": "...", "tags": [...]}``.
        Si el LLM falla, completa con heurísticas sobre ``content``.
    """

    def _section_parse(resp: str, content_text: str = "") -> dict[str, Any]:
        keywords_text = _extract_section(resp, "KEYWORDS", ["CONTEXT", "TAGS"])
        context_text = _extract_section(resp, "CONTEXT", ["TAGS", "KEYWORDS"])
        tags_text = _extract_section(resp, "TAGS", ["KEYWORDS", "CONTEXT"])

        return {
            "keywords": _parse_list_items(keywords_text),
            "context": context_text.strip() if context_text.strip() else "",
            "tags": _parse_list_items(tags_text),
        }

    result = parse_with_json_fallback(response, _section_parse, content)
    return validate_analysis_result(result, content)


def parse_evolution_decision(response: str) -> dict[str, str]:
    """Parsea la respuesta de evolution decision.

    Returns:
        ``{"decision": "<one of valid>", "reason": "..."}``.
        Default: ``NO_EVOLUTION`` si no se puede inferir nada.
    """

    def _section_parse(resp: str) -> dict[str, str]:
        decision_text = _extract_section(resp, "DECISION", ["REASON"])
        reason_text = _extract_section(resp, "REASON", ["DECISION"])

        decision = decision_text.strip().upper().replace(" ", "_")
        valid = {
            "NO_EVOLUTION",
            "STRENGTHEN",
            "UPDATE_NEIGHBOR",
            "STRENGTHEN_AND_UPDATE",
        }
        if decision not in valid:
            resp_upper = resp.upper()
            if "STRENGTHEN" in resp_upper and "UPDATE" in resp_upper:
                decision = "STRENGTHEN_AND_UPDATE"
            elif "STRENGTHEN" in resp_upper:
                decision = "STRENGTHEN"
            elif "UPDATE" in resp_upper:
                decision = "UPDATE_NEIGHBOR"
            else:
                decision = "NO_EVOLUTION"

        return {"decision": decision, "reason": reason_text.strip()}

    result = parse_with_json_fallback(response, _section_parse)

    # Mapeo desde el JSON-style del pipeline simple, por si el modelo emite eso.
    if "should_evolve" in result:
        should_evolve = result.get("should_evolve", False)
        actions = result.get("actions", []) or []
        if not should_evolve:
            decision = "NO_EVOLUTION"
        elif "strengthen" in actions and "update_neighbor" in actions:
            decision = "STRENGTHEN_AND_UPDATE"
        elif "strengthen" in actions:
            decision = "STRENGTHEN"
        elif "update_neighbor" in actions:
            decision = "UPDATE_NEIGHBOR"
        else:
            decision = "NO_EVOLUTION"
        result = {"decision": decision, "reason": ""}

    result.setdefault("decision", "NO_EVOLUTION")
    result.setdefault("reason", "")
    return result


def parse_strengthen_details(response: str) -> dict[str, Any]:
    """Parsea la respuesta de strengthen details.

    Returns:
        ``{"connections": [int, ...], "tags": [str, ...]}``.
    """

    def _section_parse(resp: str) -> dict[str, Any]:
        conn_text = _extract_section(resp, "CONNECTIONS", ["TAGS"])
        tags_text = _extract_section(resp, "TAGS", ["CONNECTIONS"])

        connections: list[int] = []
        for item in _parse_list_items(conn_text):
            try:
                connections.append(int(item.strip()))
            except (ValueError, TypeError):
                pass
        return {"connections": connections, "tags": _parse_list_items(tags_text)}

    result = parse_with_json_fallback(response, _section_parse)

    if "suggested_connections" in result and "connections" not in result:
        result["connections"] = [
            int(x)
            for x in result.get("suggested_connections", [])
            if isinstance(x, (int, float))
        ]
    if "tags_to_update" in result and "tags" not in result:
        result["tags"] = result.get("tags_to_update", [])

    result.setdefault("connections", [])
    result.setdefault("tags", [])
    return result


def parse_update_neighbors(
    response: str, num_neighbors: int
) -> list[dict[str, Any]]:
    """Parsea la respuesta de update_neighbors.

    Returns:
        Lista de ``num_neighbors`` dicts ``{"context": "...", "tags": [...]}``.
        Si un vecino no fue mencionado en la respuesta, se devuelve dict vacío
        para esa posición.
    """

    def _section_parse(resp: str, n_neighbors: int) -> list[dict[str, Any]]:
        neighbors: list[dict[str, Any]] = []
        for i in range(n_neighbors):
            pattern = re.compile(rf"NEIGHBOR\s+{i}\s*:", re.IGNORECASE)
            match = pattern.search(resp)
            if not match:
                neighbors.append({"context": "", "tags": []})
                continue

            next_pattern = re.compile(r"NEIGHBOR\s+\d+\s*:", re.IGNORECASE)
            next_match = next_pattern.search(resp, match.end())
            block_end = next_match.start() if next_match else len(resp)
            block = resp[match.end() : block_end]

            ctx = _extract_section(block, "CONTEXT", ["TAGS"])
            tags_text = _extract_section(block, "TAGS", ["CONTEXT"])
            neighbors.append(
                {"context": ctx.strip(), "tags": _parse_list_items(tags_text)}
            )
        return neighbors

    # Intentamos JSON primero (forma del pipeline simple)
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            contexts = data.get("new_context_neighborhood", [])
            tags_list = data.get("new_tags_neighborhood", [])
            out: list[dict[str, Any]] = []
            for i in range(num_neighbors):
                ctx = contexts[i] if i < len(contexts) else ""
                tags = tags_list[i] if i < len(tags_list) else []
                out.append({"context": ctx, "tags": tags})
            return out
    except (json.JSONDecodeError, ValueError):
        pass

    return _section_parse(response, num_neighbors)


# ----------------------------------------------------------------------------
# Validación + repair heurístico
# ----------------------------------------------------------------------------


def validate_analysis_result(
    result: dict[str, Any], content: str = ""
) -> dict[str, Any]:
    """Valida y repara el resultado de analyze_content.

    - Si keywords vacíos: extrae heurísticamente de ``content``.
    - Si context vacío: primera oración de ``content``.
    - Si tags vacíos: deriva de keywords.
    """
    if not isinstance(result, dict):
        result = {"keywords": [], "context": "", "tags": []}

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    if isinstance(keywords, str):
        keywords = _parse_list_items(keywords)
    if isinstance(tags, str):
        tags = _parse_list_items(tags)
    if isinstance(context, list):
        context = " ".join(context)

    if not keywords and content:
        keywords = _heuristic_keywords(content)
    if not context and content:
        context = _heuristic_context(content)
    if not tags and keywords:
        tags = keywords[:3]

    result["keywords"] = keywords
    result["context"] = context
    result["tags"] = tags
    return result


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "because", "but", "and", "or",
        "if", "while", "about", "up", "it", "its", "i", "me", "my",
        "you", "your", "he", "she", "they", "we", "this", "that", "these",
        "those", "what", "which", "who", "whom", "says", "said", "speaker",
    }
)


def _heuristic_keywords(content: str, max_keywords: int = 5) -> list[str]:
    """Keywords heurísticos sin LLM. Prioriza palabras capitalizadas."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", content)
    scored: list[tuple[str, int]] = []
    seen: set[str] = set()
    for w in words:
        w_lower = w.lower()
        if w_lower in _STOP_WORDS or w_lower in seen:
            continue
        seen.add(w_lower)
        score = 2 if w[0].isupper() else 1
        scored.append((w_lower, score))
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_keywords]]


def _heuristic_context(content: str) -> str:
    """Context heurístico: primera oración del content, o primeros 200 chars."""
    match = re.match(r"(.+?[.!?])\s", content)
    if match:
        return match.group(1).strip()
    return content[:200].strip()
