"""
memory_arena.evaluation.recsys_scorer
--------------------------------------
Scorer específico para `recsys_redial_full` (TTL de MAB).

Calcula Recall@1, Recall@5 y Recall@10 entre las películas recomendadas por el
modelo y las gold movie IDs del dataset ReDial (Li et al., NeurIPS 2018).

Flujo (igual al del paper MAB, `utils/eval_other_utils.py::_process_recsys_dataset`):
  1. Cargar `entity2id.json` (map `{"Movie Name (YEAR)": id_int}`), construible
     desde `data/recsys_redial/movies_with_mentions.csv` via
     `scripts/build_entity2id.py`.
  2. Invertir a `id_to_name` aplicando `extract_movie_name` (saca paréntesis del
     año, normaliza whitespace) — así el matching es tolerante a variantes.
  3. Parsear el output del modelo en una lista de recomendaciones
     (`extract_recommendation_list`: split on `"1."`, si falla split on comma).
  4. Para cada item predicho, hacer fuzzy-match por edit distance contra los
     candidates (`find_nearest_movie`).
  5. Convertir los gold IDs a nombres limpios y calcular recall@K.

Atribución: las funciones `clean_parentheses`, `normalize_whitespace`,
`clean_text_elements`, `extract_movie_name`, `find_nearest_movie` y
`extract_recommendation_list` son adaptaciones fieles del repo oficial de
MemoryAgentBench (Apache 2.0, HUST-AI-HYZ). Se reproduce la lógica para
mantener paridad metodológica con el paper.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from editdistance import eval as edit_distance


# ----------------------------------------------------------------------------
# Carga del entity map
# ----------------------------------------------------------------------------

DEFAULT_ENTITY2ID_PATHS: list[Path] = [
    Path("data/recsys_redial/entity2id.json"),
    Path("./processed_data/Recsys_Redial/entity2id.json"),  # convención del paper
]


def load_entity2id(path: Path | None = None) -> dict[str, int]:
    """Carga entity2id.json. Si `path` es None, prueba los paths default.

    Raises:
        FileNotFoundError: si no encuentra el archivo en ningún path candidato.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = DEFAULT_ENTITY2ID_PATHS

    for cand in candidates:
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        "No encontré entity2id.json. Correr primero "
        "`uv run python scripts/build_entity2id.py`. Paths probados: "
        f"{[str(c) for c in candidates]}"
    )


# ----------------------------------------------------------------------------
# Limpieza de texto (adaptado del oficial)
# ----------------------------------------------------------------------------


def clean_parentheses(text: str) -> str:
    """Remueve el contenido entre paréntesis (ej: quitar el año)."""
    return re.sub(r"\([^()]*\)", "", text)


def normalize_whitespace(text: str) -> str:
    """Colapsa whitespace múltiple a un solo espacio y trim."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text_elements(
    text: str,
    remove_parentheses: bool = True,
    normalize_ws: bool = True,
    remove_nums: bool = True,
) -> str:
    """Limpia un elemento de la lista de recomendaciones."""
    if remove_parentheses:
        text = re.sub(r"\([^()]*\)", "", text)
    if remove_nums:
        # Saca numeración del inicio (ej: "1. ", "2) ", "3 - ")
        text = re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)
    if normalize_ws:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_movie_name(text: str) -> str:
    """Limpia un nombre del formato del CSV a un string canónico.

    Ejemplo: 'Angels in the Outfield  (1994)' -> 'Angels in the Outfield'.
    """
    filename = text.split("/")[-1]
    cleaned = filename.replace("_", " ").replace("-", " ").replace(">", " ")
    return normalize_whitespace(clean_parentheses(cleaned))


# ----------------------------------------------------------------------------
# Fuzzy matching
# ----------------------------------------------------------------------------


def find_nearest_movie(target_name: str, candidate_movies: list[str]) -> dict:
    """Encuentra el candidate más cercano por edit distance (case-insensitive).

    Returns:
        {'movie_name': target_name, 'min_edit_distance': int, 'nearest_movie': str}
    """
    unique_candidates = list(set(candidate_movies))
    if not unique_candidates:
        return {
            "movie_name": target_name,
            "min_edit_distance": -1,
            "nearest_movie": "",
        }
    distances = [
        edit_distance(target_name.lower(), c.lower()) for c in unique_candidates
    ]
    nearest_idx = min(range(len(distances)), key=lambda i: distances[i])
    return {
        "movie_name": target_name,
        "min_edit_distance": distances[nearest_idx],
        "nearest_movie": unique_candidates[nearest_idx],
    }


def extract_recommendation_list(
    text: str, movie_candidates: list[str] | None = None
) -> tuple[list[dict] | list[str], str]:
    """Extrae la lista de recomendaciones del output del modelo.

    Si `movie_candidates` es None, devuelve la lista raw (strings). Si se pasa,
    hace fuzzy-match de cada item contra los candidates y devuelve lista de dicts.

    Returns:
        (recommendation_list, preference_text): preference_text es el preamble
        antes de la lista numerada (generalmente ignorado).
    """
    try:
        preference_text, recommendation_text = text.split("1.", maxsplit=1)
    except ValueError:
        preference_text = ""
        # Fallback: reemplazar comas por newlines para parsear.
        recommendation_text = text.replace(",", "\n")

    raw_recommendations = [
        clean_text_elements(item.strip()) for item in recommendation_text.split("\n")
    ]
    # Sacar strings vacíos (líneas en blanco después de limpiar).
    raw_recommendations = [r for r in raw_recommendations if r]

    if movie_candidates is not None:
        matched = [find_nearest_movie(item, movie_candidates) for item in raw_recommendations]
        return matched, preference_text

    return raw_recommendations, preference_text


# ----------------------------------------------------------------------------
# Scoring principal
# ----------------------------------------------------------------------------


def score_recsys_response(
    system_answer: str,
    gold_answers: list | str,
    entity_map: dict[str, int] | None = None,
) -> dict:
    """Scorea una respuesta de recsys contra gold movie IDs.

    Args:
        system_answer: texto crudo del modelo con la lista de recomendaciones.
        gold_answers: IDs de películas gold. Puede ser lista de ints/strings,
            string con IDs separados por coma, o lista de listas.
        entity_map: dict {name_with_year: id}. Si None, se carga con los paths default.

    Returns:
        Dict con:
            - recsys_recall@1, recsys_recall@5, recsys_recall@10
            - n_gold: cantidad de gold movies
            - n_predicted: cantidad de recomendaciones extraídas
            - parsed_output: lista de nombres de películas predichas (limpias)
            - gt_movies: lista de nombres gold (limpios)

    Nota: si `gold_answers` no tiene ningún ID parseable, devuelve recall=0 con
    `n_gold=0` (el llamador puede filtrar estos casos del agregado).
    """
    if entity_map is None:
        entity_map = load_entity2id()

    # id_to_name con nombres limpios (sin año entre paréntesis).
    id_to_name = {
        mid: extract_movie_name(name) for name, mid in entity_map.items()
    }
    movie_candidates = list(id_to_name.values())

    # Parsear gold_answers a lista de ints.
    gold_ids = _parse_gold_ids(gold_answers)
    ground_truth_movies = [id_to_name[mid] for mid in gold_ids if mid in id_to_name]

    # Parsear el output del modelo.
    predicted_list, _ = extract_recommendation_list(system_answer, movie_candidates)
    predicted_movies: list[str] = [
        item["nearest_movie"] if isinstance(item, dict) else item
        for item in predicted_list
    ]

    metrics: dict[str, float | int] = {
        "n_gold": len(ground_truth_movies),
        "n_predicted": len(predicted_movies),
    }

    if not ground_truth_movies:
        # Sin gold no podemos calcular recall — devolvemos 0 y el llamador
        # decide si filtrar.
        metrics.update({
            "recsys_recall@1": 0.0,
            "recsys_recall@5": 0.0,
            "recsys_recall@10": 0.0,
        })
    else:
        for k in (1, 5, 10):
            top_k = predicted_movies[:k]
            hits = sum(1 for gm in ground_truth_movies if gm in top_k)
            metrics[f"recsys_recall@{k}"] = hits / len(ground_truth_movies)

    # Debug info para auditar en el JSONL de salida.
    metrics["_parsed_output"] = predicted_movies
    metrics["_gt_movies"] = ground_truth_movies
    return metrics


def _parse_gold_ids(gold_answers: list | str) -> list[int]:
    """Convierte gold_answers a lista de ints (movie IDs).

    Formatos aceptados:
      - [101, 603]                   -> [101, 603]
      - ["101", "603"]               -> [101, 603]
      - "101,603"                    -> [101, 603]
      - [["101"], ["603"]]           -> [101, 603]
      - [["101,603"]]                -> [101, 603]
    """
    result: list[int] = []
    if isinstance(gold_answers, str):
        for tok in gold_answers.split(","):
            tok = tok.strip()
            if tok.isdigit():
                result.append(int(tok))
        return result

    if not isinstance(gold_answers, list):
        return result

    def flatten(x):
        if isinstance(x, (list, tuple)):
            for item in x:
                yield from flatten(item)
        else:
            yield x

    for item in flatten(gold_answers):
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, str):
            for tok in item.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    result.append(int(tok))
    return result
