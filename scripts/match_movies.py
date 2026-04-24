"""
scripts/match_movies.py
------------------------
Port a Python 3 del script `scripts/match_movies.py` del repo oficial de
ReDial (Li et al., NeurIPS 2018,
https://github.com/RaymondLi0/conversational-recommendations).

Cruza `movies_with_mentions.csv` (ReDial) con `movies.csv` (MovieLens)
matcheando por nombre (con variantes ±"The"/"A"/"&→and"), y produce
`movies_merged.csv` con 4 columnas:

    index, movieName, databaseId, movielensId

Donde `index` es un reindex secuencial 0-N sobre todas las películas
de ReDial + todas las de MovieLens que no matchearon.

Este `index` es (hipótesis fuerte) lo que MemoryAgentBench usa como
movie_id en el dataset `recsys_redial_full`. Si la hipótesis es correcta,
después podemos regenerar `entity2id.json` desde `movies_merged.csv` y
scorear Recall@K de forma compatible con el paper.

Atribución: lógica de matching copiada textualmente del repo oficial; solo
se portó a Python 3 y se acomodaron los paths al repo.

Uso:
    uv run python scripts/match_movies.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REDIAL_CSV = REPO_ROOT / "data" / "recsys_redial" / "movies_with_mentions.csv"
DEFAULT_ML_CSV = REPO_ROOT / "data" / "movielens" / "ml-25m" / "movies.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "recsys_redial" / "movies_merged.csv"


def get_movies_db(path: Path) -> dict[int, tuple[str, str | None]]:
    """Carga un CSV de películas y separa nombre/año.

    Acepta tanto el formato ReDial (`movieId, movieName, nbMentions`) como
    el formato MovieLens (`movieId, title, genres`). Ambos tienen `movieId`
    en la primera columna y el título en la segunda.

    Returns:
        {movieId: (movieName_sin_año, year_str | None)}
    """
    id2movie: dict[int, tuple[str, str | None]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "movieId":
                continue
            # Separar el título de la película del año si está presente
            pattern = re.compile(r"(.+)\((\d+)\)")
            match = re.search(pattern, row[1])
            if match is not None:
                content = (match.group(1).strip(), match.group(2))
            else:
                content = (row[1].strip(), None)
            id2movie[int(row[0])] = content
    print(f"loaded {len(id2movie)} movies from {path}")
    return id2movie


def make_name(name: str, year: str | None) -> str:
    """Reconstruye el nombre canónico 'Name (YEAR)' si hay año."""
    if year is None:
        return name
    if int(year) >= 1900:
        return name + " (" + year + ")"
    return name


def find_in_file(
    db_path: Path, movielens_path: Path, write_to: Path
) -> dict:
    """Para cada película de ReDial, busca su equivalente en MovieLens.

    Matching (igual al oficial):
      - igual nombre exacto
      - o con ", The" / ", A" appended al final
      - o con "The " / "A " prepended al inicio
      - Año tiene que ser None o igual.
    """
    movielens = get_movies_db(movielens_path)
    movies_db = get_movies_db(db_path)
    matched_movies: dict[int, tuple] = {}

    total_exact_matches = 0
    total_movie_not_matched = 0
    for movieId, (db_name, year) in tqdm(movies_db.items()):
        processed_name = db_name.strip().replace("&", "and")
        # Remover "The " / "A " al inicio para evitar problemas de formato
        # (ej. "Avengers, The (2012)").
        if processed_name.startswith("The "):
            processed_name = processed_name[4:]
        if processed_name.startswith("A "):
            processed_name = processed_name[2:]

        found = 0
        for movielensId, (movielens_name, movielens_year) in movielens.items():
            ml_name = movielens_name.replace("&", "and")
            if (
                processed_name == ml_name
                or processed_name + ", The" == ml_name
                or "The " + processed_name == ml_name
                or processed_name + ", A" == ml_name
                or "A " + processed_name == ml_name
            ) and (
                movielens_year is None or year is None or movielens_year == year
            ):
                found = 1
                matched_movies[movieId] = (
                    db_name,
                    year,
                    movielensId,
                    (movielens_name, movielens_year),
                )
                total_exact_matches += 1
                break
        if found == 0:
            total_movie_not_matched += 1

    print(
        f"De {len(movies_db)} películas en ReDial, {total_exact_matches} "
        f"matchearon exacto con MovieLens; {total_movie_not_matched} sin match."
    )

    with open(write_to, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["movieId", "movieName", "movielensId"])
        for key, val in movies_db.items():
            movielensId = matched_movies[key][2] if key in matched_movies else -1
            writer.writerow([key, make_name(*val), movielensId])

    return matched_movies


def _read_csv_id2rest(path: Path) -> dict[int, list[str]]:
    """Lee CSV: devuelve {movieId(int) -> [resto_de_la_fila]}.

    Skipea la primera fila si la primera columna se llama 'movieId'.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return {
            int(row[0]): row[1:]
            for row in reader
            if row[0] != "movieId"
        }


def merge_indexes(
    matched_db_path: Path, movielens_path: Path, write_to: Path
) -> None:
    """Merge ambos listados en un solo CSV con columnas:
    index, movieName, databaseId, movielensId.

    Primero todas las películas de ReDial (con match o sin match), después las
    de MovieLens que no matchearon. `index` es un reindex 0-N secuencial.
    """
    movielens = _read_csv_id2rest(movielens_path)
    matched_db = _read_csv_id2rest(matched_db_path)

    # matched_db es {dbId: [movieName, movielensId]}
    merged = [
        [movie[0], db_id, movie[1]] for db_id, movie in matched_db.items()
    ]

    # Recordar las películas de MovieLens que no tienen match en ReDial
    to_add = {movielensId: True for movielensId in movielens}
    for _db_id, movie in matched_db.items():
        if int(movie[1]) != -1:
            to_add[int(movie[1])] = False

    for movielensId in movielens:
        if to_add[movielensId]:
            # Estas películas no tienen db_id (solo vienen de MovieLens).
            merged.append([movielens[movielensId][0], -1, movielensId])

    with open(write_to, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "movieName", "databaseId", "movielensId"])
        for i, movie in enumerate(merged):
            writer.writerow([i] + movie)

    print(f"\nMerged file escrito a: {write_to}")
    print(f"  Total de filas (index 0 a {len(merged)-1}): {len(merged)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redial_movies_path",
        type=Path,
        default=DEFAULT_REDIAL_CSV,
        help=f"Default: {DEFAULT_REDIAL_CSV}",
    )
    parser.add_argument(
        "--ml_movies_path",
        type=Path,
        default=DEFAULT_ML_CSV,
        help=f"Default: {DEFAULT_ML_CSV}",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    # Verificar inputs
    for p in (args.redial_movies_path, args.ml_movies_path):
        if not p.exists():
            print(f"ERROR: no existe {p}")
            return 1

    args.destination.parent.mkdir(parents=True, exist_ok=True)

    # Archivo intermedio (se borra al final)
    intermediate = args.destination.parent / "movies_matched.csv"

    print(f"Matching ReDial <-> MovieLens...")
    print(f"  ReDial:    {args.redial_movies_path}")
    print(f"  MovieLens: {args.ml_movies_path}")
    print(f"  Output:    {args.destination}")

    find_in_file(
        args.redial_movies_path,
        args.ml_movies_path,
        write_to=intermediate,
    )
    merge_indexes(intermediate, args.ml_movies_path, args.destination)

    # Limpieza
    if intermediate.exists():
        os.remove(intermediate)

    print("\nListo. Próximo paso: regenerar entity2id.json desde "
          f"{args.destination.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
