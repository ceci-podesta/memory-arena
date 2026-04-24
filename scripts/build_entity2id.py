"""
scripts/build_entity2id.py
---------------------------
Construye `data/recsys_redial/entity2id.json` a partir de `movies_merged.csv`
(generado por `scripts/match_movies.py`, que cruza ReDial × MovieLens).

`movies_merged.csv` tiene 4 columnas: index, movieName, databaseId, movielensId.
El `index` es el reindex 0-N sobre películas de ReDial + películas de MovieLens
no matcheadas — hipótesis fuerte: es el movie_id que usa MemoryAgentBench como
gold en `recsys_redial_full`.

Formato de salida: `{"Movie Name (YEAR)": index}`. Compatible con el scorer
oficial de MAB (`_process_recsys_dataset` en eval_other_utils.py).

Versión histórica: hasta 2026-04-23 este script leía `movies_with_mentions.csv`
directo (6924 entradas con movieId de ReDial 75796-206092). Resultó incompatible
con los gold IDs de MAB (509-23561). Lo reemplazamos por la versión que usa
`movies_merged.csv` después del match con MovieLens.

Uso:
    uv run python scripts/match_movies.py       # primero (genera movies_merged.csv)
    uv run python scripts/build_entity2id.py    # después
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGED_CSV_PATH = REPO_ROOT / "data" / "recsys_redial" / "movies_merged.csv"
OUTPUT_JSON = REPO_ROOT / "data" / "recsys_redial" / "entity2id.json"


def main() -> int:
    if not MERGED_CSV_PATH.exists():
        print(f"ERROR: no existe {MERGED_CSV_PATH}", file=sys.stderr)
        print(
            "Correr primero:\n"
            "    uv run python scripts/match_movies.py",
            file=sys.stderr,
        )
        return 1

    entity2id: dict[str, int] = {}
    duplicate_names: list[tuple[str, int, int]] = []

    with open(MERGED_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            name = row["movieName"]
            if name in entity2id:
                duplicate_names.append((name, entity2id[name], idx))
            entity2id[name] = idx  # última wins (consistente con el oficial)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(entity2id, f, ensure_ascii=False, indent=2)

    # Stats útiles para el informe.
    ids = list(entity2id.values())
    print(f"Generado: {OUTPUT_JSON.relative_to(REPO_ROOT)}")
    print(f"  Películas indexadas: {len(entity2id)}")
    print(f"  Fuente: {MERGED_CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"  Rango de index: {min(ids)} a {max(ids)}")

    # Chequeo rápido: ¿los IDs que MAB usa como gold (rango 500-24000) caen
    # dentro del rango de nuestro mapping?
    mab_lo, mab_hi = 500, 24000
    in_mab_range = sum(1 for x in ids if mab_lo <= x <= mab_hi)
    print(
        f"  IDs en rango MAB (~{mab_lo}-{mab_hi}): "
        f"{in_mab_range} de {len(ids)}"
    )

    if duplicate_names:
        print(
            f"\n  Nota: {len(duplicate_names)} nombres duplicados en el CSV "
            f"(variantes con mismo título+año); la última index queda en el dict."
        )
        for name, prev_idx, new_idx in duplicate_names[:5]:
            print(f"    - '{name}': prev_idx={prev_idx} -> kept new_idx={new_idx}")
        if len(duplicate_names) > 5:
            print(f"    ... (+{len(duplicate_names) - 5} más)")

    print(
        "\nPróximo paso: re-correr `uv run python scripts/score_d_ttl_recsys.py` "
        "para ver la cobertura real contra los gold IDs de MAB."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
