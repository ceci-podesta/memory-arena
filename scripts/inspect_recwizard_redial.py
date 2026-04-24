"""
scripts/inspect_recwizard_redial.py
------------------------------------
Script de inspección (no escribe nada). Carga los 3 subsets del dataset
`recwizard/redial` en HuggingFace, baja 5 filas de cada uno, y reporta:

- Campos disponibles
- Rango de movieId (si está)
- Primeras filas (truncadas)

Objetivo: verificar si alguno de los subsets (SA, autorec, formatted) usa un
esquema de IDs en el rango que usa MemoryAgentBench (509 a ~23561) — lo que
nos permitiría construir un entity2id.json compatible.

No descarga el dataset completo: `load_dataset` normalmente lo cachea, pero
solo iteramos 5 filas de cada subset, así que el tiempo/datos es mínimo.

Uso:
    uv run python scripts/inspect_recwizard_redial.py
"""

from __future__ import annotations

import sys
from typing import Any

from datasets import load_dataset


DATASET_NAME = "recwizard/redial"
SUBSETS = ["SA", "autorec", "formatted"]
N_SAMPLES = 5  # solo para peek; no cargamos el dataset entero


def _truncate_value(v: Any, max_len: int = 200) -> str:
    """Formatea un valor para imprimir sin spamear."""
    s = repr(v)
    if len(s) > max_len:
        return s[:max_len] + f"... (len={len(s)})"
    return s


def inspect_subset(name: str) -> None:
    print("\n" + "=" * 72)
    print(f"SUBSET: {name}")
    print("=" * 72)

    try:
        # streaming=True evita descargar el dataset entero; solo leemos N filas.
        ds = load_dataset(DATASET_NAME, name, split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR al cargar: {e}")
        return

    it = iter(ds)
    rows: list[dict] = []
    for _ in range(N_SAMPLES):
        try:
            rows.append(next(it))
        except StopIteration:
            break

    if not rows:
        print("  (subset vacío)")
        return

    print(f"\nCampos disponibles: {list(rows[0].keys())}")

    # Rango de movieId (si el campo existe)
    if "movieId" in rows[0]:
        ids_sample = [r["movieId"] for r in rows]
        print(f"movieId muestra: {ids_sample}")
        print(f"movieId min/max (en muestra): {min(ids_sample)} / {max(ids_sample)}")

    # Intentar recorrer más filas SOLO para el rango de movieId (sin imprimir).
    # Límite seguro: 2000 filas para no tardar mucho.
    if "movieId" in rows[0]:
        print("Chequeando rango en 2000 filas adicionales...")
        all_ids = list(ids_sample)
        for i, extra in enumerate(it):
            if i >= 2000:
                break
            if "movieId" in extra:
                all_ids.append(extra["movieId"])
        print(f"  movieId min/max (en {len(all_ids)} filas): "
              f"{min(all_ids)} / {max(all_ids)}")
        # Check si hay IDs en el rango de MAB
        mab_range_lo, mab_range_hi = 500, 24000
        in_range = sum(1 for x in all_ids if mab_range_lo <= x <= mab_range_hi)
        print(
            f"  IDs en rango MAB ({mab_range_lo}-{mab_range_hi}): "
            f"{in_range}/{len(all_ids)} ({100*in_range/len(all_ids):.1f}%)"
        )

    # Muestra las primeras 3 filas (campos, valores truncados)
    print("\nPrimeras 3 filas (valores truncados):")
    for idx, row in enumerate(rows[:3]):
        print(f"\n  --- fila #{idx} ---")
        for k, v in row.items():
            print(f"    {k}: {_truncate_value(v)}")


def main() -> int:
    print(f"Inspeccionando dataset: {DATASET_NAME}")
    print(f"Subsets: {SUBSETS}")

    for sub in SUBSETS:
        try:
            inspect_subset(sub)
        except Exception as e:
            print(f"\n[{sub}] ERROR inesperado: {e}", file=sys.stderr)

    print("\n" + "=" * 72)
    print("Fin de la inspección.")
    print("=" * 72)
    print(
        "Buscamos un subset con movieIds en rango MAB (500-24000) "
        "para usarlo como fuente del entity2id."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
