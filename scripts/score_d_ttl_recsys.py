"""
scripts/score_d_ttl_recsys.py
------------------------------
Bloque D.TTL recsys — Fase B: scorea el JSONL de respuestas de
`recsys_redial_full` usando el scorer Bloque F (entity2id + fuzzy matching).

Reporta Recall@1, Recall@5, Recall@10 (globales), más estadísticas de cobertura
del mapping (cuántos gold IDs fueron encontrados en nuestro `entity2id.json`).

La cobertura es el chequeo de sanity para validar que nuestro entity2id
reconstruido desde ReDial es compatible con los gold IDs que trae HuggingFace.
Si la cobertura es baja (<80%), es indicio de que MAB usó una versión distinta
del mapping — documentado como limitación en la sección 10 del informe.

Uso:
    uv run python scripts/run_d_ttl_recsys.py    # primero (si no lo hiciste)
    uv run python scripts/score_d_ttl_recsys.py  # después
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_arena.evaluation.mab_scoring import score_jsonl
from memory_arena.evaluation.recsys_scorer import load_entity2id


RESPONSES_DIR = REPO_ROOT / "results" / "responses"
RUNS_DIR = REPO_ROOT / "results" / "runs"
OUTPUT_CSV = RUNS_DIR / "d_ttl_recsys_scores.csv"

METRIC_KEYS = ["recsys_recall@1", "recsys_recall@5", "recsys_recall@10"]


def find_latest_recsys_jsonl() -> Path | None:
    """Busca el JSONL más reciente de NoMemoria sobre recsys_redial_full."""
    if not RESPONSES_DIR.exists():
        return None
    candidates = [
        p for p in RESPONSES_DIR.glob("*.jsonl")
        if "recsys_redial_full" in p.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def check_entity2id_coverage(jsonl_path: Path) -> dict:
    """Chequea qué porcentaje de gold IDs del JSONL existen en entity2id.json.

    Devuelve {total_golds, found, coverage_pct, missing_ids_sample}.
    Si entity2id no existe, devuelve None (el scorer ya va a fallar).
    """
    try:
        name_to_id = load_entity2id()
    except FileNotFoundError:
        return {
            "total_golds": 0,
            "found": 0,
            "coverage_pct": 0.0,
            "error": "entity2id.json no encontrado — correr build_entity2id.py",
            "missing_ids_sample": [],
        }

    known_ids = set(name_to_id.values())

    total_golds = 0
    found = 0
    missing: list[int] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            golds = rec.get("gold_answers") or []
            if isinstance(golds, str):
                golds = [golds]
            for g in golds:
                if isinstance(g, list):
                    # Puede venir como lista de lista.
                    for sub in g:
                        _tally_gold(sub, known_ids, missing_sink=missing, total=lambda: None)
                        total_golds += 1
                        if _gold_in_known(sub, known_ids):
                            found += 1
                else:
                    total_golds += 1
                    if _gold_in_known(g, known_ids):
                        found += 1
                    else:
                        missing.append(_gold_to_int(g))

    coverage = (found / total_golds * 100.0) if total_golds else 0.0
    return {
        "total_golds": total_golds,
        "found": found,
        "coverage_pct": round(coverage, 2),
        "missing_ids_sample": missing[:10],  # primeros 10 para debug
    }


def _gold_to_int(g) -> int:
    if isinstance(g, int):
        return g
    if isinstance(g, str) and g.strip().isdigit():
        return int(g.strip())
    return -1


def _gold_in_known(g, known_ids: set[int]) -> bool:
    gi = _gold_to_int(g)
    return gi in known_ids


def _tally_gold(g, known_ids, missing_sink, total):
    # Helper para recorrer nested lists — dejamos el counting en la función principal.
    pass


def main() -> int:
    jsonl = find_latest_recsys_jsonl()
    if jsonl is None:
        print(
            f"No hay JSONL de recsys_redial_full en {RESPONSES_DIR}. "
            "Correr scripts/run_d_ttl_recsys.py primero.",
            file=sys.stderr,
        )
        return 1

    print("=" * 72)
    print("D.TTL recsys — Fase B (scoring)")
    print("=" * 72)
    print(f"Input: {jsonl.relative_to(REPO_ROOT)}")

    # Cobertura del mapping (sanity check antes de scorear)
    print("\nChequeando cobertura entity2id vs gold IDs del JSONL...")
    cov = check_entity2id_coverage(jsonl)
    if "error" in cov:
        print(f"  ERROR: {cov['error']}", file=sys.stderr)
        return 1
    print(f"  Gold IDs totales: {cov['total_golds']}")
    print(f"  Encontrados en entity2id: {cov['found']}  ({cov['coverage_pct']}%)")
    if cov["missing_ids_sample"]:
        print(
            f"  Primeros missing IDs (debug): {cov['missing_ids_sample']}"
        )
    if cov["coverage_pct"] < 80.0:
        print(
            "  ⚠ Cobertura < 80%: el entity2id reconstruido puede no ser "
            "compatible con el usado por MAB. Revisar sección 10 del informe."
        )

    # Scoring
    print(f"\nScoreando con scorer recsys (Recall@1/5/10)...")
    result = score_jsonl(jsonl)
    agg = result["aggregates"]
    n_total = agg["n_total"]
    by_metric = agg["by_metric"]

    print("\n" + "=" * 72)
    print(f"Resultados globales (n={n_total} queries)")
    print("=" * 72)
    for k in METRIC_KEYS:
        if k in by_metric:
            m = by_metric[k]
            print(f"  {k:<25}: {m['mean']:.4f}  (n={m['n']})")
        else:
            print(f"  {k:<25}: — (no reportada)")

    # n_gold / n_predicted también agregadas si vinieron
    for extra in ("n_gold", "n_predicted"):
        if extra in by_metric:
            print(f"  {extra:<25}: mean={by_metric[extra]['mean']:.2f}")

    # Dump CSV
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "n"])
        for k in METRIC_KEYS:
            if k in by_metric:
                m = by_metric[k]
                writer.writerow([k, f"{m['mean']:.6f}", m["n"]])
        for extra in ("n_gold", "n_predicted"):
            if extra in by_metric:
                m = by_metric[extra]
                writer.writerow([extra, f"{m['mean']:.4f}", m["n"]])
        # coverage fila final
        writer.writerow([
            "_entity2id_coverage_pct",
            f"{cov['coverage_pct']:.2f}",
            cov["total_golds"],
        ])
    print(f"\nCSV: {OUTPUT_CSV.relative_to(REPO_ROOT)}")

    print("\nNOTA: NoMemoria para recsys es el piso parametric (sin contexto).")
    print("Las estrategias con memoria real (verbatim+RAG, etc.) deberían")
    print("superar significativamente estos valores en Recall@5 y Recall@10.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
