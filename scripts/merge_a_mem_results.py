#!/usr/bin/env python3
"""
Merge A_MEM results: copia archivos faltantes del VM staging al local,
y mergea los infbench_sum chunks en un único jsonl con metadata reconstruido.

NO BORRA NADA. Solo agrega.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING = REPO_ROOT / "from_vm" / "staging" / "results"
LOCAL = REPO_ROOT / "results"

INFBENCH_SUM_SOURCES = [
    "responses/20260518_050310_a_mem_mab_Long_Range_Understanding_infbench_sum_eng_shots2.jsonl",
    "responses/20260519_000651_a_mem_mab_Long_Range_Understanding_infbench_sum_eng_shots2.jsonl",
    "responses/20260519_145142_a_mem_mab_Long_Range_Understanding_infbench_sum_eng_shots2.jsonl",
]
INFBENCH_SUM_METADATA_TEMPLATE = (
    "runs/20260519_145142_a_mem_mab_Long_Range_Understanding_infbench_sum_eng_shots2.json"
)


def step_a_copy_missing(execute):
    copied = 0
    skipped = 0
    print("STEP A - Copiar archivos faltantes del VM staging al local")
    print("-" * 72)
    if not STAGING.exists():
        print(f"  [ERROR] staging no existe: {STAGING}")
        return 0, 0

    for subdir in ["responses", "runs", "judgments"]:
        staging_subdir = STAGING / subdir
        local_subdir = LOCAL / subdir
        if not staging_subdir.exists():
            print(f"  [INFO] staging/{subdir}/ no existe, skip")
            continue
        for src in sorted(staging_subdir.iterdir()):
            if not src.is_file():
                continue
            dst = local_subdir / src.name
            if dst.exists():
                skipped += 1
                print(f"  [SKIP] ya en local:  {subdir}/{src.name}")
                continue
            if execute:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
                print(f"  [COPY] {subdir}/{src.name}")
            else:
                copied += 1
                print(f"  [DRY-RUN COPY] {subdir}/{src.name}")
    print(f"\n  Resumen Step A: {copied} {'copiados' if execute else 'a copiar'}, "
          f"{skipped} skipped (ya existian)")
    print()
    return copied, skipped


def step_b_merge_infbench_sum(execute):
    print("STEP B - Mergear infbench_sum_eng_shots2 en un unico jsonl")
    print("-" * 72)

    seen_samples = set()
    records_out = []
    total_input_lines = 0
    per_source_counts = []

    for src_rel in INFBENCH_SUM_SOURCES:
        src = LOCAL / src_rel
        if not src.exists():
            print(f"  [WARN] source missing: {src_rel}")
            continue
        lines_in_source = 0
        added_from_source = 0
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_input_lines += 1
                lines_in_source += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [WARN] linea malformada en {src.name}")
                    continue
                qid = record.get("sample_id")
                if not qid:
                    print(f"  [WARN] record sin sample_id en {src.name}")
                    continue
                if qid in seen_samples:
                    continue
                seen_samples.add(qid)
                records_out.append(record)
                added_from_source += 1
        per_source_counts.append((src.name, lines_in_source, added_from_source))

    for name, n_lines, n_added in per_source_counts:
        print(f"  source {name}: {n_lines} lineas, {n_added} samples nuevos")
    print(f"  TOTAL unique samples a escribir: {len(records_out)}")
    print(f"  (input lines: {total_input_lines}, duplicados ignorados: "
          f"{total_input_lines - len(records_out)})")

    meta_template_path = LOCAL / INFBENCH_SUM_METADATA_TEMPLATE
    if not meta_template_path.exists():
        print(f"  [ERROR] metadata template no existe: {INFBENCH_SUM_METADATA_TEMPLATE}")
        return False
    template = json.loads(meta_template_path.read_text(encoding="utf-8"))

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    new_run_id = (
        f"{timestamp}_a_mem_mab_Long_Range_Understanding_"
        f"infbench_sum_eng_shots2_merged"
    )

    new_meta = dict(template)
    new_meta["run_id"] = new_run_id
    new_meta["num_samples"] = len(records_out)
    new_meta["started_at"] = now.isoformat()
    new_meta["ended_at"] = now.isoformat()
    new_meta["_note"] = (
        "Merged offline desde 20260518_050310, 20260519_000651, 20260519_145142. "
        "Dedupeado por sample_id. Template metadata: 20260519_145142."
    )

    new_jsonl_path = LOCAL / f"responses/{new_run_id}.jsonl"
    new_meta_path = LOCAL / f"runs/{new_run_id}.json"

    if new_jsonl_path.exists():
        print(f"  [SKIP] ya existe: {new_jsonl_path.relative_to(REPO_ROOT)}")
        return False
    if new_meta_path.exists():
        print(f"  [SKIP] ya existe: {new_meta_path.relative_to(REPO_ROOT)}")
        return False

    if execute:
        new_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_jsonl_path, "w", encoding="utf-8") as f:
            for record in records_out:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        new_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_meta_path, "w", encoding="utf-8") as f:
            json.dump(new_meta, f, ensure_ascii=False, indent=2)

        print(f"  [WRITE] {new_jsonl_path.relative_to(REPO_ROOT)} "
              f"({len(records_out)} samples)")
        print(f"  [WRITE] {new_meta_path.relative_to(REPO_ROOT)}")
    else:
        print(f"  [DRY-RUN WRITE] {new_jsonl_path.relative_to(REPO_ROOT)} "
              f"({len(records_out)} samples)")
        print(f"  [DRY-RUN WRITE] {new_meta_path.relative_to(REPO_ROOT)}")
    print()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    execute = args.execute
    mode = "EXECUTE (cambios reales)" if execute else "DRY-RUN (sin cambios)"

    print("=" * 72)
    print(f"merge_a_mem_results.py - {mode}")
    print("=" * 72)
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"STAGING:   {STAGING.relative_to(REPO_ROOT)}")
    print(f"LOCAL:     {LOCAL.relative_to(REPO_ROOT)}")
    print()

    if not LOCAL.exists():
        print(f"[ERROR] local results no existe: {LOCAL}")
        return 1
    if not STAGING.exists():
        print(f"[ERROR] VM staging no existe: {STAGING}")
        return 1

    copied, skipped = step_a_copy_missing(execute)
    ok_b = step_b_merge_infbench_sum(execute)

    print("=" * 72)
    print("RESUMEN FINAL")
    print("=" * 72)
    print(f"  Step A - copias: {copied} {'realizadas' if execute else 'a realizar'}, "
          f"{skipped} skipped")
    print(f"  Step B - merge infbench_sum: {'OK' if ok_b else 'FALLO'}")
    print()
    if not execute:
        print("Esto fue dry-run. Para ejecutar de verdad:")
        print("  python scripts/merge_a_mem_results.py --execute")
    else:
        print("Cambios aplicados. Verifica con:")
        print("  ls results/responses/ | grep merged")
        print("  wc -l results/responses/*merged*.jsonl")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
