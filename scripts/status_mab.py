"""
Diagnostico de progreso MAB para A-MEM.


Recorre todos los (split, sub_dataset) de MAB, lee los .jsonl ya escritos
en results/responses/ con strategy=a_mem, y reporta cuantos samples de cada
test estan completos vs el total.


Read-only, no llama al LLM, no toca data. Tarda <30s la primera vez
(carga datasets de HF), <5s las veces siguientes (cache).


Uso:
  uv run python scripts/status_mab.py
"""
from __future__ import annotations


import json
from pathlib import Path


from memory_arena.benchmarks.memory_agent_bench import (
    SPLIT_AR,
    SPLIT_CR,
    SPLIT_LRU,
    SPLIT_TTL,
    SUB_AR_EVENTQA_131K,
    SUB_AR_EVENTQA_65K,
    SUB_AR_EVENTQA_FULL,
    SUB_AR_LONGMEMEVAL_S,
    SUB_AR_RULER_QA1,
    SUB_AR_RULER_QA2,
    SUB_CR_FACTCONSOL_MH_6K,
    SUB_CR_FACTCONSOL_MH_32K,
    SUB_CR_FACTCONSOL_MH_64K,
    SUB_CR_FACTCONSOL_MH_262K,
    SUB_CR_FACTCONSOL_SH_6K,
    SUB_CR_FACTCONSOL_SH_32K,
    SUB_CR_FACTCONSOL_SH_64K,
    SUB_CR_FACTCONSOL_SH_262K,
    SUB_LRU_DETECTIVE_QA,
    SUB_LRU_INFBENCH_SUM,
    SUB_TTL_ICL_BANKING77,
    SUB_TTL_ICL_CLINIC150,
    SUB_TTL_ICL_NLU,
    SUB_TTL_ICL_TREC_COARSE,
    SUB_TTL_ICL_TREC_FINE,
    load_mab,
)


STRATEGY = "a_mem"
RESPONSES_DIR = Path("results/responses")


ALL_COMBOS: list[tuple[str, list[str]]] = [
    (SPLIT_LRU, [SUB_LRU_DETECTIVE_QA, SUB_LRU_INFBENCH_SUM]),
    (SPLIT_CR, [
        SUB_CR_FACTCONSOL_SH_6K,
        SUB_CR_FACTCONSOL_SH_32K,
        SUB_CR_FACTCONSOL_SH_64K,
        SUB_CR_FACTCONSOL_SH_262K,
        SUB_CR_FACTCONSOL_MH_6K,
        SUB_CR_FACTCONSOL_MH_32K,
        SUB_CR_FACTCONSOL_MH_64K,
        SUB_CR_FACTCONSOL_MH_262K,
    ]),
    (SPLIT_AR, [
        SUB_AR_LONGMEMEVAL_S,
        SUB_AR_EVENTQA_FULL,
        SUB_AR_EVENTQA_65K,
        SUB_AR_EVENTQA_131K,
        SUB_AR_RULER_QA1,
        SUB_AR_RULER_QA2,
    ]),
    (SPLIT_TTL, [
        SUB_TTL_ICL_BANKING77,
        SUB_TTL_ICL_CLINIC150,
        SUB_TTL_ICL_NLU,
        SUB_TTL_ICL_TREC_COARSE,
        SUB_TTL_ICL_TREC_FINE,
    ]),
]




def done_sample_ids(split: str, sub: str) -> set[str]:
    sub_clean = sub.replace("*", "_star").replace("/", "_")
    pattern = f"*_{STRATEGY}_mab_{split}_{sub_clean}.jsonl"
    done: set[str] = set()
    for fp in RESPONSES_DIR.glob(pattern):
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        sid = rec.get("sample_id")
                        if sid:
                            done.add(sid)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return done




def main() -> None:
    print("=" * 85)
    print(f"STATUS MAB ({STRATEGY})")
    print("=" * 85)
    header = f"{'SPLIT':<26} {'SUB_DATASET':<32} {'DONE/TOT':>10}  PCT   ST"
    print(header)
    print("-" * 85)


    grand_done = 0
    grand_total = 0


    for split, subs in ALL_COMBOS:
        for sub in subs:
            try:
                samples = load_mab(split, sub)
                total = len(samples)
            except Exception as e:
                print(f"{split:<26} {sub:<32}      ERROR  {e}")
                continue


            done = done_sample_ids(split, sub)
            n_done = len(done)
            pct = (100 * n_done / total) if total else 0
            if n_done == 0:
                status = "❌"
            elif n_done == total:
                status = "✓"
            else:
                status = "⚠"


            print(f"{split:<26} {sub:<32} {n_done:>4}/{total:<4}  {pct:5.1f}%  {status}")
            grand_done += n_done
            grand_total += total


    print("-" * 85)
    grand_pct = (100 * grand_done / grand_total) if grand_total else 0
    print(f"{'TOTAL':<26} {'':<32} {grand_done:>4}/{grand_total:<4}  {grand_pct:5.1f}%")
    print()
    print("Leyenda: ✓ completo  |  ⚠ parcial  |  ❌ sin empezar")
    print(f"Para retomar un parcial:  uv run python scripts/run_mab_resume.py --split <X> --sub <Y>")




if __name__ == "__main__":
    main()
