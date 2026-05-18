"""
Resume runner generico para MAB.


Re-corre el pipeline MAB para una combinacion (split, sub_dataset, strategy),
salteando los sample_ids que ya tengan respuesta en CUALQUIER jsonl previo
en results/responses/ para esa combinacion.


Uso:
  uv run python scripts/run_mab_resume.py \
      --split Long_Range_Understanding \
      --sub infbench_sum_eng_shots2


Splits validos: Long_Range_Understanding | Conflict_Resolution
                Accurate_Retrieval       | Test_Time_Learning


El sub-dataset es el string tal cual aparece en los nombres de archivo
(ej. detective_qa, factconsolidation_sh_32k, longmemeval_s*).


Al terminar, vas a tener un nuevo .jsonl con los samples faltantes.
Para tener el set completo, al momento de scoring concatena todos
los .jsonl de esa (split, sub).
"""
from __future__ import annotations


import argparse
import json
from pathlib import Path


from memory_arena.benchmarks.memory_agent_bench import load_mab
from memory_arena.evaluation.mab_eta_helper import (
    run_strategy_mab_with_eta as run_strategy_mab,
)
from memory_arena.llm.ollama_client import OllamaClient
from memory_arena.memories.a_mem import AgenticMemory




# Solo A-MEM por ahora. Agregar otras estrategias aca si se necesita.
STRATEGY_FACTORIES = {
    "a_mem": lambda llm: AgenticMemory(
        llm=llm, chunk_size=1500, chunk_overlap=100
    ),
}




def find_done_sample_ids(
    responses_dir: Path,
    strategy: str,
    split: str,
    sub_dataset: str,
) -> set[str]:
    """Escanea responses/ y devuelve set de sample_ids ya completados."""
    # mab_eta_helper normaliza el nombre de sub asi:
    sub_clean = sub_dataset.replace("*", "_star").replace("/", "_")
    pattern = f"*_{strategy}_mab_{split}_{sub_clean}.jsonl"
    done: set[str] = set()
    matched_files: list[Path] = []
    for fp in responses_dir.glob(pattern):
        matched_files.append(fp)
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
        except OSError as e:
            print(f"  WARNING: no pude leer {fp}: {e}")
    print(f"  Archivos matched: {len(matched_files)}")
    for f in matched_files:
        print(f"    - {f.name}")
    return done




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                    help="Long_Range_Understanding | Conflict_Resolution | Accurate_Retrieval | Test_Time_Learning")
    ap.add_argument("--sub", required=True,
                    help="Nombre del sub_dataset (ej. infbench_sum_eng_shots2)")
    ap.add_argument("--strategy", default="a_mem",
                    choices=list(STRATEGY_FACTORIES.keys()))
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()


    results_dir = Path(args.results_dir)
    responses_dir = results_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)


    print("=" * 70)
    print(f"RESUME — split={args.split} sub={args.sub} strategy={args.strategy}")
    print("=" * 70)


    print("\n[1/4] Buscando samples ya completados...")
    done = find_done_sample_ids(
        responses_dir, args.strategy, args.split, args.sub
    )
    print(f"  -> {len(done)} sample_id(s) ya completados.")


    print("\n[2/4] Cargando dataset...")
    samples = load_mab(args.split, args.sub)
    print(f"  -> {len(samples)} sample(s) totales.")


    print("\n[3/4] Filtrando faltantes...")
    missing = [s for s in samples if s.sample_id not in done]
    print(f"  -> {len(missing)} sample(s) a correr.")


    if not missing:
        print("\nNada que correr — todo completo. Exit.")
        return


    print("\n[4/4] Corriendo...")
    llm = OllamaClient()
    strategy = STRATEGY_FACTORIES[args.strategy](llm)
    print(f"  LLM: {llm.model} | num_ctx: {llm.num_ctx}")


    meta = run_strategy_mab(
        strategy=strategy,
        samples=missing,
        llm=llm,
        strategy_name=args.strategy,
        split=args.split,
        sub_dataset=args.sub,
        output_dir=results_dir,
    )


    print(f"\n=== Done. New run_id: {meta.run_id} ===")
    print("Para el set completo, concatenar todos los .jsonl de esa (split, sub).")




if __name__ == "__main__":
    main()
