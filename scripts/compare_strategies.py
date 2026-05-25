"""
Compara a_mem vs no_memoria en los 5 benchmarks.
Imprime una tabla por benchmark con delta a_mem vs baseline.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, a_mem_csv, baseline_csv, label_col)
BENCHMARKS = [
    ("D.AR (deterministic)",
     "results/runs/d_ar_a_mem_scores.csv",
     "results/runs/d_ar_nomemoria_scores.csv",
     "sub_dataset"),
    ("D.CR (deterministic)",
     "results/runs/d_cr_a_mem_scores.csv",
     "results/runs/d_cr_nomemoria_scores.csv",
     "sub_dataset"),
    ("D.TTL (deterministic)",
     "results/runs/d_ttl_a_mem_scores.csv",
     "results/runs/d_ttl_nomemoria_scores.csv",
     "sub_dataset"),
    ("D.LRU (Mistral judge)",
     "results/runs/d_lru_a_mem_scores.csv",
     "results/runs/d_lru_nomemoria_scores.csv",
     "sub_dataset"),
    ("LongMemEval Oracle (Mistral judge)",
     "results/longmemeval_longmemeval_oracle_a_mem_scores.csv",
     "results/longmemeval_longmemeval_oracle_no_memory_scores.csv",
     "question_type"),
]


def load_csv(rel_path):
    p = ROOT / rel_path
    if not p.exists():
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def fmt(val):
    try:
        return f"{float(val):.4f}"
    except (ValueError, TypeError):
        return str(val) if val else "-"


def fmt_delta(a, b):
    try:
        a, b = float(a), float(b)
        d = a - b
        sign = "+" if d >= 0 else ""
        return f"{a:.4f}  Δ={sign}{d:.4f}"
    except (ValueError, TypeError):
        return f"{fmt(a)}"


def compare(name, a_path, b_path, label_col):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)
    a = load_csv(a_path)
    b = load_csv(b_path)
    if a is None:
        print(f"  [MISSING a_mem] {a_path}")
        return
    if b is None:
        print(f"  [MISSING baseline] {b_path}")
        return

    # Index baseline by label_col
    b_idx = {row.get(label_col, ""): row for row in b}

    # Detect metric columns (numeric, excluding label and n)
    skip = {label_col, "n", "kind"}
    metric_cols = [c for c in a[0].keys() if c not in skip]

    # Print header
    h = f"{label_col:<35} {'n':>5}  "
    h += "  ".join(f"{m:<22}" for m in metric_cols)
    print(h)
    print("-" * len(h))

    for row in a:
        lbl = row.get(label_col, "")
        n = row.get("n", "")
        br = b_idx.get(lbl, {})
        line = f"{lbl:<35} {str(n):>5}  "
        for m in metric_cols:
            av = row.get(m, "")
            bv = br.get(m, "")
            line += f"{fmt_delta(av, bv):<22}  "
        print(line)
    print("-" * len(h))

    # Note for missing baseline rows
    a_labels = {row.get(label_col, "") for row in a}
    b_only = [lbl for lbl in b_idx.keys() if lbl not in a_labels]
    if b_only:
        print(f"  (baseline tiene labels que a_mem no: {b_only})")


def main():
    print("COMPARACIÓN  a_mem  vs  no_memoria  —  Δ = a_mem - baseline")
    for name, ap, bp, lbl in BENCHMARKS:
        compare(name, ap, bp, lbl)
    print()


if __name__ == "__main__":
    main()
