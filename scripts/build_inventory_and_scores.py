"""Arma file_index.md + all_scores_long.csv + all_scores_wide.csv."""
import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

STRATEGIES_RAW = ['a_mem', 'no_memoria', 'no_memory', 'nomemoria']
SPLITS = ['Long_Range_Understanding', 'Conflict_Resolution',
          'Test_Time_Learning', 'Accurate_Retrieval']


def norm_strategy(s):
    if s in ('no_memoria', 'no_memory', 'nomemoria'):
        return 'no_memoria'
    return s


def parse_filename(name):
    base = name.rsplit('.', 1)[0]
    is_judge = base.endswith('__mistral')
    if is_judge:
        base = base[:-len('__mistral')]
    m = re.match(r'^(\d{8}_\d{6})_(.+)$', base)
    if not m:
        return None
    ts, after_ts = m.group(1), m.group(2)
    strategy = None
    for s in STRATEGIES_RAW:
        if after_ts.startswith(s + '_') or after_ts == s:
            strategy = norm_strategy(s)
            after_ts = after_ts[len(s) + 1:] if after_ts != s else ""
            break
    if strategy is None:
        return None
    if after_ts.startswith('mab_'):
        rest = after_ts[4:]
        for sp in SPLITS:
            if rest.startswith(sp + '_'):
                sub = rest[len(sp) + 1:]
                return {'timestamp': ts, 'strategy': strategy, 'family': 'mab',
                        'split': sp, 'sub_dataset': sub, 'is_judge': is_judge}
    elif after_ts.startswith('longmemeval_'):
        sub = after_ts[len('longmemeval_'):]
        return {'timestamp': ts, 'strategy': strategy, 'family': 'longmemeval',
                'split': None, 'sub_dataset': sub, 'is_judge': is_judge}
    return None


def collect_response_files():
    groups = defaultdict(lambda: defaultdict(
        lambda: {'responses': [], 'metadata': [], 'judgments': []}))
    for subdir, kind in [('responses', 'responses'), ('runs', 'metadata'),
                          ('judgments', 'judgments')]:
        d = RESULTS / subdir
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            parsed = parse_filename(p.name)
            if parsed is None:
                continue
            bench_key = (parsed['family'], parsed['split'], parsed['sub_dataset'])
            groups[bench_key][parsed['strategy']][kind].append({
                'path': str(p.relative_to(ROOT)),
                'timestamp': parsed['timestamp'],
                'is_judge': parsed['is_judge'],
            })
    return groups


def build_index_md(groups):
    SPLIT_ORDER = ['Accurate_Retrieval', 'Conflict_Resolution',
                   'Test_Time_Learning', 'Long_Range_Understanding']
    lines = ["# Index de archivos — a_mem y no_memoria", "",
             "Archivos por benchmark, sub_dataset y estrategia. Paths "
             "relativos a la raiz del repo.", ""]

    longmem_keys = sorted([k for k in groups.keys() if k[0] == 'longmemeval'])
    mab_keys = sorted([k for k in groups.keys() if k[0] == 'mab'],
                      key=lambda x: (SPLIT_ORDER.index(x[1])
                                      if x[1] in SPLIT_ORDER else 999, x[2]))

    for keylist, family_title in [(longmem_keys, "LongMemEval"),
                                    (mab_keys, "MAB benchmarks")]:
        if not keylist:
            continue
        lines.append(f"## {family_title}")
        lines.append("")
        for (family, split, sub) in keylist:
            heading = (f"{split.replace('_', ' ')} — `{sub}`"
                       if family == 'mab' else f"`{sub}`")
            lines.append(f"### {heading}")
            lines.append("")
            for strategy in ['a_mem', 'no_memoria']:
                if strategy not in groups[(family, split, sub)]:
                    continue
                files = groups[(family, split, sub)][strategy]
                if not any(files.values()):
                    continue
                lines.append(f"**{strategy}**")
                lines.append("")
                if files['responses']:
                    lines.append("Responses:")
                    for f in files['responses']:
                        lines.append(f"- `{f['path']}`")
                if files['metadata']:
                    lines.append("")
                    lines.append("Metadata (RunMetadata):")
                    for f in files['metadata']:
                        suffix = "  (judge meta)" if f['is_judge'] else ""
                        lines.append(f"- `{f['path']}`{suffix}")
                if files['judgments']:
                    lines.append("")
                    lines.append("Judgments (verdicts del juez Mistral):")
                    for f in files['judgments']:
                        lines.append(f"- `{f['path']}`")
                lines.append("")
            lines.append("")

    # Scoring CSVs
    lines.append("## Scoring CSVs")
    lines.append("")
    score_files = sorted(set(
        list((RESULTS / 'runs').glob('*scores.csv'))
        + list(RESULTS.glob('*scores.csv'))
    ))
    for p in score_files:
        lines.append(f"- `{p.relative_to(ROOT)}`")
    lines.append("")
    return "\n".join(lines)


def bench_from_csv_name(stem):
    if stem.startswith('d_ar'):
        return 'D.AR'
    if stem.startswith('d_cr'):
        return 'D.CR'
    if stem.startswith('d_ttl'):
        return 'D.TTL'
    if stem.startswith('d_lru'):
        return 'D.LRU (judge)' if 'judge' in stem else 'D.LRU (det.)'
    if stem.startswith('longmemeval'):
        return 'LongMemEval Oracle'
    return None


def strategy_from_csv_name(stem):
    for s in STRATEGIES_RAW:
        if f'_{s}_' in stem or stem.endswith(f'_{s}'):
            return norm_strategy(s)
    if stem == 'd_lru_judge_scores':  # legacy nombre sin strategy
        return 'no_memoria'
    return None


def export_scores_long_wide():
    csv_paths = sorted(set(
        list((RESULTS / 'runs').glob('*scores.csv'))
        + list(RESULTS.glob('*scores.csv'))
    ))
    rows_long = []
    for p in csv_paths:
        bench = bench_from_csv_name(p.stem)
        strategy = strategy_from_csv_name(p.stem)
        if bench is None or strategy is None:
            print(f"  [SKIP] {p.name} (bench={bench}, strategy={strategy})")
            continue
        with open(p) as f:
            reader = csv.DictReader(f)
            fns = reader.fieldnames or []
            label_col = 'sub_dataset' if 'sub_dataset' in fns else (
                'question_type' if 'question_type' in fns else None)
            if label_col is None:
                continue
            for row in reader:
                label = row.get(label_col, '')
                n = row.get('n', '')
                for metric, val in row.items():
                    if metric in (label_col, 'n', 'kind', 'source_path'):
                        continue
                    if not val:
                        continue
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    rows_long.append({
                        'benchmark': bench, 'strategy': strategy,
                        'sub_dataset': label, 'n': n,
                        'metric': metric, 'value': v,
                        'source_csv': str(p.relative_to(ROOT)),
                    })

    out_long = RESULTS / 'all_scores_long.csv'
    with open(out_long, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['benchmark', 'strategy', 'sub_dataset',
                                           'n', 'metric', 'value', 'source_csv'])
        w.writeheader()
        w.writerows(rows_long)

    wide_idx = defaultdict(dict)
    for r in rows_long:
        k = (r['benchmark'], r['sub_dataset'], r['metric'])
        wide_idx[k][r['strategy']] = r['value']
        wide_idx[k]['n'] = r['n']

    out_wide = RESULTS / 'all_scores_wide.csv'
    with open(out_wide, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['benchmark', 'sub_dataset', 'metric',
                                            'n', 'a_mem', 'no_memoria',
                                            'delta_a_mem_minus_no_memoria'])
        w.writeheader()
        for (bench, sub, metric), vals in sorted(wide_idx.items()):
            am = vals.get('a_mem')
            nm = vals.get('no_memoria')
            delta = (am - nm) if (am is not None and nm is not None) else ''
            w.writerow({
                'benchmark': bench, 'sub_dataset': sub, 'metric': metric,
                'n': vals.get('n', ''),
                'a_mem': am if am is not None else '',
                'no_memoria': nm if nm is not None else '',
                'delta_a_mem_minus_no_memoria': delta,
            })
    return len(rows_long), out_long.relative_to(ROOT), out_wide.relative_to(ROOT)


def main():
    print("Recolectando responses / metadata / judgments...")
    groups = collect_response_files()
    print(f"  {len(groups)} benchmarks distintos detectados")

    print("Generando file_index.md...")
    md = build_index_md(groups)
    out_md = RESULTS / 'file_index.md'
    out_md.write_text(md, encoding='utf-8')
    print(f"  -> {out_md.relative_to(ROOT)}")

    print("Exportando scores (long + wide)...")
    n, long_p, wide_p = export_scores_long_wide()
    print(f"  -> {long_p}  ({n} filas)")
    print(f"  -> {wide_p}")
    print("\nListo.")


if __name__ == "__main__":
    main()
