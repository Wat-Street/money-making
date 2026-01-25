#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 2 (Intraday only): Performance by RV regime
- Columns: Model, Low RV, Medium RV, High RV (SMAPE% mean ± s.e.)
- Rows: all models present in CSV by default; --no-variants to keep cores only
- Formatting: bold = best, underline = second-best per column
- Notes: lower is better; ± is standard error; regimes defined by RV terciles (33%/66%)
"""

import os
import argparse
import numpy as np
import pandas as pd

DISPLAY_ORDER = [
    'HAR','HAR_J','HAR_CJ','HAR_TCJ',
    'PM','PM_VW','PM_AD',
    'CP','CP_CJ',
    'EXH','HAM','RAND','CRS'
]

DISPLAY_NAME = {
    'HAR': 'HAR-RV',
    'HAR_J': 'HAR-RV-J',
    'HAR_CJ': 'HAR-RV-CJ',
    'HAR_TCJ': 'HAR-RV-TCJ',
    'PM': 'Prime Modulo (PM)',
    'PM_VW': 'Prime Modulo (PM-VW)',
    'PM_AD': 'Prime Modulo (PM-AD)',
    'CP': 'Contiguous Prime (CP)',
    'CP_CJ': 'Contiguous Prime (CP-CJ)',
    'EXH': 'Exhaustive Prefixes (EXH)',
    'HAM': 'Hamming Codes (HAM)',
    'RAND': 'Randomized (control)',
    'CRS': 'Contiguous Random (control)',
}

def fmt_pm(x, se, digits=2):
    x = float(x) if pd.notna(x) else np.nan
    se = float(se) if pd.notna(se) else 0.0
    if not np.isfinite(x):
        return '--'
    return f'{x:.{digits}f} $\\pm$ {se:.{digits}f}'

def bold_underline(col_strs, higher_is_better=False):
    # col_strs: list of formatted strings like "123.45 ± 0.67"
    # Find numeric core value for ranking (left of first space)
    vals = []
    for s in col_strs:
        try:
            num = s.split(' ')[0].replace('\\textbf{','').replace('\\underline{','').replace('}','')
            vals.append(float(num))
        except Exception:
            vals.append(np.inf if not higher_is_better else -np.inf)
    vals = np.array(vals, dtype=float)
    order = np.argsort(-vals) if higher_is_better else np.argsort(vals)
    out = []
    for i, s in enumerate(col_strs):
        if i == order[0]:
            out.append(r'\textbf{' + s + '}')
        elif len(order) > 1 and i == order[1]:
            out.append(r'\underline{' + s + '}')
        else:
            out.append(s)
    return out

def build_table(df: pd.DataFrame, include_variants: bool, caption: str, label: str) -> str:
    keep = DISPLAY_ORDER if include_variants else [
        'HAR','HAR_J','HAR_CJ','HAR_TCJ','PM','CP','EXH','HAM','RAND','CRS'
    ]
    df = df[df['model'].isin(keep)].copy()
    df['model'] = pd.Categorical(df['model'], categories=DISPLAY_ORDER, ordered=True)
    df = df.sort_values('model')

    names = [DISPLAY_NAME.get(m, m) for m in df['model']]

    low = [fmt_pm(a, b, 2) for a, b in zip(df['smape_low_pct_mean'], df['smape_low_pct_se'])]
    med = [fmt_pm(a, b, 2) for a, b in zip(df['smape_med_pct_mean'], df['smape_med_pct_se'])]
    high = [fmt_pm(a, b, 2) for a, b in zip(df['smape_high_pct_mean'], df['smape_high_pct_se'])]

    low  = bold_underline(low,  higher_is_better=False)
    med  = bold_underline(med,  higher_is_better=False)
    high = bold_underline(high, higher_is_better=False)

    colspec = 'lccc'
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{' + caption + r'}')
    lines.append(r'\label{' + label + r'}')
    lines.append(r'\begin{tabular}{' + colspec + r'}')
    lines.append(r'\toprule')
    lines.append(r'Model & Low RV & Medium RV & High RV \\')
    lines.append(r'\midrule')
    for i in range(len(df)):
        lines.append(' {} & {} & {} & {} \\\\'.format(names[i], low[i], med[i], high[i]))
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\vspace{2pt}\par\footnotesize Lower is better. Values are mean SMAPE\% across assets with $\pm$ standard error. ' +
                 r'Regimes are terciles of realized volatility (33\% / 66\%).')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Path to code/outputs/intraday/tables/Section_2_regimes.csv')
    ap.add_argument('--out', required=True, help='Output .tex path')
    ap.add_argument('--include-variants', dest='include_variants', action='store_true', default=True)
    ap.add_argument('--no-variants', dest='include_variants', action='store_false')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    tex = build_table(
        df,
        include_variants=args.include_variants,
        caption='Table 2 (Intraday): SMAPE\\% by realized-volatility regime.',
        label='tab:regimes_intraday'
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f'[ok] wrote {args.out}')

if __name__ == '__main__':
    main()
