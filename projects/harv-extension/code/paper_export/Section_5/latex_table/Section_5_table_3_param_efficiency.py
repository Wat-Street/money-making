#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 3: Parameter efficiency â€” Error vs Feature Count
Reads the joined CSV (metrics + feature_count). Bold the Pareto frontier
(lowest SMAPE for a given or lower feature count). Optionally include a
'Wins p<0.05' column if a DM-summary CSV is provided.

Expected columns in --csv:
  model, feature_count, smape_pct_mean, smape_pct_se,
  mae_mean, mae_se, rmse_mean, rmse_se,
  diracc_pct_mean, diracc_pct_se
Optional: sig_mark_smape

Optional --dm-summary CSV columns:
  model, wins_p05, n_assets

Usage:
  python code/paper_export/table_3_param_eff_latex.py \
    --csv run_results/current_intraday/tables/Section_5_table_3_param_eff_joined.csv \
    --out code/paper/latex/Section_5_table_3_param_eff.tex
"""

import os, argparse
import numpy as np
import pandas as pd

DISPLAY = {
    'HAR':'HAR-RV', 'HAR_J':'HAR-RV-J', 'HAR_CJ':'HAR-RV-CJ', 'HAR_TCJ':'HAR-RV-TCJ',
    'PM':'Prime Modulo (PM)', 'PM_VW':'PM-VW', 'PM_AD':'PM-AD',
    'CP':'Contiguous Prime (CP)', 'CP_CJ':'CP-CJ',
    'EXH':'Exhaustive Prefixes (EXH)', 'HAM':'Hamming Codes (HAM)',
    'RAND':'Random (control)',
    'CRS':'Contiguous Random (control)',
}

ORDER = ['HAR','HAR_J','HAR_CJ','HAR_TCJ','PM','PM_VW','PM_AD','CP','CP_CJ','EXH','HAM','RAND','CRS']

def fmt_mean_se(mean, se, pct=False):
    if pd.isna(mean):
        return '--'
    if pct:
        if pd.isna(se): return f"{mean:.1f}"
        return f"{mean:.1f} $\\pm$ {se:.1f}"
    else:
        if pd.isna(se): return f"{mean:.3g}"
        return f"{mean:.3g} $\\pm$ {se:.3g}"

def pareto_bools(df):
    # mark rows on the Pareto frontier in (feature_count, smape) space
    d = df.sort_values(['feature_count','smape_pct_mean']).copy()
    best = np.inf
    on = np.zeros(len(d), dtype=bool)
    for i,(fc,sm) in enumerate(zip(d['feature_count'], d['smape_pct_mean'])):
        if sm < best - 1e-12:
            best = sm
            on[i] = True
    d['on_frontier'] = on
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dm-summary', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ['model','feature_count','smape_pct_mean','mae_mean','rmse_mean','diracc_pct_mean']:
        if c not in df.columns:
            raise SystemExit(f"Missing column in {args.csv}: {c}")

    if args.dm_summary and os.path.exists(args.dm_summary):
        dm = pd.read_csv(args.dm_summary)
        if not {'model','wins_p05','n_assets'} <= set(dm.columns):
            dm = None
    else:
        dm = None

    # Keep only models we care about and order them
    df['model'] = df['model'].astype(str)
    df = df[df['model'].isin(ORDER)].copy()
    df['display'] = df['model'].map(DISPLAY).fillna(df['model'])
    df['order'] = df['model'].apply(lambda m: ORDER.index(m))
    df = df.sort_values('order')

    # mark Pareto frontier
    tagged = pareto_bools(df[['model','feature_count','smape_pct_mean']])
    df = df.merge(tagged[['model','on_frontier']], on='model', how='left')

    # attach DM wins if supplied
    if dm is not None:
        df = df.merge(dm[['model','wins_p05','n_assets']], on='model', how='left')

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Table 3: Parameter efficiency. Error vs feature count across models. Bold indicates Pareto-efficient points (lowest SMAPE for a given capacity).}")
    lines.append(r"\label{tab:param_efficiency}")
    cols = [r"Model", r"Feat.", r"SMAPE~(\%)", r"MAE", r"RMSE", r"Dir.~Acc.~(\%)"]
    if dm is not None:
        cols.append(r"Wins $p<0.05$")
    lines.append(r"\begin{tabular}{l r r r r r" + ("" if dm is None else " r") + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        name = row['display']
        if bool(row.get('on_frontier', False)):
            name = r"\textbf{" + name + "}"

        sm = fmt_mean_se(row['smape_pct_mean'], row.get('smape_pct_se', np.nan), pct=True)
        mae = fmt_mean_se(row['mae_mean'], row.get('mae_se', np.nan))
        rmse = fmt_mean_se(row['rmse_mean'], row.get('rmse_se', np.nan))
        da = fmt_mean_se(row.get('diracc_pct_mean', np.nan), row.get('diracc_pct_se', np.nan), pct=True)

        fields = [name, f"{int(row['feature_count'])}", sm, mae, rmse, da]
        if dm is not None:
            wp = row.get('wins_p05', np.nan)
            na = row.get('n_assets', np.nan)
            fields.append("--" if pd.isna(wp) or pd.isna(na) else f"{int(wp)}/{int(na)}")
        lines.append(" & ".join(fields) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\footnotesize{Mean across assets; $\pm$ indicates s.e. across tickers. Pareto frontier is computed in the (feature count, SMAPE) plane.}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w') as f:
        f.write("\n".join(lines))
    print(f"[ok] wrote {args.out}")

if __name__ == '__main__':
    main()



