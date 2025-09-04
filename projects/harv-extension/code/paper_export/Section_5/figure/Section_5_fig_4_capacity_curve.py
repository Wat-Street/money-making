#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3A: #Primes vs SMAPE (capacity vs performance)

Default behavior:
- Reads a prepared CSV:
    outputs/intraday/tables/table_5a_pm_num_primes.csv
- Renders a journal-style line with error bars (mean Â± s.e.)
- Writes PDF + SVG + a LaTeX snippet.

Optional:
- If --auto-sweep is set, this script will first call:
    code/sweep_pm_capacity.py
  with the provided --assets, --k-grid, --n, --warmup, --local-dir to
  create the feeder CSV, then plot.

CSV expected schema:
  num_primes, smape_pct_mean, smape_pct_se [, family]

Usage (plot only):
  python code/paper_export/fig_3a_pm_capacity.py \
    --csv outputs/intraday/tables/table_5a_pm_num_primes.csv \
    --outdir outputs/intraday/figures

Usage (auto-sweep + plot):
  python code/paper_export/fig_3a_pm_capacity.py \
    --auto-sweep \
    --assets SPY,QQQ \
    --k-grid 3,4,5,6,7,8 \
    --n 390 --warmup 200 \
    --local-dir code/Datasets/clean \
    --csv outputs/intraday/tables/table_5a_pm_num_primes.csv \
    --outdir outputs/intraday/figures
"""

import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _run_sweep_if_requested(args):
    if not args.auto_sweep:
        return
    sweep_py = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Helpers_for_paper_export_scripts", "Section_5", "sweep_pm_capacity.py")
    if not os.path.exists(sweep_py):
        raise SystemExit(f"sweep script not found: {sweep_py}")
    cmd = [
        'python', sweep_py,
        '--assets', args.assets,
        '--k-grid', args.k_grid,
        '--n', str(args.n),
        '--warmup', str(args.warmup),
        '--local-dir', args.local_dir,
        '--out-csv', args.csv
    ]
    print("[fig_3a] running sweep:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--prefix', default='Section_5_fig_4_capacity_curve')

    # optional auto-sweep
    ap.add_argument('--auto-sweep', action='store_true')
    ap.add_argument('--assets', type=str, default=None)
    ap.add_argument('--k-grid', type=str, default=None)
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--local-dir', type=str, default=None)

    args = ap.parse_args()

    # optionally build feeder CSV
    if args.auto_sweep:
        for req in ['assets','k_grid','n','local_dir']:
            if getattr(args, req) in (None, ''):
                raise SystemExit(f"--auto-sweep requires --{req.replace('_','-')}")
        _run_sweep_if_requested(args)

    if not os.path.exists(args.csv):
        raise SystemExit(f"Capacity CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if 'num_primes' not in df.columns or 'smape_pct_mean' not in df.columns:
        raise SystemExit("CSV must include 'num_primes' and 'smape_pct_mean' (and optionally 'smape_pct_se').")
    if 'family' not in df.columns:
        df['family'] = 'PM'

    families = df['family'].unique().tolist()

    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    colors = {'PM':'#222222', 'CP':'#666666'}
    markers = {'PM':'o', 'CP':'s'}

    for fam in families:
        sub = df[df['family']==fam].sort_values('num_primes')
        x = sub['num_primes'].values
        y = sub['smape_pct_mean'].values
        se = sub.get('smape_pct_se', pd.Series([np.nan]*len(sub))).values
        ax.errorbar(x, y, yerr=se, fmt=markers.get(fam,'o'), color=colors.get(fam,'#222222'),
                    linewidth=1.4, capsize=3, label=('Prime Modulo' if fam=='PM' else 'Contiguous Prime'))

    ax.set_xlabel('# of primes (capacity)')
    ax.set_ylabel('SMAPE (%)')
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if len(families) > 1: ax.legend(frameon=False)
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    pdf = os.path.join(args.outdir, f'{args.prefix}.pdf')
    svg = os.path.join(args.outdir, f'{args.prefix}.svg')
    fig.savefig(pdf, bbox_inches='tight'); fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)

    tex = os.path.join(args.outdir, f'{args.prefix}.tex')
    with open(tex,'w') as f:
        f.write(
r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{%s}
\caption{Figure 3A: Capacity vs performance for the prime-based family. We plot SMAPE (mean $\pm$ s.e.) as a function of the number of primes used to construct features. We expect diminishing returns beyond the minimal covering set.}
\label{fig:pm_capacity}
\end{figure}
""" % (pdf.replace('\\','/'))
        )
    print(f"[ok] wrote {pdf}\n[ok] wrote {svg}\n[ok] wrote {tex}")

if __name__ == '__main__':
    main()

