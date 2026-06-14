#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 6: Statistical Significance by Asset Groups (Intraday focus).

This script can:
  - auto-build group significance CSVs by calling Helpers_for_paper_export_scripts/Section_6/asset_group_significance.py
  - and then render Table 5 (LaTeX) plus an optional grouped-bar figure.

Outputs (defaults):
  - LaTeX table: code/paper/latex/Section_6_table_5_asset_groups.tex
  - Optional figure: code/paper/figures/Section_6_fig_7_asset_groups.pdf

Default behavior (auto-build ON):
  - Reads per-asset predictions from --pred-dir
  - Baseline HAR vs models PM,CP (configurable)
  - Uses your group mapping (embedded) unless --groups-file is provided
  - Writes intermediate CSVs to --tables-dir:
      Section_6_groups_long.csv (per-asset rows)
      Section_6_groups_pivot.csv (Fisher p per group x model)
      Section_6_groups_pivot_long.csv (group-level aggregates)
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Your preferred grouping (every ticker exactly once) ----------
# If you provide --groups-file, that will override this mapping.
USER_GROUPS = {
    # US Mega Tech
    "AAPL":"US Mega Tech","MSFT":"US Mega Tech","GOOGL":"US Mega Tech","AMZN":"US Mega Tech","META":"US Mega Tech",
    # Semiconductors
    "NVDA":"Semiconductors","SMH":"Semiconductors","SOXL":"Semiconductors",
    # Broad US Equity Index
    "SPY":"Broad US Equity","QQQ":"Broad US Equity","IWM":"Broad US Equity","RSP":"Broad US Equity",
    # Leveraged Index
    "TQQQ":"Leveraged Index","SPXL":"Leveraged Index",
    # US Sector ETFs
    "XLK":"US Sector ETFs","XLF":"US Sector ETFs","XLE":"US Sector ETFs","XLU":"US Sector ETFs","XBI":"US Sector ETFs",
    # US Single-name (non-mega growth)
    "TSLA":"US Single-name","PLTR":"US Single-name","JPM":"US Single-name","JNJ":"US Single-name","BTI":"US Single-name",
    # International Equity
    "EEM":"International Equity","EFA":"International Equity","INDA":"International Equity","EWW":"International Equity",
    "EWZ":"International Equity","FXI":"International Equity","KWEB":"International Equity","BABA":"International Equity",
    # Commodities/Metals
    "GLD":"Commodities/Metals","SLV":"Commodities/Metals","DBB":"Commodities/Metals",
    # Energy/Oil
    "USO":"Energy/Oil",
    # Fixed Income
    "TLT":"Fixed Income","HYG":"Fixed Income",
    # Currency & Crypto proxies
    "UUP":"Currency/Crypto","FXY":"Currency/Crypto","BITO":"Currency/Crypto",
    # Thematic/Alt/VIX/Other
    "ARKK":"Thematic/Alt/VIX/Other","ARKW":"Thematic/Alt/VIX/Other","MJ":"Thematic/Alt/VIX/Other","VXX":"Thematic/Alt/VIX/Other",
}

DISPLAY_GROUP_ORDER = [
    "US Mega Tech","Semiconductors","Broad US Equity","Leveraged Index",
    "US Sector ETFs","US Single-name","International Equity",
    "Commodities/Metals","Energy/Oil","Fixed Income","Currency/Crypto",
    "Thematic/Alt/VIX/Other"
]

DISPLAY_MODEL = {"PM":"Prime Modulo (PM)", "CP":"Contiguous Prime (CP)"}


def write_temp_groups_csv(path_csv: str, mapping: dict):
    rows = [{'ticker': t, 'group': g} for t, g in sorted(mapping.items())]
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(path_csv, index=False)


def run_builder(pred_dir, tables_dir, baseline, models, groups_file, min_points, tickers):
    # helper path: .../code/Helpers_for_paper_export_scripts/Section_6/asset_group_significance.py
    code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # up to .../code
    helper = os.path.join(
        code_dir, "Helpers_for_paper_export_scripts", "Section_6", "asset_group_significance.py"
    )
    if not os.path.exists(helper):
        raise SystemExit(f"asset_group_significance.py not found at {helper}")

    out_long = os.path.join(tables_dir, 'Section_6_groups_long.csv')
    out_pivot = os.path.join(tables_dir, 'Section_6_groups_pivot.csv')

    cmd = [
        sys.executable, helper,
        '--pred-dir', pred_dir,
        '--baseline', baseline,
        '--models', models,
        '--out-long', out_long,
        '--out-pivot', out_pivot,
        '--min-points', str(min_points),
    ]
    if groups_file:
        cmd += ['--groups-file', groups_file]
    if tickers:
        cmd += ['--tickers', tickers]

    # Ensure helper sees "utils" regardless of CWD by setting PYTHONPATH=.../code
    env = os.environ.copy()
    env['PYTHONPATH'] = (env.get('PYTHONPATH', '') + (os.pathsep if env.get('PYTHONPATH') else '') + code_dir)

    print("[section_6] building group significance:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    return out_long, out_pivot, os.path.splitext(out_pivot)[0] + "_long.csv"


def load_group_long(tables_dir: str) -> pd.DataFrame:
    # Use the aggregated group-level long file produced by the helper.
    path = os.path.join(tables_dir, 'Section_6_groups_pivot_long.csv')
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run with --auto-build or create it via the helper.")
    return pd.read_csv(path)


def render_table(df_group_long: pd.DataFrame, out_tex: str):
    """
    Expect columns:
      group, baseline, model, n_assets,
      delta_smape_pct_mean, delta_smape_pct_se, winner_frac_smape, fisher_p_dm, sig_mark
    """
    keep_cols = [
        'group', 'model', 'n_assets',
        'delta_smape_pct_mean', 'delta_smape_pct_se',
        'winner_frac_smape', 'fisher_p_dm', 'sig_mark'
    ]
    df = df_group_long[keep_cols].copy()
    df = df[df['model'].isin(['PM', 'CP'])].copy()

    df['group'] = df['group'].astype(str)
    df['group_order'] = df['group'].apply(lambda g: DISPLAY_GROUP_ORDER.index(g) if g in DISPLAY_GROUP_ORDER else 999)
    df = df.sort_values(['group_order', 'model'])

    def fmt_mean_se(mn, se):
        if pd.isna(mn):
            return '--'
        try:
            mn_f = float(mn)
        except Exception:
            return '--'
        if pd.isna(se):
            return f"{mn_f:.2f}"
        try:
            se_f = float(se)
            return f"{mn_f:.2f} $\\pm$ {se_f:.2f}"
        except Exception:
            return f"{mn_f:.2f}"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Table 5: Group-level significance (intraday). $\Delta$SMAPE is HAR $-$ Model (positive indicates improvement). Wins are the fraction of assets in each group with $p<0.05$ (DM test on MAE).}")
    lines.append(r"\label{tab:groups}")
    lines.append(r"\begin{tabular}{l r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Asset Group & $\Delta$SMAPE (PM) & Wins (PM) & Fisher $p$ (PM) & $\Delta$SMAPE (CP) & Wins (CP) \\")
    lines.append(r"\midrule")

    for g in DISPLAY_GROUP_ORDER:
        sub = df[df['group'] == g]
        if sub.empty:
            continue
        pm = sub[sub['model'] == 'PM']
        cp = sub[sub['model'] == 'CP']

        def pick(rowframe):
            if rowframe.empty:
                return ("--", "--", "--")
            r = rowframe.iloc[0]
            d = fmt_mean_se(r.get('delta_smape_pct_mean'), r.get('delta_smape_pct_se'))
            sm = r.get('sig_mark', '')
            if isinstance(sm, str) and sm.strip():
                d = d + sm.strip()
            w = r.get('winner_frac_smape')
            wins = '--' if pd.isna(w) else f"{int(round(100 * float(w)))}\\%"
            p = r.get('fisher_p_dm')
            fisher = '--' if pd.isna(p) else f"{float(p):.2e}"
            return (d, wins, fisher)

        d_pm, w_pm, f_pm = pick(pm)
        d_cp, w_cp, f_cp = pick(cp)

        lines.append(f"{g} & {d_pm} & {w_pm} & {f_pm} & {d_cp} & {w_cp} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\footnotesize{Wins are the share of assets in group with $p<0.05$ (DM vs HAR). Fisher combines per-asset $p$-values.}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"[ok] wrote {out_tex}")


def render_figure(df_group_long: pd.DataFrame, out_fig: str):
    # Grouped bar plot of ?SMAPE by group for PM and CP
    df = df_group_long[df_group_long['model'].isin(['PM', 'CP'])].copy()
    df['group'] = df['group'].astype(str)
    df['group_order'] = df['group'].apply(lambda g: DISPLAY_GROUP_ORDER.index(g) if g in DISPLAY_GROUP_ORDER else 999)
    df = df.sort_values(['group_order', 'model'])

    groups = [g for g in DISPLAY_GROUP_ORDER if g in set(df['group'])]
    idx = np.arange(len(groups))
    width = 0.38

    pm_means, pm_se = [], []
    cp_means, cp_se = [], []

    for g in groups:
        sub = df[df['group'] == g]
        pm = sub[sub['model'] == 'PM']
        cp = sub[sub['model'] == 'CP']

        def to_float_safe(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        pm_means.append(to_float_safe(pm['delta_smape_pct_mean'].iloc[0]) if not pm.empty else np.nan)
        pm_se.append(to_float_safe(pm['delta_smape_pct_se'].iloc[0]) if not pm.empty else np.nan)
        cp_means.append(to_float_safe(cp['delta_smape_pct_mean'].iloc[0]) if not cp.empty else np.nan)
        cp_se.append(to_float_safe(cp['delta_smape_pct_se'].iloc[0]) if not cp.empty else np.nan)

    pretty_labels = {
        'US Mega Tech': 'US Mega\nTech',
        'Broad US Equity': 'Broad US\nEquity',
        'International Equity': 'International\nEquity',
        'Commodities/Metals': 'Commodities/\nMetals',
        'Fixed Income': 'Fixed\nIncome',
    }
    group_labels = [pretty_labels.get(g, g) for g in groups]

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
    })
    fig, ax = plt.subplots(figsize=(7.4, 3.35))
    ax.bar(idx - width/2, pm_means, width, yerr=pm_se, capsize=3, label='PM',
           color='#3f3f3f', alpha=0.95)
    ax.bar(idx + width/2, cp_means, width, yerr=cp_se, capsize=3, label='CP',
           color='#8c8c8c', alpha=0.95)

    ax.set_xticks(idx)
    ax.set_xticklabels(group_labels, rotation=0, ha='center')
    ax.set_ylabel(r'$\Delta$SMAPE relative to HAR-RV (pp)')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper right', ncol=2, handlelength=1.4)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    pdf = out_fig if out_fig.endswith('.pdf') else os.path.splitext(out_fig)[0] + '.pdf'
    svg = os.path.splitext(pdf)[0] + '.svg'
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] wrote {pdf}\n[ok] wrote {svg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', type=str, default='run_results/current_intraday/predictions')
    ap.add_argument('--tables-dir', type=str, default='run_results/current_intraday/tables')
    ap.add_argument('--out-tex', type=str, default='code/paper/latex/Section_6_table_5_asset_groups.tex')
    ap.add_argument('--out-fig', type=str, default='code/paper/figures/Section_6_fig_7_asset_groups.pdf', help='Grouped-bar figure path (.pdf)')
    ap.add_argument('--baseline', type=str, default='HAR')
    ap.add_argument('--models', type=str, default='PM,CP')
    ap.add_argument('--groups-file', type=str, default='', help='Optional CSV overriding grouping (ticker,group)')
    ap.add_argument('--tickers', type=str, default='', help='Optional comma list to restrict tickers')
    ap.add_argument('--min-points', type=int, default=200)
    ap.add_argument('--auto-build', dest='auto_build', action='store_true')
    ap.add_argument('--no-auto-build', dest='auto_build', action='store_false')
    ap.set_defaults(auto_build=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    os.makedirs(args.tables_dir, exist_ok=True)

    # If no groups-file supplied, write a temp one from USER_GROUPS to ensure your preferred mapping.
    groups_file = args.groups_file
    if not groups_file:
        groups_file = os.path.join(args.tables_dir, 'Section_6_groups_mapping.csv')
        rows = [{'ticker': t, 'group': g} for t, g in sorted(USER_GROUPS.items())]
        pd.DataFrame(rows).to_csv(groups_file, index=False)

    if args.auto_build:
        _out_long, _out_pivot, _out_pivot_long = run_builder(
            pred_dir=args.pred_dir,
            tables_dir=args.tables_dir,
            baseline=args.baseline,
            models=args.models,
            groups_file=groups_file,
            min_points=args.min_points,
            tickers=args.tickers
        )

    # Load group-level long table (aggregates) and render outputs
    grp_long = load_group_long(args.tables_dir)
    render_table(grp_long, args.out_tex)
    if args.out_fig:
        render_figure(grp_long, args.out_fig)


if __name__ == '__main__':
    main()
