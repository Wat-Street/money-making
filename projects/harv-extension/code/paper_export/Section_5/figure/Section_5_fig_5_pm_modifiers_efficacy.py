#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Section 5 — PM modifiers efficacy (Fig. 5)
Reads feeder:
  run_results/current_intraday/tables/Section_5_table_5a_pm_ablations.csv
Columns expected:
  model, n_assets, delta_smape_pct_mean, delta_smape_pct_se, winner_frac_smape, fisher_p_dm, sig_mark

Writes:
  code/paper/figures/Section_5_fig_5_pm_modifiers.pdf
  code/paper/figures/Section_5_fig_5_pm_modifiers.svg
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _fix_sig_mark(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if s in {"?", "Ã˜", "Ø"}:
        return ""
    # common mojibake normalizations
    s = s.replace("â˜…", "★").replace("â€¡", "‡").replace("â€ ", "†").replace("â€\xA0", "†")
    return s

def _autorange_y(deltas: np.ndarray, ses: np.ndarray, pad=0.02):
    # symmetric around 0; ensure at least a small range
    m = float(np.nanmax(np.abs(deltas) + np.nan_to_num(2*ses)))
    m = max(m, 0.02)  # at least ±0.02 pp
    return (-m - pad, m + pad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="run_results/current_intraday/tables")
    ap.add_argument("--in-csv", default="", help="Optional explicit CSV path")
    ap.add_argument("--outdir", default="code/paper/figures")
    ap.add_argument("--prefix", default="Section_5_fig_5_pm_modifiers")
    ap.add_argument("--style", choices=["mono","color"], default="mono")
    ap.add_argument("--ylabel", default="Δ SMAPE vs PM core (pp)")
    args = ap.parse_args()

    in_csv = args.in_csv or os.path.join(args.tables_dir, "Section_5_table_5a_pm_ablations.csv")
    df = pd.read_csv(in_csv, encoding="utf-8")
    # normalize names for nice display
    name_map = {"PM_AD": "PM-AD", "PM_VW": "PM-VW"}
    df["name"] = df["model"].map(name_map).fillna(df["model"])
    df["sig"] = df["sig_mark"].map(_fix_sig_mark)

    order = [m for m in ["PM_AD", "PM_VW"] if m in set(df["model"])]
    if not order:
        raise SystemExit(f"No supported PM modifier rows found in feeder CSV: {in_csv}")
    df = df.set_index("model").loc[order].reset_index()

    vals = df["delta_smape_pct_mean"].astype(float).values
    ses  = df["delta_smape_pct_se"].astype(float).values
    labels = df["name"].tolist()
    sigs   = df["sig"].tolist()

    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    x = np.arange(len(labels))

    if args.style == "mono":
        bar_kwargs = dict(color="#666666", edgecolor="#333333")
        err_kwargs = dict(ecolor="#222222", capsize=3, lw=1)
    else:
        bar_kwargs = dict(color="#7aa0c4", edgecolor="#2b4c6f")
        err_kwargs = dict(ecolor="#2b4c6f", capsize=3, lw=1)

    ax.bar(x, vals, yerr=ses, **bar_kwargs, **err_kwargs)

    # annotate numbers & significance
    for i, (v, s) in enumerate(zip(vals, sigs)):
        txt = f"{v:+.2f}"
        if s:
            txt += f" {s}"
        va = "bottom" if v >= 0 else "top"
        offset = 0.004 if v >= 0 else -0.004
        ax.text(i, v + offset, txt, ha="center", va=va, fontsize=10)

    ax.set_xticks(x, labels)
    ax.set_ylabel(args.ylabel)
    ax.set_xlabel("")
    ax.axhline(0.0, color="#333333", lw=0.8)
    lo, hi = _autorange_y(vals, ses)
    ax.set_ylim(lo, hi)
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(args.outdir, exist_ok=True)
    pdf = os.path.join(args.outdir, f"{args.prefix}.pdf")
    svg = os.path.join(args.outdir, f"{args.prefix}.svg")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(f"[ok] wrote {pdf}\n[ok] wrote {svg}")

if __name__ == "__main__":
    main()
