#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 5 — CP vs CJ variant (Fig. 6)

Reads feeder (default):
  code/outputs/intraday/tables/Section_5_table_5b_cp_ablations.csv

Expected columns (preferred):
  n_assets,
  delta_smape_pct_mean, delta_smape_pct_se, fisher_p_dm_smape, sig_mark_smape,
  delta_mae_mean,       delta_mae_se,       fisher_p_dm_mae,   sig_mark_mae,
  delta_rmse_mean,      delta_rmse_se,      fisher_p_dm_rmse,  sig_mark_rmse,
  delta_diracc_pct_mean,delta_diracc_pct_se,fisher_p_diracc,   sig_mark_diracc

Also tolerated (builder variants):
  fisher_p_dm, sig_mark  (generic), and any numeric fields that might be
  serialized as strings such as "(np.float64(-7.98e-15), nan)".

Writes:
  code/paper/figures/Section_5_fig_6_cp_cj.pdf
  code/paper/figures/Section_5_fig_6_cp_cj.svg
"""
import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ---------- helpers ----------

_float_pat = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

def _coerce_float(val, prefer_second_if_tuple: bool = False) -> float:
    """
    Turn 'val' into a float robustly.
    Handles numbers, NaNs, and strings like "(np.float64(-7.98e-15), nan)".
    If prefer_second_if_tuple=True and two numbers are present, return the 2nd.
    """
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return np.nan
    if isinstance(val, (int, float, np.floating)):
        return float(val)

    s = str(val).strip()
    if not s:
        return np.nan
    # quick nan checks
    low = s.lower()
    if low in {"nan", "none"}:
        return np.nan

    nums = _float_pat.findall(s)
    if not nums:
        return np.nan

    try:
        if prefer_second_if_tuple and len(nums) >= 2:
            return float(nums[1])
        return float(nums[0])
    except Exception:
        return np.nan


def _fix_sig_mark(s: str) -> str:
    """Normalize mojibake to † ‡ ★."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return (
        s.replace("â˜…", "★")
         .replace("â€¡", "‡")
         .replace("â€ ", "†")
         .replace("â€\xA0", "†")
         .replace("*", "★")  # just in case
    )


def _auto_ylim(vals, ses, min_span=0.005, pad=0.0005):
    a = np.asarray(vals, float)
    e = np.asarray(ses, float)
    a[~np.isfinite(a)] = 0.0
    e[~np.isfinite(e)] = 0.0
    span = max(float(np.max(np.abs(a)) + 2 * np.max(e)), min_span)
    span += pad
    return (-span, span)


def _get_row_value(row, key, prefer_second_if_tuple=False, fallbacks=()):
    if key in row and pd.notna(row[key]):
        return _coerce_float(row[key], prefer_second_if_tuple=prefer_second_if_tuple)
    for fb in fallbacks:
        if fb in row and pd.notna(row[fb]):
            return _coerce_float(row[fb], prefer_second_if_tuple=prefer_second_if_tuple)
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="code/outputs/intraday/tables")
    ap.add_argument("--in-csv", default="", help="Optional explicit CSV path")
    ap.add_argument("--outdir", default="code/paper/figures")
    ap.add_argument("--prefix", default="Section_5_fig_6_cp_cj")
    ap.add_argument("--style", choices=["mono", "color"], default="mono")
    ap.add_argument("--metrics", default="smape,mae,rmse,diracc",
                    help="Comma list among smape,mae,rmse,diracc")
    args = ap.parse_args()

    in_csv = args.in_csv or os.path.join(
        args.tables_dir, "Section_5_table_5b_cp_ablations.csv"
    )
    df = pd.read_csv(in_csv, encoding="utf-8")
    if df.empty:
        raise SystemExit(f"No rows in feeder CSV: {in_csv}")

    # use single summary row (aggregated across assets)
    row = df.iloc[0].to_dict()

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    valid = {"smape", "mae", "rmse", "diracc"}
    metrics = [m for m in metrics if m in valid] or ["smape"]

    # style
    if args.style == "mono":
        bar_kwargs = dict(color="#666666", edgecolor="#333333")
        err_kwargs = dict(ecolor="#222222", capsize=3, lw=1)
    else:
        bar_kwargs = dict(color="#6baed6", edgecolor="#08519c")
        err_kwargs = dict(ecolor="#08519c", capsize=3, lw=1)

    labels, vals, ses, sigs, units = [], [], [], [], []

    for m in metrics:
        if m == "smape":
            v = _get_row_value(
                row, "delta_smape_pct_mean",
                prefer_second_if_tuple=False,
                fallbacks=("delta_smape_mean", "delta_smape", "delta_smape_pp")
            )
            se = _get_row_value(
                row, "delta_smape_pct_se",
                prefer_second_if_tuple=True,   # if a "(mean, se)" tuple landed here
                fallbacks=("delta_smape_se",)
            )
            sig = row.get("sig_mark_smape", row.get("sig_mark", ""))
            labels.append("Δ SMAPE vs CP (pp)")
            units.append("pp")
        elif m == "mae":
            v = _get_row_value(row, "delta_mae_mean", fallbacks=("delta_mae",))
            se = _get_row_value(row, "delta_mae_se", prefer_second_if_tuple=True)
            sig = row.get("sig_mark_mae", row.get("sig_mark", ""))
            labels.append("Δ MAE vs CP")
            units.append("abs")
        elif m == "rmse":
            v = _get_row_value(row, "delta_rmse_mean", fallbacks=("delta_rmse",))
            se = _get_row_value(row, "delta_rmse_se", prefer_second_if_tuple=True)
            sig = row.get("sig_mark_rmse", row.get("sig_mark", ""))
            labels.append("Δ RMSE vs CP")
            units.append("abs")
        else:  # diracc
            v = _get_row_value(
                row, "delta_diracc_pct_mean",
                fallbacks=("delta_diracc_mean", "delta_diracc_pp")
            )
            se = _get_row_value(
                row, "delta_diracc_pct_se",
                prefer_second_if_tuple=True,
                fallbacks=("delta_diracc_se",)
            )
            sig = row.get("sig_mark_diracc", row.get("sig_mark", ""))
            labels.append("Δ DirAcc vs CP (pp)")
            units.append("pp")

        vals.append(float(v))
        ses.append(float(se) if np.isfinite(se) else np.nan)
        sigs.append(_fix_sig_mark(sig))

    k = len(metrics)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 3.6), constrained_layout=True)
    if k == 1:
        axes = [axes]

    for ax, v, se, lab, unit, sig in zip(axes, vals, ses, labels, units, sigs):
        ax.bar([0], [v], yerr=[se] if np.isfinite(se) else None, **bar_kwargs, **err_kwargs)
        ax.axhline(0.0, color="#333", lw=0.8)
        lo, hi = _auto_ylim([v], [se], min_span=0.005 if unit == "pp" else 1e-7, pad=0.0)
        ax.set_ylim(lo, hi)
        ax.set_xticks([0], ["CP-CJ"])
        ax.set_ylabel(lab)
        ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # value label
        if unit == "pp":
            txt = f"{v:+.2f}"
            if abs(v) < 0.0045:
                txt = "≈ 0.00"
            yoff = 0.0009 if v >= 0 else -0.0009
        else:
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 2))
            txt = f"{v:.3g}"
            yoff = 0.0

        if sig:
            txt += f" {sig}"

        ax.text(0, v + yoff, txt, ha="center", va=("bottom" if v >= 0 else "top"), fontsize=10)

    os.makedirs(args.outdir, exist_ok=True)
    pdf = os.path.join(args.outdir, f"{args.prefix}.pdf")
    svg = os.path.join(args.outdir, f"{args.prefix}.svg")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(f"[ok] wrote {pdf}\n[ok] wrote {svg}")


if __name__ == "__main__":
    main()
