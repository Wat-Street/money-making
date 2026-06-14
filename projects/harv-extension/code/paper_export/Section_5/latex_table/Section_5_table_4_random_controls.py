#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Section 5 â€” Table 4 (Randomized control).
Reads the aggregated random-control CSV and emits a Booktabs-style LaTeX table.

Input  (default):
  run_results/current_intraday/tables/Section_5_table_4_random_controls.csv

Output (default):
  code/paper/latex/Section_5_table_4_random_controls.tex

CSV schema expected:
  model,n_assets,delta_smape_pct_mean,delta_smape_pct_se,
  delta_mae_mean,delta_mae_se,delta_rmse_mean,delta_rmse_se,
  winner_frac_smape,fisher_p_dm,sig_mark
Where deltas are (HAR âˆ’ Model). Positive = our model improves over HAR.
"""

import argparse
import os
import math
import pandas as pd

DEFAULT_IN = os.path.join(
    "run_results", "current_intraday", "tables", "Section_5_table_4_random_controls.csv"
)
DEFAULT_OUTDIR = os.path.join("run_results", "current_intraday", "latex")

MODEL_NAME = {
    "RAND": "Random (control)",
    "CRS": "Contiguous Random (control)",
}

def fmt_pm(mean, se, digits=2, pct=False):
    """Format mean Â± s.e. Optionally as percentage with requested digits."""
    if pd.isna(mean):
        return "--"
    if pct:
        m = f"{mean:.{digits}f}"
        s = f"{se:.{digits}f}" if se is not None and not pd.isna(se) else "0"
        return f"{m} $\\pm$ {s}"
    # scientific for tiny numbers; otherwise fixed
    if abs(mean) < 1e-4 or abs(mean) >= 1e3:
        m = f"{mean:.2e}"
        s = f"{se:.2e}" if se is not None and not pd.isna(se) else "0.00e+00"
    else:
        m = f"{mean:.{digits+4}f}".rstrip("0").rstrip(".")
        s = f"{se:.{digits+4}f}".rstrip("0").rstrip(".") if se is not None and not pd.isna(se) else "0"
    return f"{m} $\\pm$ {s}"


def sci_to_tex(x, digits=2):
    """Format a p-value (or any float) to LaTeX sci notation a Ã— 10^{b}."""
    if pd.isna(x):
        return "--"
    if x == 0:
        return "$0$"
    mant, exp = f"{x:.{digits}e}".split("e")
    exp = int(exp)
    # strip trailing zeros from mantissa
    mant = mant.rstrip("0").rstrip(".")
    return f"${mant} \\times 10^{{{exp}}}$"


def escape(s):
    if s is None:
        return ""
    return str(s).replace("_", "\\_")


def build_table(df: pd.DataFrame, scope: str) -> str:
    # Compute wins = round(winner_frac_smape * n_assets)
    wins_col = []
    for _, r in df.iterrows():
        n = int(r.get("n_assets", 0) or 0)
        frac = float(r.get("winner_frac_smape", 0) or 0.0)
        wins = int(round(frac * n))
        wins_col.append(f"{wins}/{n}")
    df = df.copy()
    df["wins_str"] = wins_col

    # Determine bolding for best Î”SMAPE (highest mean improvement)
    # Positive Î” means improvement (HAR âˆ’ Model).
    if "delta_smape_pct_mean" in df.columns:
        idx_best = df["delta_smape_pct_mean"].astype(float).idxmax()
    else:
        idx_best = None

    header = (
        "% Auto-generated: Section 5 â€” Table 4 (Randomized control)\n"
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\caption{Section 5, Table 4 ("
        + escape(scope.capitalize())
        + "). Randomized feature sets as a negative control. "
          "Deltas are reported as HAR$-$Model, so positive values indicate improvement over HAR.}\n"
        "  \\label{tab:section5_random_control}\n"
        "  \\begin{tabular}{lcccccc}\n"
        "    \\toprule\n"
        "    \\textbf{Model} & \\textbf{$\\Delta$SMAPE (\\%)} & \\textbf{$\\Delta$MAE} & \\textbf{$\\Delta$RMSE} & \\textbf{Wins $p{<}.05$} & \\textbf{Fisher $p$} & \\textbf{Sig.} \\\\\n"
        "    \\midrule\n"
    )

    rows = []
    for i, r in df.iterrows():
        model_code = r.get("model", "RAND")
        model = escape(MODEL_NAME.get(model_code, model_code))
        ds_mean = float(r.get("delta_smape_pct_mean", float("nan")))
        ds_se = float(r.get("delta_smape_pct_se", float("nan")))
        dmae_mean = float(r.get("delta_mae_mean", float("nan")))
        dmae_se = float(r.get("delta_mae_se", float("nan")))
        drmse_mean = float(r.get("delta_rmse_mean", float("nan")))
        drmse_se = float(r.get("delta_rmse_se", float("nan")))
        wins = r.get("wins_str", "--")
        fisher = r.get("fisher_p_dm", float("nan"))
        sig = escape(r.get("sig_mark", ""))

        # format cells
        smape_cell = fmt_pm(ds_mean, ds_se, digits=2, pct=True)
        if i == idx_best:
            smape_cell = f"\\textbf{{{smape_cell}}}"

        mae_cell = fmt_pm(dmae_mean, dmae_se, digits=3, pct=False)
        rmse_cell = fmt_pm(drmse_mean, drmse_se, digits=3, pct=False)
        fisher_cell = sci_to_tex(fisher, digits=2)

        row = f"    {model} & {smape_cell} & {mae_cell} & {rmse_cell} & {wins} & {fisher_cell} & {sig} \\\\"
        rows.append(row)

    footer = (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "  \\vspace{2pt}\n"
        "  {\\footnotesize Notes: $\\Delta$ columns are HAR$-$Model; positive means improvement. "
        "Wins counts assets with Diebold--Mariano $p{<}.05$ vs HAR.}\n"
        "\\end{table}\n"
    )

    return header + "\n".join(rows) + "\n" + footer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN, help="Input CSV path")
    ap.add_argument("--outdir", dest="outdir", default=DEFAULT_OUTDIR, help="Output dir for .tex")
    ap.add_argument(
        "--scope", default="intraday", choices=["intraday", "daily"], help="Scope for captioning"
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_tex = os.path.join(args.outdir, "Section_5_table_4_random_controls.tex")

    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"Input CSV not found: {args.in_path}")

    df = pd.read_csv(args.in_path)
    tex = build_table(df, scope=args.scope)

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"[OK] Wrote LaTeX to: {out_tex}")


if __name__ == "__main__":
    main()


