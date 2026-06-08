#!/usr/bin/env python3
"""Combine per-ticker PM capacity sweeps into the Section 5 Figure 4 input."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", required=True)
    parser.add_argument(
        "--out",
        default="Section_5_fig_4_capacity_curve.csv",
        help="Output CSV name or path. Relative paths are written under --tables-dir.",
    )
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = tables_dir / out_path

    files = sorted(tables_dir.glob("Section_5_fig_4_capacity_curve_*.csv"))
    rows = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        ticker = path.stem.replace("Section_5_fig_4_capacity_curve_", "")
        df["ticker"] = ticker
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No per-ticker capacity files found under {tables_dir}")

    combined = pd.concat(rows, ignore_index=True)
    required = {"family", "num_primes", "smape_pct_mean"}
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"Capacity files missing required columns: {sorted(missing)}")

    out_rows = []
    for (family, num_primes), sub in combined.groupby(["family", "num_primes"], sort=True):
        vals = pd.to_numeric(sub["smape_pct_mean"], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            mean = np.nan
            se = np.nan
        else:
            mean = float(vals.mean())
            se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan
        out_rows.append(
            {
                "family": family,
                "num_primes": int(num_primes),
                "smape_pct_mean": mean,
                "smape_pct_se": se,
            }
        )

    pd.DataFrame(out_rows).sort_values(["family", "num_primes"]).to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
