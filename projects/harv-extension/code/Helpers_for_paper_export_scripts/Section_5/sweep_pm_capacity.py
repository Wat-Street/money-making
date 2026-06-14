#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep capacity for Prime Modulo (PM): #primes → SMAPE (mean ± s.e.)
Writes: run_results/current_intraday/tables/Section_5_fig_4_capacity_curve.csv with columns:
  num_primes, smape_pct_mean, smape_pct_se, family

Assumptions:
- Local clean intraday files exist as <local_dir>/<TICKER>_5m.csv
- We reuse your existing utilities:
    utils.data_utils.handleIntraday
    utils.data_utils.calculate_intraday_realized_volatility
    utils.data_utils.fit_and_predict_extended
    utils.models_utils.add_prime_modulo_terms (creates PM_k<p>_r<r> columns)
- We DO NOT modify models_utils; we generate the full PM feature set once
  per asset, then subset the first K prime-columns (sorted by p) for each K.

Usage example:
  python code/Helpers_for_paper_export_scripts/Section_5/sweep_pm_capacity.py ^
    --assets SPY ^
    --k-grid 3,4,5,6,7,8 ^
    --n 390 ^
    --warmup 200 ^
    --local-dir data/market_data/clean ^
    --out-csv run_results/current_intraday/tables/Section_5_fig_4_capacity_curve.csv
"""

import os
import sys
import argparse
from typing import List

import numpy as np
import pandas as pd

# --- Make 'code/' importable so "from utils import ..." works ---
# This file lives at: code/Helpers_for_paper_export_scripts/Section_5/sweep_pm_capacity.py
# Going up two levels lands in the 'code' directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from utils.data_utils import (  # noqa: E402
    handleIntraday,
    calculate_intraday_realized_volatility,
    fit_and_predict_extended,
)
from utils.models_utils import add_prime_modulo_terms  # noqa: E402

EPS = 1e-12


def _normalize_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Return the actual column name in df matching any of the candidate names
    (case-insensitive, ignoring spaces and underscores). Raises KeyError if none.
    """
    norm = {c: c for c in df.columns}
    key_map = {}
    for c in df.columns:
        k = c.lower().replace(" ", "").replace("_", "")
        key_map[k] = c
    for cand in candidates:
        k = cand.lower().replace(" ", "").replace("_", "")
        if k in key_map:
            return key_map[k]
        # also try plural/singular accidentally
        if k.endswith("s") and k[:-1] in key_map:
            return key_map[k[:-1]]
        if (k + "s") in key_map:
            return key_map[k + "s"]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def _read_clean_5m(local_dir: str, ticker: str) -> pd.DataFrame:
    path = os.path.join(local_dir, f"{ticker}_5m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path)

    # locate datetime-like column
    try:
        dt_col = _normalize_col(df, ["Datetime", "Date", "timestamp", "time", "Time"])
    except KeyError:
        # as a last resort, try the first column if it looks like a time axis
        first = df.columns[0]
        try:
            pd.to_datetime(df[first], utc=True)
            dt_col = first
        except Exception as _:
            raise ValueError("Could not find a datetime column in the CSV.")

    # parse with utc=True to avoid mixed-timezone warning/future error
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()

    # map price/volume to canonical names
    try:
        close_col = _normalize_col(df, ["Close", "Adj Close", "adj_close", "adjclose", "close", "Price", "Last"])
    except KeyError as e:
        raise ValueError(f"CSV must include a Close-like column (Close/Adj Close/etc.): {path}") from e

    try:
        vol_col = _normalize_col(df, ["Volume", "volume", "Vol", "totalvolume"])
    except KeyError as e:
        raise ValueError(f"CSV must include a Volume-like column: {path}") from e

    # rename to canonical
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})
    if vol_col != "Volume":
        df = df.rename(columns={vol_col: "Volume"})

    # ensure numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    # keep only necessary columns plus any others (OHLCV) for downstream utils if needed
    return df


def _smape_series(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(EPS, np.abs(y_true) + np.abs(y_pred))
    return 200.0 * np.abs(y_true - y_pred) / denom


def _prime_cols(df: pd.DataFrame) -> List[str]:
    """Detect PM_k<p>_r<r> columns and return them sorted by prime p ascending."""
    mods = []
    for c in df.columns:
        if c.startswith("PM_k"):
            try:
                # PM_k{prime}_r{remainder}
                p_part = c.split("_")[1]  # k{prime}
                p = int(p_part.replace("k", ""))
                mods.append((p, c))
            except Exception:
                continue
    mods.sort(key=lambda x: x[0])
    return [c for _, c in mods]


def _fit_smape_for_K(
    extended: pd.DataFrame,
    base_feats: List[str],
    prime_cols: List[str],
    K: int,
    n: int,
    warmup: int,
) -> float:
    cols = base_feats + prime_cols[:K]
    preds = fit_and_predict_extended(extended, cols, n=n, warmup=warmup)
    if preds is None or preds.empty:
        return np.nan
    return float(_smape_series(preds["Actual"].values, preds["Predicted"].values).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, help="Comma-separated tickers, e.g. SPY")
    ap.add_argument("--k-grid", required=True, help="Comma-separated K list, e.g. 3,4,5,6,7,8")
    ap.add_argument("--n", type=int, required=True, help="Time horizon n used in feature builders")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    tickers = [t.strip() for t in args.assets.split(",") if t.strip()]
    k_list = [int(k) for k in args.k_grid.split(",") if k.strip()]

    per_k_values = {K: [] for K in k_list}

    for t in tickers:
        # load + build RV features
        raw = _read_clean_5m(args.local_dir, t)
        intr = handleIntraday(raw)  # adds Log_Return, Squared_Return, Volume
        vol = calculate_intraday_realized_volatility(intr)  # RV_d, RV_w, RV_m, Volume

        # add full PM feature set (generates PM_k<p>_r<r> columns)
        pm_full = add_prime_modulo_terms(vol.copy(), args.n)

        # collect primes and base features
        prime_cols = _prime_cols(pm_full)
        if not prime_cols:
            raise RuntimeError(
                "No PM_k<p>_r<r> columns were generated. Check add_prime_modulo_terms and 'n'."
            )

        base_feats = [c for c in ["RV_d", "RV_w", "RV_m"] if c in pm_full.columns]
        if not base_feats:
            base_feats = ["RV_d"]  # fallback

        # evaluate each K
        for K in k_list:
            val = _fit_smape_for_K(pm_full, base_feats, prime_cols, K, n=args.n, warmup=args.warmup)
            per_k_values[K].append(val)

    # aggregate across assets
    rows = []
    for K in k_list:
        arr = np.asarray([x for x in per_k_values[K] if np.isfinite(x)], float)
        if arr.size == 0:
            mean, se = np.nan, np.nan
        else:
            mean = float(arr.mean())
            se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else np.nan
        rows.append({"num_primes": K, "smape_pct_mean": mean, "smape_pct_se": se, "family": "PM"})

    out_dir = os.path.dirname(args.out_csv)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).sort_values("num_primes").to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
