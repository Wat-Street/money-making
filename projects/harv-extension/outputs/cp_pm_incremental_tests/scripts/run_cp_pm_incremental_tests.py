"""
CP + PM Incremental Value Test
==============================

Purpose
-------
This script tests whether adding Prime-Modulo (PM) temporal-granularity
features improves a Contiguous-Prime (CP) local-volatility baseline.

The prior GitHub Actions version timed out because it called statsmodels OLS
at every timestamp for every asset/model. This version keeps the same expanding
window forecasting design but uses incremental normal-equation updates so the
full 10-asset CP vs CP+PM test can finish on GitHub Actions.

Two outputs matter:
1. Saved-CP diagnostic validation: fresh CP is compared with the saved
   run_results/current_intraday Predicted_CP series. This is diagnostic only.
2. Fresh apples-to-apples incremental test: fresh CP is compared with fresh
   CP+PM on identical data, timestamps, target alignment, warmup, and fit logic.

Anti-lookahead rule
-------------------
All conditional state labels are built only from lagged Actual values,
Actual.shift(k) for k >= 1.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from utils.data_utils import calculate_intraday_realized_volatility, fetch_intraday_data  # noqa: E402
from utils.models_utils import add_prime_modulo_terms, contig_prime_modulo  # noqa: E402

PRED_DIR = PROJECT_ROOT / "run_results" / "current_intraday" / "predictions"
DATA_DIR = PROJECT_ROOT / "data" / "market_data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cp_pm_incremental_tests"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
LATEX_DIR = OUTPUT_DIR / "latex"

N_LAGS = int(os.environ.get("N_LAGS", "22"))
WARMUP = int(os.environ.get("WARMUP", "600"))
BOOTSTRAP_RESAMPLES = int(os.environ.get("BOOTSTRAP_RESAMPLES", "300"))
BOOTSTRAP_SEED = int(os.environ.get("BOOTSTRAP_SEED", "42"))
SHIFT_OFFSETS = [-2, -1, 0, 1, 2]
NUMERIC_RIDGE = float(os.environ.get("NUMERIC_RIDGE", "1e-12"))
MAX_TICKERS = int(os.environ.get("MAX_TICKERS", "0"))  # 0 means all.

CP_FEATURE_PREFIXES = ("RV", "CP")
HYBRID_FEATURE_PREFIXES = ("RV", "CP", "PM")

CONDITION_ORDER = [
    "all_observations",
    "cluster_entry_loose",
    "cluster_entry_strict",
    "cluster_exit_loose",
    "cluster_exit_strict",
    "inflection_points",
    "high_within_block_dispersion",
    "recent_spike_position",
    "older_spike_position",
    "same_average_different_path",
]

for directory in [RESULTS_DIR, FIGURES_DIR, LATEX_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_date(value):
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value).isoformat()
    except Exception:
        return str(value)


def select_feature_columns(frame: pd.DataFrame, prefixes) -> list[str]:
    cols = [column for column in frame.columns if column.startswith(prefixes)]
    # Keep deterministic column order so fresh CP and hybrid runs are reproducible.
    return sorted(cols)


def load_existing_predictions() -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    if not PRED_DIR.exists():
        raise FileNotFoundError(f"Missing predictions directory: {PRED_DIR}")

    files = sorted(PRED_DIR.glob("*.csv"))
    if MAX_TICKERS > 0:
        files = files[:MAX_TICKERS]

    for csv_file in files:
        ticker = csv_file.stem
        try:
            frame = pd.read_csv(csv_file, parse_dates=["Date"])
            required = {"Date", "Actual", "Predicted_CP"}
            if not required.issubset(frame.columns):
                print(f"Skipping {ticker}: missing required columns {required - set(frame.columns)}")
                continue
            frame = frame.sort_values("Date").reset_index(drop=True)
            predictions[ticker] = frame
            print(f"Loaded existing predictions for {ticker}: {len(frame):,} rows")
        except Exception as exc:
            print(f"Could not load {csv_file}: {exc}")

    return predictions


def fast_expanding_ols_predict(
    data: pd.DataFrame,
    features: list[str],
    n: int,
    warmup: int,
    model_name: str,
) -> pd.DataFrame:
    """Fast expanding-window one-step-ahead linear forecasts.

    This matches the repository's fit_and_predict_extended alignment:
    - feature row i forecasts RV_d at i+1;
    - training at forecast index i uses rows 0..i-1 with targets shifted -1;
    - output Date is data.index[i+1].

    Instead of refitting statsmodels OLS from scratch each timestamp, it updates
    X'X and X'y incrementally. A tiny ridge is added only for numerical stability
    in solving the normal equations; CP and CP+PM use the same solver, so the
    incremental comparison remains apples-to-apples.
    """
    if data.empty or not features:
        return pd.DataFrame()

    frame = data.copy()
    x_df = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_values = x_df.to_numpy(dtype=float)
    t_obs = len(frame)
    if t_obs < n + warmup + 2:
        return pd.DataFrame()

    x_all = np.column_stack([np.ones(t_obs, dtype=float), x_values])
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)

    valid_x = np.isfinite(x_all).all(axis=1)
    valid_y = np.isfinite(y_next)
    p = x_all.shape[1]
    min_train_obs = max(5, p + 2)

    xtx = np.zeros((p, p), dtype=float)
    xty = np.zeros(p, dtype=float)
    identity = np.eye(p, dtype=float)
    train_count = 0
    rows = []

    start_time = time.perf_counter()
    for i in range(0, t_obs - 1):
        # At forecast index i, add row i-1 to the expanding training set.
        if i > 0:
            train_idx = i - 1
            if valid_x[train_idx] and valid_y[train_idx]:
                x = x_all[train_idx]
                y = y_next[train_idx]
                xtx += np.outer(x, x)
                xty += x * y
                train_count += 1

        if i < n + warmup:
            continue
        if train_count < min_train_obs:
            continue
        if not (valid_x[i] and valid_y[i]):
            continue

        try:
            beta = np.linalg.solve(xtx + NUMERIC_RIDGE * identity, xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtx, rcond=1e-10) @ xty

        pred = float(x_all[i] @ beta)
        y_true_next = float(y_next[i])
        if not (np.isfinite(pred) and np.isfinite(y_true_next)):
            continue

        err = y_true_next - pred
        abs_err = abs(err)
        denom = max(1e-12, abs(y_true_next) + abs(pred))
        smape_pct = 200.0 * abs_err / denom
        rows.append({
            "Date": frame.index[i + 1],
            "Actual": y_true_next,
            f"Predicted_{model_name}": pred,
            f"Err_{model_name}": err,
            f"AbsErr_{model_name}": abs_err,
            f"SMAPE_{model_name}_pct": smape_pct,
        })

    elapsed = time.perf_counter() - start_time
    print(
        f"{model_name}: built {len(rows):,} forecasts with {len(features)} features "
        f"and {train_count:,} final train rows in {elapsed:.1f}s"
    )
    return pd.DataFrame(rows)


def normalize_prediction_frame(preds: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if preds is None or preds.empty:
        return pd.DataFrame()
    frame = preds.copy()
    if "Date" not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "Date"})
    pred_col = f"Predicted_{model_name}"
    if pred_col not in frame.columns:
        raise ValueError(f"Could not find {pred_col} in prediction frame")
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame[["Date", "Actual", pred_col]].sort_values("Date").reset_index(drop=True)


def build_fresh_forecasts_for_ticker(ticker: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """Load data once, build CP and CP+PM features once, and forecast both models."""
    meta = {"ticker": ticker}
    started = time.perf_counter()
    try:
        raw = fetch_intraday_data(ticker, use_local=True, local_dir=str(DATA_DIR))
        vol = calculate_intraday_realized_volatility(raw)
        cp_frame = contig_prime_modulo(vol.copy(), n=N_LAGS, per_day_normalize=False)
        hybrid_frame = add_prime_modulo_terms(cp_frame.copy(), n=N_LAGS)

        cp_features = select_feature_columns(cp_frame, CP_FEATURE_PREFIXES)
        hybrid_features = select_feature_columns(hybrid_frame, HYBRID_FEATURE_PREFIXES)
        meta.update({
            "raw_rows": len(raw),
            "vol_rows": len(vol),
            "cp_features": len(cp_features),
            "hybrid_features": len(hybrid_features),
            "cp_feature_names": cp_features,
            "hybrid_feature_names": hybrid_features,
        })

        if not cp_features:
            raise RuntimeError("No CP features selected")
        if not hybrid_features:
            raise RuntimeError("No hybrid features selected")

        fresh_cp = normalize_prediction_frame(
            fast_expanding_ols_predict(cp_frame, cp_features, N_LAGS, WARMUP, "CP"),
            "CP",
        )
        fresh_hybrid = normalize_prediction_frame(
            fast_expanding_ols_predict(hybrid_frame, hybrid_features, N_LAGS, WARMUP, "CP_PLUS_PM_RAW"),
            "CP_PLUS_PM_RAW",
        )
        meta["elapsed_seconds"] = time.perf_counter() - started
        meta["status"] = "ok"
        return fresh_cp, fresh_hybrid, meta
    except Exception as exc:
        meta["elapsed_seconds"] = time.perf_counter() - started
        meta["status"] = "failed"
        meta["error"] = str(exc)
        print(f"{ticker}: failed to build fresh forecasts: {exc}")
        return None, None, meta


def validate_cp_recomputation(existing_df: pd.DataFrame, fresh_cp_df: pd.DataFrame, ticker: str) -> dict:
    if fresh_cp_df is None or fresh_cp_df.empty:
        return {
            "ticker": ticker,
            "status": "failed",
            "message": "Fresh CP forecast missing or empty",
            "n_existing": len(existing_df),
            "n_recomputed": 0,
            "n_aligned": 0,
            "passes": False,
        }

    merged = existing_df[["Date", "Predicted_CP"]].merge(
        fresh_cp_df[["Date", "Predicted_CP"]],
        on="Date",
        how="inner",
        suffixes=("_existing", "_recomputed"),
    ).sort_values("Date")

    if merged.empty:
        return {
            "ticker": ticker,
            "status": "failed",
            "message": "No aligned dates",
            "n_existing": len(existing_df),
            "n_recomputed": len(fresh_cp_df),
            "n_aligned": 0,
            "passes": False,
        }

    existing = merged["Predicted_CP_existing"].astype(float)
    recomputed = merged["Predicted_CP_recomputed"].astype(float)
    abs_diff = (existing - recomputed).abs()
    corr = existing.corr(recomputed) if len(merged) > 1 else np.nan
    passes = bool((corr is not None) and (not pd.isna(corr)) and corr > 0.995 and abs_diff.mean() < 1e-8)

    return {
        "ticker": ticker,
        "status": "passed" if passes else "failed",
        "message": "diagnostic exact-saved-CP validation only",
        "n_existing": int(len(existing_df)),
        "n_recomputed": int(len(fresh_cp_df)),
        "n_aligned": int(len(merged)),
        "mean_existing_CP": safe_float(existing.mean()),
        "mean_recomputed_CP": safe_float(recomputed.mean()),
        "std_existing_CP": safe_float(existing.std(ddof=1)),
        "std_recomputed_CP": safe_float(recomputed.std(ddof=1)),
        "mean_abs_diff": safe_float(abs_diff.mean()),
        "median_abs_diff": safe_float(abs_diff.median()),
        "max_abs_diff": safe_float(abs_diff.max()),
        "correlation": safe_float(corr),
        "first_aligned_date": safe_date(merged["Date"].min()),
        "last_aligned_date": safe_date(merged["Date"].max()),
        "passes": passes,
    }


def build_shift_diagnostics(existing_df: pd.DataFrame, fresh_cp_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rows = []
    if fresh_cp_df is None or fresh_cp_df.empty:
        return pd.DataFrame(rows)

    existing = existing_df[["Date", "Predicted_CP"]].copy()
    fresh = fresh_cp_df[["Date", "Predicted_CP"]].copy()
    for shift in SHIFT_OFFSETS:
        shifted = fresh.copy()
        shifted["Predicted_CP"] = shifted["Predicted_CP"].shift(shift)
        merged = existing.merge(shifted, on="Date", how="inner", suffixes=("_existing", "_shifted")).sort_values("Date")
        if merged.empty:
            rows.append({"ticker": ticker, "shift": shift, "n_aligned": 0})
            continue
        existing_vals = merged["Predicted_CP_existing"].astype(float)
        shifted_vals = merged["Predicted_CP_shifted"].astype(float)
        abs_diff = (existing_vals - shifted_vals).abs()
        rows.append({
            "ticker": ticker,
            "shift": shift,
            "n_aligned": int(len(merged)),
            "mean_abs_diff": safe_float(abs_diff.mean()),
            "median_abs_diff": safe_float(abs_diff.median()),
            "max_abs_diff": safe_float(abs_diff.max()),
            "correlation": safe_float(existing_vals.corr(shifted_vals)),
            "first_aligned_date": safe_date(merged["Date"].min()),
            "last_aligned_date": safe_date(merged["Date"].max()),
        })
    return pd.DataFrame(rows)


def merge_fresh_cp_and_hybrid(ticker: str, fresh_cp: pd.DataFrame, fresh_hybrid: pd.DataFrame) -> pd.DataFrame:
    merged = fresh_cp.merge(fresh_hybrid, on="Date", how="inner", suffixes=("_CP", "_HYBRID")).sort_values("Date").reset_index(drop=True)
    if merged.empty:
        return merged

    actual_diff = (merged["Actual_CP"] - merged["Actual_HYBRID"]).abs().max()
    if pd.notna(actual_diff) and actual_diff > 1e-12:
        print(f"Warning {ticker}: Actual mismatch between fresh CP and hybrid, max={actual_diff}")

    merged = merged.rename(columns={
        "Actual_CP": "Actual",
        "Predicted_CP": "Predicted_CP_FRESH",
        "Predicted_CP_PLUS_PM_RAW": "Predicted_CP_PLUS_PM_RAW",
    })
    if "Actual_HYBRID" in merged.columns:
        merged = merged.drop(columns=["Actual_HYBRID"])

    denom_cp = merged["Actual"].abs() + merged["Predicted_CP_FRESH"].abs()
    denom_hybrid = merged["Actual"].abs() + merged["Predicted_CP_PLUS_PM_RAW"].abs()
    merged["AbsErr_CP_FRESH"] = (merged["Actual"] - merged["Predicted_CP_FRESH"]).abs()
    merged["AbsErr_CP_PLUS_PM_RAW"] = (merged["Actual"] - merged["Predicted_CP_PLUS_PM_RAW"]).abs()
    merged["SMAPE_CP_FRESH_pct"] = np.where(denom_cp > 0, 200 * merged["AbsErr_CP_FRESH"] / denom_cp, np.nan)
    merged["SMAPE_CP_PLUS_PM_RAW_pct"] = np.where(denom_hybrid > 0, 200 * merged["AbsErr_CP_PLUS_PM_RAW"] / denom_hybrid, np.nan)
    merged["Hybrid_advantage_vs_CP"] = merged["SMAPE_CP_FRESH_pct"] - merged["SMAPE_CP_PLUS_PM_RAW_pct"]
    merged["AbsErr_advantage_vs_CP"] = merged["AbsErr_CP_FRESH"] - merged["AbsErr_CP_PLUS_PM_RAW"]
    merged["asset"] = ticker
    return merged


def add_lagged_actuals(frame: pd.DataFrame, max_lag: int = N_LAGS) -> pd.DataFrame:
    out = frame.sort_values("Date").reset_index(drop=True).copy()
    for lag in range(1, max_lag + 1):
        out[f"lag{lag}"] = out["Actual"].shift(lag)
    return out.dropna(subset=[f"lag{max_lag}"]).reset_index(drop=True)


def define_conditions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    conditions = {"all_observations": pd.Series(True, index=frame.index)}
    lag_cols_1_3 = [f"lag{i}" for i in range(1, 4)]
    lag_cols_1_5 = [f"lag{i}" for i in range(1, 6)]
    lag_cols_4_6 = [f"lag{i}" for i in range(4, 7)]
    lag_cols_6_10 = [f"lag{i}" for i in range(6, 11)]
    lag_cols_7_10 = [f"lag{i}" for i in range(7, 11)]
    lag_cols_1_10 = [f"lag{i}" for i in range(1, 11)]

    recent_mean_1_5 = frame[lag_cols_1_5].mean(axis=1)
    older_mean_6_10 = frame[lag_cols_6_10].mean(axis=1)
    recent_mean_1_3 = frame[lag_cols_1_3].mean(axis=1)
    prior_mean_4_6 = frame[lag_cols_4_6].mean(axis=1)
    prior_mean_7_10 = frame[lag_cols_7_10].mean(axis=1)
    local_mean_1_10 = frame[lag_cols_1_10].mean(axis=1)
    local_std_1_10 = frame[lag_cols_1_10].std(axis=1)
    lagged_median = np.nanmedian(frame[lag_cols_1_10].to_numpy().ravel())

    entry_score = recent_mean_1_5 - older_mean_6_10
    entry_loose = entry_score >= entry_score.quantile(0.80)
    conditions["cluster_entry_loose"] = entry_loose
    conditions["cluster_entry_strict"] = entry_loose & (recent_mean_1_5 >= lagged_median)

    exit_score = older_mean_6_10 - recent_mean_1_5
    exit_loose = exit_score >= exit_score.quantile(0.80)
    conditions["cluster_exit_loose"] = exit_loose
    conditions["cluster_exit_strict"] = exit_loose & (older_mean_6_10 >= lagged_median)

    slope_recent = recent_mean_1_3 - prior_mean_4_6
    slope_prior = prior_mean_4_6 - prior_mean_7_10
    movement = (slope_recent - slope_prior).abs()
    conditions["inflection_points"] = (slope_recent * slope_prior < 0) & (movement >= movement.quantile(0.50))

    dispersion = local_std_1_10 / local_mean_1_10.replace(0, np.nan)
    conditions["high_within_block_dispersion"] = dispersion >= dispersion.quantile(0.80)

    lag_matrix = frame[lag_cols_1_10].to_numpy()
    max_positions = np.nanargmax(lag_matrix, axis=1) + 1
    conditions["recent_spike_position"] = pd.Series(np.isin(max_positions, [1, 2, 3]), index=frame.index)
    conditions["older_spike_position"] = pd.Series(np.isin(max_positions, [8, 9, 10]), index=frame.index)

    path_score = (recent_mean_1_5 - older_mean_6_10).abs()
    mean_low = local_mean_1_10.quantile(0.40)
    mean_high = local_mean_1_10.quantile(0.60)
    conditions["same_average_different_path"] = local_mean_1_10.between(mean_low, mean_high) & (path_score >= path_score.quantile(0.80))

    return {name: mask.fillna(False).astype(bool) for name, mask in conditions.items()}


def bootstrap_ci(values: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1 or n_resamples <= 0:
        return float(np.mean(values)), float(np.mean(values))
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        means[idx] = np.mean(sample)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_ttest(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan
    try:
        return float(stats.ttest_1samp(values, popmean=0.0, nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def compute_asset_condition_metrics(asset: str, frame: pd.DataFrame, condition_name: str, mask: pd.Series) -> dict:
    subset = frame.loc[mask].copy()
    if subset.empty:
        return {"asset": asset, "condition": condition_name, "n_obs": 0, "share_of_asset_obs": 0.0}

    advantage = subset["Hybrid_advantage_vs_CP"].astype(float)
    abs_advantage = subset["AbsErr_advantage_vs_CP"].astype(float)
    ci_low, ci_high = bootstrap_ci(advantage.to_numpy())
    return {
        "asset": asset,
        "condition": condition_name,
        "n_obs": int(len(subset)),
        "share_of_asset_obs": float(len(subset) / len(frame)) if len(frame) else np.nan,
        "CP_mean_SMAPE": safe_float(subset["SMAPE_CP_FRESH_pct"].mean()),
        "HYBRID_mean_SMAPE": safe_float(subset["SMAPE_CP_PLUS_PM_RAW_pct"].mean()),
        "mean_Hybrid_advantage_vs_CP": safe_float(advantage.mean()),
        "median_Hybrid_advantage_vs_CP": safe_float(advantage.median()),
        "hybrid_win_rate_vs_CP": safe_float((advantage > 0).mean()),
        "mean_AbsErr_CP": safe_float(subset["AbsErr_CP_FRESH"].mean()),
        "mean_AbsErr_HYBRID": safe_float(subset["AbsErr_CP_PLUS_PM_RAW"].mean()),
        "mean_AbsErr_advantage_vs_CP": safe_float(abs_advantage.mean()),
        "ttest_pval": paired_ttest(advantage.to_numpy()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }


def compute_pooled_metrics(asset_results: pd.DataFrame, lagged_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        asset_subset = asset_results[(asset_results["condition"] == condition) & (asset_results["n_obs"] > 0)].copy()
        pooled_parts = []
        for _, frame in lagged_frames.items():
            conditions = define_conditions(frame)
            pooled_parts.append(frame.loc[conditions[condition]])
        pooled = pd.concat(pooled_parts, ignore_index=True) if pooled_parts else pd.DataFrame()

        if asset_subset.empty or pooled.empty:
            rows.append({"condition": condition, "total_n_obs": 0, "number_of_assets": 0})
            continue

        advantage = pooled["Hybrid_advantage_vs_CP"].astype(float)
        rows.append({
            "condition": condition,
            "total_n_obs": int(len(pooled)),
            "number_of_assets": int(asset_subset["asset"].nunique()),
            "equal_weight_asset_mean_advantage": safe_float(asset_subset["mean_Hybrid_advantage_vs_CP"].mean()),
            "equal_weight_asset_median_advantage": safe_float(asset_subset["median_Hybrid_advantage_vs_CP"].mean()),
            "equal_weight_asset_hybrid_win_rate": safe_float(asset_subset["hybrid_win_rate_vs_CP"].mean()),
            "pooled_CP_mean_SMAPE": safe_float(pooled["SMAPE_CP_FRESH_pct"].mean()),
            "pooled_HYBRID_mean_SMAPE": safe_float(pooled["SMAPE_CP_PLUS_PM_RAW_pct"].mean()),
            "pooled_mean_Hybrid_advantage_vs_CP": safe_float(advantage.mean()),
            "pooled_hybrid_win_rate_vs_CP": safe_float((advantage > 0).mean()),
            "number_of_assets_where_hybrid_beats_CP": int((asset_subset["mean_Hybrid_advantage_vs_CP"] > 0).sum()),
            "asset_win_rate": safe_float((asset_subset["mean_Hybrid_advantage_vs_CP"] > 0).mean()),
        })
    return pd.DataFrame(rows)


def make_plots(pooled: pd.DataFrame):
    if pooled.empty:
        return
    ordered = pooled.copy()
    ordered["condition"] = pd.Categorical(ordered["condition"], categories=CONDITION_ORDER, ordered=True)
    ordered = ordered.sort_values("condition")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ordered["condition"], ordered["equal_weight_asset_mean_advantage"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Fresh CP+PM Advantage over Fresh CP by Condition")
    ax.set_ylabel("CP SMAPE - Hybrid SMAPE (positive = hybrid better)")
    ax.set_xlabel("Condition")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fresh_hybrid_advantage_by_condition.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ordered["condition"], ordered["equal_weight_asset_hybrid_win_rate"])
    ax.axhline(0.5, linewidth=1)
    ax.set_title("Fresh CP+PM Win Rate over Fresh CP by Condition")
    ax.set_ylabel("Hybrid win rate")
    ax.set_xlabel("Condition")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fresh_hybrid_win_rate_by_condition.png", dpi=200)
    plt.close(fig)


def write_latex_table(pooled: pd.DataFrame):
    if pooled.empty:
        (LATEX_DIR / "fresh_conditional_hybrid_summary_pooled.tex").write_text("% empty\n", encoding="utf-8")
        return
    cols = ["condition", "total_n_obs", "equal_weight_asset_mean_advantage", "equal_weight_asset_hybrid_win_rate", "asset_win_rate"]
    table = pooled[cols].copy().rename(columns={
        "condition": "Condition",
        "total_n_obs": "N",
        "equal_weight_asset_mean_advantage": "Adv.",
        "equal_weight_asset_hybrid_win_rate": "Win Rate",
        "asset_win_rate": "Asset Win Rate",
    })
    (LATEX_DIR / "fresh_conditional_hybrid_summary_pooled.tex").write_text(table.to_latex(index=False, float_format="%.4f"), encoding="utf-8")


def dataframe_to_text(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame is None or frame.empty:
        return "[empty]"
    return frame.head(max_rows).to_string(index=False)


def write_readme(validation: pd.DataFrame, shift_diag: pd.DataFrame, pooled: pd.DataFrame, mode: str, tickers: list[str], meta: list[dict]):
    validation_passes = int(validation["passes"].sum()) if not validation.empty and "passes" in validation else 0
    total_validation = len(validation)
    readme = f"""# CP + PM Incremental Value Test

Generated: {datetime.utcnow().isoformat()}Z

## Purpose

This run tests whether adding Prime-Modulo (PM) temporal-granularity features improves a Contiguous-Prime (CP) local-volatility baseline.

## Methodological Choice

Saved CP validation passed for {validation_passes}/{total_validation} assets.

Because saved `Predicted_CP` could not be treated as perfectly reproducible for every asset, the primary interpretable test uses a **fresh apples-to-apples baseline**:

- Fresh CP: recomputed from raw data with current code/settings.
- Fresh CP+PM raw: recomputed from the same data with the same target, same warmup, same row alignment, and same expanding-window logic.

Mode used: `{mode}`

This answers: holding the current pipeline fixed, does adding PM improve CP?

## Speed/fit implementation

The prior script timed out because it called statsmodels OLS at every timestamp. This run uses incremental normal-equation updates with the same expanding-window alignment. A tiny numerical ridge of `{NUMERIC_RIDGE}` is used only for stable solves. CP and CP+PM use identical solver logic.

## Anti-Lookahead Rule

All conditional states are built using lagged `Actual.shift(k)` values only, with k >= 1.

## Tickers Included

{', '.join(tickers)}

## Validation Summary

{dataframe_to_text(validation)}

## Shift Diagnostic Preview

{dataframe_to_text(shift_diag)}

## Fresh Conditional Hybrid Summary

Positive advantage means fresh CP+PM raw beats fresh CP.

{dataframe_to_text(pooled)}

## Runtime Metadata Preview

{dataframe_to_text(pd.DataFrame(meta))}

## Key Files

- `results/cp_recompute_validation.csv`
- `results/cp_alignment_shift_diagnostics.csv`
- `results/fresh_run_metadata.csv`
- `results/fresh_run_failures.csv`
- `results/fresh_cp_vs_hybrid_predictions_summary.csv`
- `results/fresh_conditional_hybrid_summary_by_asset.csv`
- `results/fresh_conditional_hybrid_summary_pooled.csv`
- `results/top_fresh_hybrid_conditions.csv`
- `figures/fresh_hybrid_advantage_by_condition.png`
- `figures/fresh_hybrid_win_rate_by_condition.png`
- `latex/fresh_conditional_hybrid_summary_pooled.tex`
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def write_manifest(tickers: list[str], validation: pd.DataFrame, mode: str, started_at: float, meta: list[dict]):
    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "input_predictions_dir": str(PRED_DIR),
        "input_data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "n_lags": N_LAGS,
        "warmup": WARMUP,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "numeric_ridge": NUMERIC_RIDGE,
        "tickers": tickers,
        "mode": mode,
        "elapsed_seconds": time.perf_counter() - started_at,
        "saved_cp_validation_passes": int(validation["passes"].sum()) if not validation.empty and "passes" in validation else 0,
        "saved_cp_validation_total": int(len(validation)),
        "fit_method": "fast expanding normal-equation updates",
        "runtime_metadata": meta,
        "notes": "Fresh CP vs fresh CP+PM is the primary apples-to-apples incremental test.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    started_at = time.perf_counter()
    print("Starting CP + PM incremental test")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Predictions dir: {PRED_DIR}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Settings: n_lags={N_LAGS}, warmup={WARMUP}, bootstraps={BOOTSTRAP_RESAMPLES}, ridge={NUMERIC_RIDGE}")

    existing_predictions = load_existing_predictions()
    if not existing_predictions:
        raise RuntimeError("No existing prediction CSVs with Predicted_CP were found")

    validation_rows = []
    shift_rows = []
    fresh_frames: dict[str, pd.DataFrame] = {}
    lagged_frames: dict[str, pd.DataFrame] = {}
    overall_rows = []
    condition_rows = []
    failures = []
    runtime_meta = []

    for ticker, existing_df in existing_predictions.items():
        print(f"\n=== {ticker} ===")
        fresh_cp, fresh_hybrid, meta = build_fresh_forecasts_for_ticker(ticker)
        runtime_meta.append(meta)

        validation = validate_cp_recomputation(existing_df, fresh_cp, ticker)
        validation_rows.append(validation)
        if fresh_cp is not None and not fresh_cp.empty:
            shift_rows.append(build_shift_diagnostics(existing_df, fresh_cp, ticker))
        else:
            failures.append({"ticker": ticker, "stage": "fresh_cp", "message": validation.get("message", "fresh CP missing")})
            continue

        print(
            f"Saved CP diagnostic for {ticker}: status={validation.get('status')} "
            f"corr={validation.get('correlation')} mean_abs_diff={validation.get('mean_abs_diff')}"
        )

        if fresh_hybrid is None or fresh_hybrid.empty:
            failures.append({"ticker": ticker, "stage": "fresh_hybrid", "message": "fresh CP+PM missing"})
            continue

        merged = merge_fresh_cp_and_hybrid(ticker, fresh_cp, fresh_hybrid)
        if merged.empty:
            failures.append({"ticker": ticker, "stage": "merge", "message": "fresh CP and hybrid had no overlapping rows"})
            continue

        fresh_frames[ticker] = merged
        lagged = add_lagged_actuals(merged)
        lagged_frames[ticker] = lagged

        overall_rows.append({
            "asset": ticker,
            "n_obs": int(len(merged)),
            "CP_mean_SMAPE": safe_float(merged["SMAPE_CP_FRESH_pct"].mean()),
            "HYBRID_mean_SMAPE": safe_float(merged["SMAPE_CP_PLUS_PM_RAW_pct"].mean()),
            "mean_Hybrid_advantage_vs_CP": safe_float(merged["Hybrid_advantage_vs_CP"].mean()),
            "median_Hybrid_advantage_vs_CP": safe_float(merged["Hybrid_advantage_vs_CP"].median()),
            "hybrid_win_rate_vs_CP": safe_float((merged["Hybrid_advantage_vs_CP"] > 0).mean()),
            "mean_AbsErr_CP": safe_float(merged["AbsErr_CP_FRESH"].mean()),
            "mean_AbsErr_HYBRID": safe_float(merged["AbsErr_CP_PLUS_PM_RAW"].mean()),
            "mean_AbsErr_advantage_vs_CP": safe_float(merged["AbsErr_advantage_vs_CP"].mean()),
        })

        conditions = define_conditions(lagged)
        for condition in CONDITION_ORDER:
            condition_rows.append(compute_asset_condition_metrics(ticker, lagged, condition, conditions[condition]))

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(RESULTS_DIR / "cp_recompute_validation.csv", index=False)

    shift_df = pd.concat(shift_rows, ignore_index=True) if shift_rows else pd.DataFrame()
    shift_df.to_csv(RESULTS_DIR / "cp_alignment_shift_diagnostics.csv", index=False)

    pd.DataFrame(runtime_meta).to_csv(RESULTS_DIR / "fresh_run_metadata.csv", index=False)
    pd.DataFrame(failures).to_csv(RESULTS_DIR / "fresh_run_failures.csv", index=False)

    if not fresh_frames:
        mode = "failed_no_fresh_frames"
        empty = pd.DataFrame()
        empty.to_csv(RESULTS_DIR / "fresh_cp_vs_hybrid_predictions_summary.csv", index=False)
        empty.to_csv(RESULTS_DIR / "fresh_conditional_hybrid_summary_by_asset.csv", index=False)
        empty.to_csv(RESULTS_DIR / "fresh_conditional_hybrid_summary_pooled.csv", index=False)
        empty.to_csv(RESULTS_DIR / "top_fresh_hybrid_conditions.csv", index=False)
        write_readme(validation_df, shift_df, empty, mode, [], runtime_meta)
        write_manifest([], validation_df, mode, started_at, runtime_meta)
        raise RuntimeError("No fresh CP vs hybrid comparisons could be built")

    mode = "fresh_cp_vs_fresh_cp_plus_pm_raw"
    overall_df = pd.DataFrame(overall_rows).sort_values("asset")
    by_asset_df = pd.DataFrame(condition_rows)
    pooled_df = compute_pooled_metrics(by_asset_df, lagged_frames)
    top_df = pooled_df.sort_values("equal_weight_asset_mean_advantage", ascending=False).reset_index(drop=True)

    overall_df.to_csv(RESULTS_DIR / "fresh_cp_vs_hybrid_predictions_summary.csv", index=False)
    by_asset_df.to_csv(RESULTS_DIR / "fresh_conditional_hybrid_summary_by_asset.csv", index=False)
    pooled_df.to_csv(RESULTS_DIR / "fresh_conditional_hybrid_summary_pooled.csv", index=False)
    top_df.to_csv(RESULTS_DIR / "top_fresh_hybrid_conditions.csv", index=False)

    make_plots(pooled_df)
    write_latex_table(pooled_df)
    write_readme(validation_df, shift_df, pooled_df, mode, sorted(fresh_frames.keys()), runtime_meta)
    write_manifest(sorted(fresh_frames.keys()), validation_df, mode, started_at, runtime_meta)

    print("\n=== Fresh CP vs Fresh CP+PM Summary ===")
    print(pooled_df.to_string(index=False))
    print("\n=== Top Hybrid Conditions ===")
    print(top_df[["condition", "equal_weight_asset_mean_advantage", "asset_win_rate"]].head(10).to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
