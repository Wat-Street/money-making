"""
CP + PM Incremental Value Test
==============================

This script runs two related checks:

1. Diagnostic validation: recompute CP and compare it with the saved
   current_intraday Predicted_CP series. This is diagnostic only.
2. Fresh apples-to-apples incremental test: compare freshly recomputed CP
   against freshly recomputed CP+PM on the exact same timestamps.

Why this structure matters:
- If saved CP predictions cannot be reproduced exactly, we should not compare
  a new CP+PM model against stale/unknown saved CP predictions.
- A fresh CP baseline and fresh CP+PM model are still a valid incremental test,
  because they use identical data, target alignment, warmup, and expanding-window
  logic.

Anti-lookahead rule: all conditional state labels are built only from lagged
Actual values, i.e. Actual.shift(k) for k >= 1.
"""

import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Add project code to path. This file lives in:
# projects/harv-extension/outputs/cp_pm_incremental_tests/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from utils.data_utils import (  # noqa: E402
    calculate_intraday_realized_volatility,
    fetch_intraday_data,
    fit_and_predict_extended,
)
from utils.models_utils import add_prime_modulo_terms, contig_prime_modulo  # noqa: E402

PRED_DIR = PROJECT_ROOT / "run_results" / "current_intraday" / "predictions"
DATA_DIR = PROJECT_ROOT / "data" / "market_data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cp_pm_incremental_tests"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
LATEX_DIR = OUTPUT_DIR / "latex"

N_LAGS = 22
WARMUP = 600
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
SHIFT_OFFSETS = [-2, -1, 0, 1, 2]
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


def select_feature_columns(frame: pd.DataFrame, prefixes) -> list[str]:
    return [column for column in frame.columns if column.startswith(prefixes)]


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


def load_existing_predictions() -> dict[str, pd.DataFrame]:
    predictions = {}
    if not PRED_DIR.exists():
        raise FileNotFoundError(f"Missing predictions directory: {PRED_DIR}")

    for csv_file in sorted(PRED_DIR.glob("*.csv")):
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


def normalize_prediction_frame(preds: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return Date, Actual, Predicted_<model_name> from a prediction DataFrame."""
    if preds is None or preds.empty:
        return pd.DataFrame()

    frame = preds.reset_index().copy()
    if "Date" not in frame.columns:
        first_col = frame.columns[0]
        frame = frame.rename(columns={first_col: "Date"})

    pred_col = f"Predicted_{model_name}"
    if pred_col not in frame.columns:
        candidates = [column for column in frame.columns if column.startswith("Predicted_")]
        if len(candidates) == 1:
            frame = frame.rename(columns={candidates[0]: pred_col})
        else:
            raise ValueError(f"Could not find prediction column {pred_col}; candidates={candidates}")

    if "Actual" not in frame.columns:
        raise ValueError("Prediction frame missing Actual column")

    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame[["Date", "Actual", pred_col]].sort_values("Date").reset_index(drop=True)


def build_forecasts(ticker: str, include_pm: bool, model_name: str) -> pd.DataFrame | None:
    """Build fresh forecasts for CP or CP+PM from raw data."""
    try:
        raw = fetch_intraday_data(ticker, use_local=True, local_dir=str(DATA_DIR))
        vol = calculate_intraday_realized_volatility(raw)
        extended = contig_prime_modulo(vol.copy(), n=N_LAGS, per_day_normalize=False)
        if include_pm:
            extended = add_prime_modulo_terms(extended.copy(), n=N_LAGS)

        prefixes = HYBRID_FEATURE_PREFIXES if include_pm else CP_FEATURE_PREFIXES
        features = select_feature_columns(extended, prefixes)
        if not features:
            print(f"{ticker}: no features selected for {model_name}")
            return None

        preds = fit_and_predict_extended(
            extended,
            features,
            n=N_LAGS,
            warmup=WARMUP,
            model_name=model_name,
        )
        if preds is None or preds.empty:
            print(f"{ticker}: empty predictions for {model_name}")
            return None
        return normalize_prediction_frame(preds, model_name)
    except Exception as exc:
        print(f"{ticker}: failed to build {model_name}: {exc}")
        return None


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

    # Diagnostic threshold only. Do not use this to decide whether fresh CP vs fresh hybrid is valid.
    passes = bool((corr is not None) and (not pd.isna(corr)) and corr > 0.995 and abs_diff.mean() < 1e-8)

    return {
        "ticker": ticker,
        "status": "passed" if passes else "failed",
        "message": "diagnostic exact-saved-CP validation",
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
        merged = existing.merge(
            shifted,
            on="Date",
            how="inner",
            suffixes=("_existing", "_shifted"),
        ).sort_values("Date")

        if merged.empty:
            rows.append({
                "ticker": ticker,
                "shift": shift,
                "n_aligned": 0,
                "mean_abs_diff": np.nan,
                "median_abs_diff": np.nan,
                "max_abs_diff": np.nan,
                "correlation": np.nan,
                "first_aligned_date": None,
                "last_aligned_date": None,
            })
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
    merged = fresh_cp.merge(
        fresh_hybrid,
        on="Date",
        how="inner",
        suffixes=("_CP", "_HYBRID"),
    ).sort_values("Date").reset_index(drop=True)

    if merged.empty:
        return merged

    # Actual values should match across fresh CP and fresh hybrid because both use the same data/target.
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
    conditions = {}
    conditions["all_observations"] = pd.Series(True, index=frame.index)

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
    conditions["same_average_different_path"] = (
        local_mean_1_10.between(mean_low, mean_high)
        & (path_score >= path_score.quantile(0.80))
    )

    return {name: mask.fillna(False).astype(bool) for name, mask in conditions.items()}


def bootstrap_ci(values: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
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
        return {
            "asset": asset,
            "condition": condition_name,
            "n_obs": 0,
            "share_of_asset_obs": 0.0,
        }

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


def compute_pooled_metrics(asset_results: pd.DataFrame, all_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        asset_subset = asset_results[(asset_results["condition"] == condition) & (asset_results["n_obs"] > 0)].copy()
        pooled_parts = []
        for asset, frame in all_frames.items():
            frame_lagged = add_lagged_actuals(frame)
            conditions = define_conditions(frame_lagged)
            mask = conditions[condition]
            pooled_parts.append(frame_lagged.loc[mask])
        pooled = pd.concat(pooled_parts, ignore_index=True) if pooled_parts else pd.DataFrame()

        if asset_subset.empty or pooled.empty:
            rows.append({
                "condition": condition,
                "total_n_obs": 0,
                "number_of_assets": 0,
            })
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
        return
    cols = [
        "condition",
        "total_n_obs",
        "equal_weight_asset_mean_advantage",
        "equal_weight_asset_hybrid_win_rate",
        "asset_win_rate",
    ]
    table = pooled[cols].copy()
    table = table.rename(columns={
        "condition": "Condition",
        "total_n_obs": "N",
        "equal_weight_asset_mean_advantage": "Adv.",
        "equal_weight_asset_hybrid_win_rate": "Win Rate",
        "asset_win_rate": "Asset Win Rate",
    })
    latex = table.to_latex(index=False, float_format="%.4f")
    (LATEX_DIR / "fresh_conditional_hybrid_summary_pooled.tex").write_text(latex, encoding="utf-8")


def dataframe_to_text(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame is None or frame.empty:
        return "[empty]"
    return frame.head(max_rows).to_string(index=False)


def write_readme(validation: pd.DataFrame, shift_diag: pd.DataFrame, pooled: pd.DataFrame, mode: str, tickers: list[str]):
    validation_passes = int(validation["passes"].sum()) if not validation.empty and "passes" in validation else 0
    total_validation = len(validation)
    readme = f"""# CP + PM Incremental Value Test

Generated: {datetime.utcnow().isoformat()}Z

## Purpose

This run tests whether adding Prime-Modulo (PM) temporal-granularity features improves a Contiguous-Prime (CP) local-volatility baseline.

## Important Methodological Choice

Saved CP validation passed for {validation_passes}/{total_validation} assets.

Because saved `Predicted_CP` could not be treated as perfectly reproducible for every asset, the interpretable incremental test uses a **fresh apples-to-apples baseline**:

- Fresh CP: recomputed from raw data with current code/settings.
- Fresh CP+PM raw: recomputed from the same data with the same target, same warmup, same row alignment, and same expanding-window logic.

Mode used: `{mode}`

This means the hybrid comparison answers:

> Holding the current pipeline fixed, does adding PM features improve CP?

It does **not** claim to compare CP+PM against the exact historical saved CP predictions from the paper.

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

## Key Files

- `results/cp_recompute_validation.csv`
- `results/cp_alignment_shift_diagnostics.csv`
- `results/fresh_cp_vs_hybrid_predictions_summary.csv`
- `results/fresh_conditional_hybrid_summary_by_asset.csv`
- `results/fresh_conditional_hybrid_summary_pooled.csv`
- `results/top_fresh_hybrid_conditions.csv`
- `figures/fresh_hybrid_advantage_by_condition.png`
- `figures/fresh_hybrid_win_rate_by_condition.png`
- `latex/fresh_conditional_hybrid_summary_pooled.tex`
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def write_manifest(tickers: list[str], validation: pd.DataFrame, mode: str):
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
        "tickers": tickers,
        "mode": mode,
        "saved_cp_validation_passes": int(validation["passes"].sum()) if not validation.empty and "passes" in validation else 0,
        "saved_cp_validation_total": int(len(validation)),
        "notes": "Fresh CP vs fresh CP+PM is the primary apples-to-apples incremental test.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    print("Starting CP + PM incremental test")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Predictions dir: {PRED_DIR}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    existing_predictions = load_existing_predictions()
    if not existing_predictions:
        raise RuntimeError("No existing prediction CSVs with Predicted_CP were found")

    validation_rows = []
    shift_rows = []
    fresh_frames = {}
    overall_rows = []
    condition_rows = []
    failures = []

    for ticker, existing_df in existing_predictions.items():
        print(f"\n=== {ticker} ===")
        fresh_cp = build_forecasts(ticker, include_pm=False, model_name="CP")
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

        fresh_hybrid = build_forecasts(ticker, include_pm=True, model_name="CP_PLUS_PM_RAW")
        if fresh_hybrid is None or fresh_hybrid.empty:
            failures.append({"ticker": ticker, "stage": "fresh_hybrid", "message": "fresh CP+PM missing"})
            continue

        merged = merge_fresh_cp_and_hybrid(ticker, fresh_cp, fresh_hybrid)
        if merged.empty:
            failures.append({"ticker": ticker, "stage": "merge", "message": "fresh CP and hybrid had no overlapping rows"})
            continue

        fresh_frames[ticker] = merged

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

        lagged = add_lagged_actuals(merged)
        conditions = define_conditions(lagged)
        for condition in CONDITION_ORDER:
            condition_rows.append(compute_asset_condition_metrics(ticker, lagged, condition, conditions[condition]))

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(RESULTS_DIR / "cp_recompute_validation.csv", index=False)

    shift_df = pd.concat(shift_rows, ignore_index=True) if shift_rows else pd.DataFrame()
    shift_df.to_csv(RESULTS_DIR / "cp_alignment_shift_diagnostics.csv", index=False)

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(RESULTS_DIR / "fresh_run_failures.csv", index=False)

    if not fresh_frames:
        mode = "failed_no_fresh_frames"
        empty = pd.DataFrame()
        empty.to_csv(RESULTS_DIR / "fresh_cp_vs_hybrid_predictions_summary.csv", index=False)
        empty.to_csv(RESULTS_DIR / "fresh_conditional_hybrid_summary_by_asset.csv", index=False)
        empty.to_csv(RESULTS_DIR / "fresh_conditional_hybrid_summary_pooled.csv", index=False)
        empty.to_csv(RESULTS_DIR / "top_fresh_hybrid_conditions.csv", index=False)
        write_readme(validation_df, shift_df, empty, mode, [])
        write_manifest([], validation_df, mode)
        raise RuntimeError("No fresh CP vs hybrid comparisons could be built")

    mode = "fresh_cp_vs_fresh_cp_plus_pm_raw"
    overall_df = pd.DataFrame(overall_rows).sort_values("asset")
    by_asset_df = pd.DataFrame(condition_rows)
    pooled_df = compute_pooled_metrics(by_asset_df, fresh_frames)
    top_df = pooled_df.sort_values("equal_weight_asset_mean_advantage", ascending=False).reset_index(drop=True)

    overall_df.to_csv(RESULTS_DIR / "fresh_cp_vs_hybrid_predictions_summary.csv", index=False)
    by_asset_df.to_csv(RESULTS_DIR / "fresh_conditional_hybrid_summary_by_asset.csv", index=False)
    pooled_df.to_csv(RESULTS_DIR / "fresh_conditional_hybrid_summary_pooled.csv", index=False)
    top_df.to_csv(RESULTS_DIR / "top_fresh_hybrid_conditions.csv", index=False)

    make_plots(pooled_df)
    write_latex_table(pooled_df)
    write_readme(validation_df, shift_df, pooled_df, mode, sorted(fresh_frames.keys()))
    write_manifest(sorted(fresh_frames.keys()), validation_df, mode)

    print("\n=== Fresh CP vs Fresh CP+PM Summary ===")
    print(pooled_df.to_string(index=False))
    print("\n=== Top Hybrid Conditions ===")
    print(top_df[["condition", "equal_weight_asset_mean_advantage", "asset_win_rate"]].head(10).to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
