"""
Selective overlay tests for PM/path-shape challengers.

The overlay uses saved prediction panels:

    overlay = CP + active_flag * (challenger - CP)

Top-percentile gates are based on the forecast-time perturbation magnitude
`abs(challenger - CP)`. Validation-selected thresholds use only an initial
chronological prefix of the saved OOS panel and evaluate on later rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTION_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "cp_repo_ops_hg_actions"
    / "28269511971"
    / "combined_manual"
    / "predictions"
    / "full"
)
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "pm_selective_overlay_tests"
DEFAULT_ASSETS = ["AAPL", "AMZN", "EEM", "FXI", "GLD", "GOOGL", "HYG", "QQQ", "SPY", "TLT"]
DEFAULT_CHALLENGERS = [
    "RAW_CP_REPO_PLUS_PM",
    "CP_REPO_OPS_HG",
    "CP_REPO_GATED_RIDGE_OPS_C",
    "CP_REPO_RECENT_SLOPE",
    "CP_REPO_HAAR_SHAPE",
    "SHUFFLED_LAG_PM_PLACEBO_REPO",
    "RANDOM_GATE_PLACEBO_REPO",
]
DEFAULT_THRESHOLD_MODES = [
    "top_5pct",
    "top_10pct",
    "top_20pct",
    "top_30pct",
    "validation_selected",
    "condition_cluster_entry_loose",
    "condition_entry_with_recent_spike",
]


@dataclass(frozen=True)
class OverlaySpec:
    challenger: str
    strategy: str
    gate_type: str
    active: pd.Series
    eval_mask: pd.Series
    selected_coverage: float
    selected_threshold: float
    validation_score: float
    condition_label_type: str


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def parse_csv(value: str | None, default: Iterable[str]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def ensure_dirs(outdir: Path) -> None:
    for name in ["results", "paper_tables", "logs"]:
        (outdir / name).mkdir(parents=True, exist_ok=True)


def smape(actual: pd.Series, pred: pd.Series) -> pd.Series:
    actual = pd.to_numeric(actual, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    denom = actual.abs() + pred.abs()
    out = 200.0 * (actual - pred).abs() / denom.replace(0, np.nan)
    return out.fillna(0.0)


def add_lagged_actuals(frame: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
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

    lag_matrix_values = frame[lag_cols_1_10].to_numpy()
    max_positions = np.nanargmax(lag_matrix_values, axis=1) + 1
    conditions["recent_spike_position"] = pd.Series(np.isin(max_positions, [1, 2, 3]), index=frame.index)
    conditions["older_spike_position"] = pd.Series(np.isin(max_positions, [8, 9, 10]), index=frame.index)

    path_score = (recent_mean_1_5 - older_mean_6_10).abs()
    mean_low = local_mean_1_10.quantile(0.40)
    mean_high = local_mean_1_10.quantile(0.60)
    conditions["same_average_different_path"] = local_mean_1_10.between(mean_low, mean_high) & (
        path_score >= path_score.quantile(0.80)
    )
    conditions["recent_ramp_high_dispersion"] = entry_loose & (dispersion >= dispersion.quantile(0.80))
    conditions["entry_with_recent_spike"] = entry_loose & conditions["recent_spike_position"]
    conditions["entry_without_recent_spike"] = entry_loose & ~conditions["recent_spike_position"]
    conditions["stable_low_dispersion"] = dispersion <= dispersion.quantile(0.20)
    return {name: mask.fillna(False).astype(bool) for name, mask in conditions.items()}


def top_active(score: pd.Series, coverage: float, mask: pd.Series | None = None) -> tuple[pd.Series, float]:
    base = pd.Series(False, index=score.index)
    eligible = score if mask is None else score[mask]
    eligible = pd.to_numeric(eligible, errors="coerce").dropna()
    if eligible.empty:
        return base, np.nan
    threshold = float(eligible.quantile(1.0 - coverage))
    base.loc[score >= threshold] = True
    if mask is not None:
        base &= mask
    return base.fillna(False).astype(bool), threshold


def threshold_coverages(modes: list[str]) -> list[float]:
    coverages = []
    for mode in modes:
        if mode.startswith("top_") and mode.endswith("pct"):
            number = mode.removeprefix("top_").removesuffix("pct")
            coverages.append(float(number) / 100.0)
    return sorted(set(coverages))


def row_metrics(actual: pd.Series, cp_pred: pd.Series, challenger_pred: pd.Series, overlay_pred: pd.Series) -> pd.DataFrame:
    cp_smape = smape(actual, cp_pred)
    challenger_smape = smape(actual, challenger_pred)
    overlay_smape = smape(actual, overlay_pred)
    return pd.DataFrame(
        {
            "cp_smape": cp_smape,
            "challenger_smape": challenger_smape,
            "overlay_smape": overlay_smape,
            "overlay_advantage": cp_smape - overlay_smape,
            "challenger_advantage": cp_smape - challenger_smape,
            "cp_abs_error": (actual - cp_pred).abs(),
            "overlay_abs_error": (actual - overlay_pred).abs(),
            "challenger_abs_error": (actual - challenger_pred).abs(),
            "cp_sq_error": (actual - cp_pred) ** 2,
            "overlay_sq_error": (actual - overlay_pred) ** 2,
            "challenger_sq_error": (actual - challenger_pred) ** 2,
        }
    )


def summarize_subset(
    asset: str,
    challenger: str,
    strategy: str,
    gate_type: str,
    subset_name: str,
    metrics: pd.DataFrame,
    active: pd.Series,
    eval_mask: pd.Series,
    condition_label_type: str,
) -> dict:
    mask = eval_mask.fillna(False).astype(bool)
    if subset_name != "all_observations":
        # The caller has already intersected eval_mask for condition subsets.
        condition_label_type = "ex_post_descriptive"
    data = metrics.loc[mask]
    active_eval = active.loc[mask].fillna(False).astype(bool)
    if data.empty:
        return {
            "asset": asset,
            "challenger_model": challenger,
            "strategy": strategy,
            "gate_type": gate_type,
            "condition": subset_name,
            "condition_label_type": condition_label_type,
            "n_obs": 0,
            "coverage": 0.0,
            "n_active": 0,
            "CP_mean_SMAPE": np.nan,
            "overlay_mean_SMAPE": np.nan,
            "mean_advantage_vs_CP": np.nan,
            "median_advantage_vs_CP": np.nan,
            "win_rate_vs_CP": np.nan,
            "mean_abs_error_overlay": np.nan,
            "mean_abs_error_CP": np.nan,
            "mean_abs_error_advantage_vs_CP": np.nan,
            "mean_sq_error_overlay": np.nan,
            "mean_sq_error_CP": np.nan,
            "mean_sq_error_advantage_vs_CP": np.nan,
            "false_positive_loss_mean": np.nan,
            "false_negative_cost_mean": np.nan,
        }
    active_adv = data.loc[active_eval, "overlay_advantage"]
    inactive_challenger_adv = data.loc[~active_eval, "challenger_advantage"]
    return {
        "asset": asset,
        "challenger_model": challenger,
        "strategy": strategy,
        "gate_type": gate_type,
        "condition": subset_name,
        "condition_label_type": condition_label_type,
        "n_obs": int(len(data)),
        "coverage": float(active_eval.mean()) if len(active_eval) else 0.0,
        "n_active": int(active_eval.sum()),
        "CP_mean_SMAPE": float(data["cp_smape"].mean()),
        "overlay_mean_SMAPE": float(data["overlay_smape"].mean()),
        "mean_advantage_vs_CP": float(data["overlay_advantage"].mean()),
        "median_advantage_vs_CP": float(data["overlay_advantage"].median()),
        "win_rate_vs_CP": float((data["overlay_advantage"] > 0).mean()),
        "mean_abs_error_overlay": float(data["overlay_abs_error"].mean()),
        "mean_abs_error_CP": float(data["cp_abs_error"].mean()),
        "mean_abs_error_advantage_vs_CP": float(data["cp_abs_error"].mean() - data["overlay_abs_error"].mean()),
        "mean_sq_error_overlay": float(data["overlay_sq_error"].mean()),
        "mean_sq_error_CP": float(data["cp_sq_error"].mean()),
        "mean_sq_error_advantage_vs_CP": float(data["cp_sq_error"].mean() - data["overlay_sq_error"].mean()),
        "false_positive_loss_mean": float((-active_adv[active_adv < 0]).mean()) if (active_adv < 0).any() else 0.0,
        "false_negative_cost_mean": float(inactive_challenger_adv[inactive_challenger_adv > 0].mean())
        if (inactive_challenger_adv > 0).any()
        else 0.0,
    }


def validation_selected_spec(
    frame: pd.DataFrame,
    score: pd.Series,
    cp_pred: pd.Series,
    challenger_pred: pd.Series,
    validation_fraction: float,
    coverages: list[float],
) -> tuple[pd.Series, pd.Series, float, float, float]:
    n_validation = max(1, int(len(frame) * validation_fraction))
    validation_mask = pd.Series(False, index=frame.index)
    validation_mask.iloc[:n_validation] = True
    eval_mask = ~validation_mask
    actual = frame["Actual"]
    best = None
    for coverage in coverages:
        active, threshold = top_active(score, coverage, validation_mask)
        overlay_pred = cp_pred.copy()
        overlay_pred.loc[active] = challenger_pred.loc[active]
        metrics = row_metrics(actual, cp_pred, challenger_pred, overlay_pred)
        validation_score = float(metrics.loc[validation_mask, "overlay_advantage"].mean())
        candidate = (validation_score, coverage, threshold)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        active = pd.Series(False, index=frame.index)
        return active, eval_mask, np.nan, np.nan, np.nan
    validation_score, coverage, threshold = best
    active = (score >= threshold) & eval_mask
    return active.fillna(False).astype(bool), eval_mask.astype(bool), coverage, float(threshold), validation_score


def make_specs(
    frame: pd.DataFrame,
    challenger: str,
    score: pd.Series,
    modes: list[str],
    conditions: dict[str, pd.Series],
    validation_fraction: float,
    seed: int,
) -> list[OverlaySpec]:
    specs: list[OverlaySpec] = []
    eval_all = pd.Series(True, index=frame.index)
    coverages = threshold_coverages(modes) or [0.05, 0.10, 0.20, 0.30]
    cp_pred = frame["_cp_pred"]
    challenger_pred = frame["_challenger_pred"]

    for mode in modes:
        if mode.startswith("top_") and mode.endswith("pct"):
            coverage = float(mode.removeprefix("top_").removesuffix("pct")) / 100.0
            active, threshold = top_active(score, coverage)
            specs.append(
                OverlaySpec(
                    challenger,
                    mode,
                    "perturbation_magnitude",
                    active,
                    eval_all,
                    coverage,
                    threshold,
                    np.nan,
                    "ex_post_score_quantile",
                )
            )

            shuffled = score.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            shuffled.index = score.index
            random_active, random_threshold = top_active(shuffled, coverage)
            specs.append(
                OverlaySpec(
                    challenger,
                    f"{mode}_random_gate",
                    "shuffled_perturbation_magnitude",
                    random_active,
                    eval_all,
                    coverage,
                    random_threshold,
                    np.nan,
                    "ex_post_randomized_score_quantile",
                )
            )
        elif mode == "validation_selected":
            active, eval_mask, coverage, threshold, score_value = validation_selected_spec(
                frame, score, cp_pred, challenger_pred, validation_fraction, coverages
            )
            specs.append(
                OverlaySpec(
                    challenger,
                    mode,
                    "prefix_validation_perturbation_magnitude",
                    active,
                    eval_mask,
                    coverage,
                    threshold,
                    score_value,
                    "prefix_validation_no_lookahead",
                )
            )
        elif mode.startswith("condition_"):
            condition_name = mode.removeprefix("condition_")
            active = conditions.get(condition_name, pd.Series(False, index=frame.index))
            specs.append(
                OverlaySpec(
                    challenger,
                    mode,
                    "condition_label",
                    active,
                    eval_all,
                    float(active.mean()) if len(active) else 0.0,
                    np.nan,
                    np.nan,
                    "ex_post_descriptive",
                )
            )
    return specs


def aggregate_by_strategy(by_asset: pd.DataFrame) -> pd.DataFrame:
    if by_asset.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["challenger_model", "strategy", "gate_type", "condition", "condition_label_type"]
    for keys, group in by_asset.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_assets"] = int(group["asset"].nunique())
        row["n_obs_total"] = int(group["n_obs"].sum())
        row["n_active_total"] = int(group["n_active"].sum())
        for col in [
            "coverage",
            "CP_mean_SMAPE",
            "overlay_mean_SMAPE",
            "mean_advantage_vs_CP",
            "median_advantage_vs_CP",
            "win_rate_vs_CP",
            "mean_abs_error_advantage_vs_CP",
            "mean_sq_error_advantage_vs_CP",
            "false_positive_loss_mean",
            "false_negative_cost_mean",
        ]:
            row[f"equal_weight_asset_{col}"] = float(group[col].mean())
        row["assets_positive"] = int((group["mean_advantage_vs_CP"] > 0).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["condition", "equal_weight_asset_mean_advantage_vs_CP"], ascending=[True, False])


def run_overlays(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    pred_root = Path(args.prediction_root)
    ensure_dirs(outdir)
    assets = parse_csv(args.assets, DEFAULT_ASSETS)
    challengers = parse_csv(args.challenger_models, DEFAULT_CHALLENGERS)
    modes = parse_csv(args.threshold_modes, DEFAULT_THRESHOLD_MODES)
    baseline = args.baseline_model

    overall_rows = []
    conditional_rows = []
    threshold_rows = []
    loss_rows = []
    fp_rows = []
    manifest_rows = []
    audit_rows = []

    for asset in assets:
        path = pred_root / f"{asset}.csv"
        if not path.exists():
            audit_rows.append({"asset": asset, "check": "prediction_file_exists", "passed": False, "detail": str(path)})
            continue
        raw = pd.read_csv(path)
        cp_col = f"Predicted_{baseline}"
        required_base = {"Date", "Actual", cp_col}
        missing_base = sorted(required_base - set(raw.columns))
        if missing_base:
            audit_rows.append({"asset": asset, "check": "required_base_columns", "passed": False, "detail": ",".join(missing_base)})
            continue
        raw["Date"] = pd.to_datetime(raw["Date"])
        frame = add_lagged_actuals(raw, 10)
        conditions = define_conditions(frame)
        cp_pred = pd.to_numeric(frame[cp_col], errors="coerce")
        frame["_cp_pred"] = cp_pred

        for challenger in challengers:
            challenger_col = f"Predicted_{challenger}"
            if challenger_col not in frame.columns:
                audit_rows.append(
                    {
                        "asset": asset,
                        "check": "challenger_column_exists",
                        "model": challenger,
                        "passed": False,
                        "detail": challenger_col,
                    }
                )
                continue
            challenger_pred = pd.to_numeric(frame[challenger_col], errors="coerce")
            valid = frame["Actual"].notna() & cp_pred.notna() & challenger_pred.notna()
            work = frame.loc[valid].copy().reset_index(drop=True)
            work["_cp_pred"] = cp_pred.loc[valid].reset_index(drop=True)
            work["_challenger_pred"] = challenger_pred.loc[valid].reset_index(drop=True)
            score = (work["_challenger_pred"] - work["_cp_pred"]).abs()
            work_conditions = define_conditions(add_lagged_actuals(work[["Date", "Actual"]].copy(), 10))
            # Re-align conditions after the second lag trimming.
            if len(work_conditions["all_observations"]) < len(work):
                work = work.iloc[-len(work_conditions["all_observations"]) :].reset_index(drop=True)
                score = score.iloc[-len(work) :].reset_index(drop=True)
                work["_cp_pred"] = work["_cp_pred"].reset_index(drop=True)
                work["_challenger_pred"] = work["_challenger_pred"].reset_index(drop=True)
            conditions_for_work = work_conditions

            specs = make_specs(
                work,
                challenger,
                score,
                modes,
                conditions_for_work,
                args.validation_fraction,
                args.seed,
            )
            for spec in specs:
                overlay_pred = work["_cp_pred"].copy()
                overlay_pred.loc[spec.active] = work.loc[spec.active, "_challenger_pred"]
                metrics = row_metrics(work["Actual"], work["_cp_pred"], work["_challenger_pred"], overlay_pred)

                overall = summarize_subset(
                    asset,
                    challenger,
                    spec.strategy,
                    spec.gate_type,
                    "all_observations",
                    metrics,
                    spec.active,
                    spec.eval_mask,
                    spec.condition_label_type,
                )
                overall_rows.append(overall)

                for condition_name, condition_mask in conditions_for_work.items():
                    if condition_name == "all_observations":
                        continue
                    subset_mask = spec.eval_mask & condition_mask
                    conditional_rows.append(
                        summarize_subset(
                            asset,
                            challenger,
                            spec.strategy,
                            spec.gate_type,
                            condition_name,
                            metrics,
                            spec.active,
                            subset_mask,
                            "ex_post_descriptive",
                        )
                    )

                threshold_rows.append(
                    {
                        "asset": asset,
                        "challenger_model": challenger,
                        "strategy": spec.strategy,
                        "gate_type": spec.gate_type,
                        "selected_coverage": spec.selected_coverage,
                        "selected_threshold": spec.selected_threshold,
                        "validation_score": spec.validation_score,
                        "validation_fraction": args.validation_fraction if spec.strategy == "validation_selected" else 0.0,
                        "uses_future_outcomes_for_threshold": False,
                        "uses_full_evaluation_scores_for_threshold": spec.condition_label_type
                        in {"ex_post_score_quantile", "ex_post_randomized_score_quantile"},
                    }
                )
                for metric_name, cp_col_metric, overlay_col_metric in [
                    ("SMAPE", "cp_smape", "overlay_smape"),
                    ("MAE", "cp_abs_error", "overlay_abs_error"),
                    ("MSE", "cp_sq_error", "overlay_sq_error"),
                ]:
                    eval_metrics = metrics.loc[spec.eval_mask]
                    loss_rows.append(
                        {
                            "asset": asset,
                            "challenger_model": challenger,
                            "strategy": spec.strategy,
                            "loss_metric": metric_name,
                            "n_obs": int(len(eval_metrics)),
                            "CP_loss": float(eval_metrics[cp_col_metric].mean()),
                            "overlay_loss": float(eval_metrics[overlay_col_metric].mean()),
                            "advantage_CP_minus_overlay": float(
                                eval_metrics[cp_col_metric].mean() - eval_metrics[overlay_col_metric].mean()
                            ),
                        }
                    )
                fp_rows.append(
                    {
                        "asset": asset,
                        "challenger_model": challenger,
                        "strategy": spec.strategy,
                        "false_positive_loss_mean": overall["false_positive_loss_mean"],
                        "false_negative_cost_mean": overall["false_negative_cost_mean"],
                    }
                )
                manifest_rows.append(
                    {
                        "asset": asset,
                        "baseline_model": baseline,
                        "challenger_model": challenger,
                        "strategy": spec.strategy,
                        "gate_type": spec.gate_type,
                        "overlay_formula": "CP + active_flag * (challenger - CP)",
                        "score_source": "saved forecast-time predictions",
                        "threshold_selection": spec.condition_label_type,
                        "seed": args.seed,
                        "paper_eligible": spec.condition_label_type == "prefix_validation_no_lookahead",
                    }
                )
            audit_rows.append(
                {
                    "asset": asset,
                    "check": "processed_challenger",
                    "model": challenger,
                    "passed": True,
                    "detail": f"rows={len(work)}",
                }
            )

    overall_by_asset = pd.DataFrame(overall_rows)
    conditional_by_asset = pd.DataFrame(conditional_rows)
    pooled = aggregate_by_strategy(pd.concat([overall_by_asset, conditional_by_asset], ignore_index=True))
    placebo = pooled[pooled["gate_type"].astype(str).str.contains("shuffled|random", case=False, na=False)].copy()

    write_outputs(
        outdir,
        overall_by_asset,
        conditional_by_asset,
        pooled,
        pd.DataFrame(threshold_rows),
        placebo,
        pd.DataFrame(loss_rows),
        pd.DataFrame(fp_rows),
        pd.DataFrame(manifest_rows),
        pd.DataFrame(audit_rows),
        args,
    )


def write_outputs(
    outdir: Path,
    overall_by_asset: pd.DataFrame,
    conditional_by_asset: pd.DataFrame,
    pooled: pd.DataFrame,
    thresholds: pd.DataFrame,
    placebo: pd.DataFrame,
    loss_breakdown: pd.DataFrame,
    fp_fn: pd.DataFrame,
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    result_dir = outdir / "results"
    overall_by_asset.to_csv(result_dir / "overlay_overall_by_asset.csv", index=False)
    conditional_by_asset.to_csv(result_dir / "overlay_conditional_by_asset.csv", index=False)
    pooled.to_csv(result_dir / "overlay_pooled_summary.csv", index=False)
    thresholds.to_csv(result_dir / "overlay_threshold_selection.csv", index=False)
    placebo.to_csv(result_dir / "overlay_placebo_summary.csv", index=False)
    loss_breakdown.to_csv(result_dir / "overlay_loss_breakdown.csv", index=False)
    fp_fn.to_csv(result_dir / "overlay_false_positive_negative.csv", index=False)
    manifest.to_csv(result_dir / "overlay_model_manifest.csv", index=False)
    audit.to_csv(result_dir / "overlay_audit.csv", index=False)
    top = pooled[pooled["condition"] == "all_observations"].sort_values(
        "equal_weight_asset_mean_advantage_vs_CP", ascending=False
    )
    top.to_csv(outdir / "paper_tables" / "selective_overlay_summary.csv", index=False)
    metadata = {
        "generated_at_utc": utc_now(),
        "prediction_root": args.prediction_root,
        "baseline_model": args.baseline_model,
        "challenger_models": parse_csv(args.challenger_models, DEFAULT_CHALLENGERS),
        "threshold_modes": parse_csv(args.threshold_modes, DEFAULT_THRESHOLD_MODES),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "uses_future_outcomes_for_threshold": False,
    }
    (outdir / "results" / "overlay_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (outdir / "README.md").write_text(
        "\n".join(
            [
                "# PM Selective Overlay Tests",
                "",
                "The overlay uses saved prediction panels and applies challenger perturbations only when a gate is active.",
                "",
                "Formula: `overlay = CP + active_flag * (challenger - CP)`.",
                "",
                "Top-percentile gates use `abs(challenger - CP)` from saved forecast-time predictions. Validation-selected thresholds use the initial chronological prefix only and evaluate on later rows.",
                "",
                "Condition strategies are ex-post descriptive and not tradable online labels unless rebuilt with train-only thresholds.",
                "",
                "Main outputs live in `results/`; the compact sorted table is `paper_tables/selective_overlay_summary.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failures = audit[~audit["passed"].fillna(False).astype(bool)] if not audit.empty and "passed" in audit.columns else pd.DataFrame()
    marker = outdir / ("OVERLAY_SUCCESS.txt" if failures.empty else "OVERLAY_NEEDS_REVIEW.txt")
    marker.write_text(f"PM selective overlay tests completed at {utc_now()}\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run selective PM/path-shape overlay tests from saved predictions")
    parser.add_argument("--prediction-root", default=str(DEFAULT_PREDICTION_ROOT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--assets", default="")
    parser.add_argument("--baseline-model", default="CP_REPO_FRESH")
    parser.add_argument("--challenger-models", default=",".join(DEFAULT_CHALLENGERS))
    parser.add_argument("--threshold-modes", default=",".join(DEFAULT_THRESHOLD_MODES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    return parser


def main() -> None:
    run_overlays(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
