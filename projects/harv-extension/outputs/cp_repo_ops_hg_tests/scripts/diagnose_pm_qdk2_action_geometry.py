#!/usr/bin/env python
"""Separate PM_QDK_2 quotient-geometry value from its loss action on fixed origins."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import run_cp_repo_ops_hg_tests as runner


DEFAULT_ASSETS = ["AAPL", "AMZN", "EEM", "FXI", "GLD", "GOOGL", "HYG", "QQQ", "SPY", "TLT"]


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big")


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5, side="left")])


def smape(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 200.0 * np.abs(actual - pred) / np.maximum(np.abs(actual) + np.abs(pred), 1e-12)


def runner_args() -> argparse.Namespace:
    args = runner.build_arg_parser().parse_args([])
    args.mode = "full"
    args.n = 22
    args.warmup = 600
    args.seed = 42
    args.pm_kernel_train_window = 2500
    args.pm_low_modes = 12
    args.log_eps = 1e-12
    args.pm_tau_grid_values = runner.parse_float_grid(args.pm_tau_grid, runner.DEFAULT_PM_TAU_GRID)
    args.pm_eta_grid_values = runner.parse_float_grid(args.pm_eta_grid, runner.DEFAULT_PM_ETA_GRID)
    args.pm_bandwidth_grid_values = runner.parse_float_grid(args.pm_bandwidth_grid, runner.DEFAULT_PM_BANDWIDTH_GRID)
    args.phqo_gamma_grid_values = runner.parse_float_grid(args.phqo_gamma_grid, runner.DEFAULT_PHQO_GAMMA_GRID)
    args.lambda_grid_values = runner.parse_float_grid(args.lambda_grid, runner.DEFAULT_LAMBDA_GRID)
    args.lambda_r_ratio_grid_values = runner.parse_float_grid(args.lambda_r_ratio_grid, runner.DEFAULT_LAMBDA_R_RATIO_GRID)
    args.model_registry = runner.build_model_registry(22)
    return args


def diagnose_asset(
    asset: str,
    args: argparse.Namespace,
    panel_root: Path,
    bandwidth: float,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    frame, _, _, _, _ = runner.build_feature_frame(asset, args)
    x = runner.lag_matrix(frame, args.n)
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    valid_x = np.isfinite(x).all(axis=1)
    level, c_state, phi = runner.pm_qdk2_embedding(x, args.n, args.pm_tau_grid_values, args.log_eps, args.pm_low_modes)
    q = runner.expanding_linear_residuals(phi, c_state, valid_x & np.isfinite(y_next))
    first_oos = args.n + args.warmup
    level_scale = float(runner.robust_scale(level[:first_oos, None])[0])
    c_scale = runner.robust_scale(c_state[:first_oos])
    q_scale = runner.robust_scale(q[:first_oos])
    panel = pd.read_csv(panel_root / f"{asset}_post_validation.csv", parse_dates=["Date"])
    cp_by_date = dict(zip(pd.to_datetime(panel["Date"]), pd.to_numeric(panel["Predicted_CP_REPO_FRESH"], errors="coerce")))
    valid = valid_x & np.isfinite(y_next) & np.isfinite(q).all(axis=1)
    origins = np.flatnonzero(valid & (np.arange(len(frame)) >= first_oos) & (np.arange(len(frame)) < len(frame) - 1))
    rng = np.random.default_rng(stable_seed(asset, seed))
    origins = np.sort(rng.choice(origins, size=min(sample_size, len(origins)), replace=False))
    rows = []
    h = max(float(bandwidth), 1e-12)
    for origin in origins:
        train_idx = runner.candidate_train_indices(valid, y_next, int(origin), args.pm_kernel_train_window)
        if train_idx.size < 5:
            continue
        dl = np.square((level[train_idx] - level[origin]) / level_scale)
        dc = (c_state[train_idx] - c_state[origin]) / c_scale
        dq = (q[train_idx] - q[origin]) / q_scale
        base_distance = dl + np.sum(np.square(dc), axis=1)
        quotient_distance = np.sum(np.square(dq), axis=1) / max(q.shape[1], 1)
        full_weights = runner.softmax_weights(-0.5 * (base_distance + quotient_distance) / (h * h))
        noq_weights = runner.softmax_weights(-0.5 * base_distance / (h * h))
        if full_weights.size == 0 or noq_weights.size == 0:
            continue
        history = y_next[train_idx]
        target_date = pd.Timestamp(frame.index[origin + 1])
        cp_pred = cp_by_date.get(target_date, np.nan)
        rows.append(
            {
                "asset": asset,
                "origin": int(origin),
                "Date": target_date,
                "Actual": float(y_next[origin]),
                "Predicted_CP_REPO_FRESH": float(cp_pred),
                "Predicted_FULL_SMAPE_ACTION": runner.weighted_smape_action(history, full_weights),
                "Predicted_FULL_MEAN_ACTION": float(np.sum(full_weights * history)),
                "Predicted_FULL_MEDIAN_ACTION": weighted_median(history, full_weights),
                "Predicted_NO_Q_SMAPE_ACTION": runner.weighted_smape_action(history, noq_weights),
                "selected_bandwidth": float(bandwidth),
                "full_effective_neighbors": float(1.0 / np.sum(np.square(full_weights))),
                "no_q_effective_neighbors": float(1.0 / np.sum(np.square(noq_weights))),
                "mean_quotient_distance_share": float(np.mean(quotient_distance / np.maximum(base_distance + quotient_distance, 1e-12))),
            }
        )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    models = {
        "CP_REPO_FRESH": "Predicted_CP_REPO_FRESH",
        "FULL_SMAPE_ACTION": "Predicted_FULL_SMAPE_ACTION",
        "FULL_MEAN_ACTION": "Predicted_FULL_MEAN_ACTION",
        "FULL_MEDIAN_ACTION": "Predicted_FULL_MEDIAN_ACTION",
        "NO_Q_SMAPE_ACTION": "Predicted_NO_Q_SMAPE_ACTION",
    }
    output = []
    groups = list(rows.groupby("asset", sort=True)) + [("POOLED", rows)]
    for asset, group in groups:
        actual = pd.to_numeric(group["Actual"], errors="coerce").to_numpy(dtype=float)
        for model, column in models.items():
            pred = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(actual) & np.isfinite(pred)
            error = actual[valid] - pred[valid]
            output.append(
                {
                    "asset": asset,
                    "model_name": model,
                    "n_obs": int(valid.sum()),
                    "SMAPE": float(np.mean(smape(actual[valid], pred[valid]))),
                    "MAE": float(np.mean(np.abs(error))),
                    "MSE": float(np.mean(np.square(error))),
                    "RMSE": float(np.sqrt(np.mean(np.square(error)))),
                    "mean_effective_neighbors": float(group["full_effective_neighbors"].mean()),
                    "mean_no_q_effective_neighbors": float(group["no_q_effective_neighbors"].mean()),
                    "mean_quotient_distance_share": float(group["mean_quotient_distance_share"].mean()),
                }
            )
    return pd.DataFrame(output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-root", default="outputs/pm_native_geometry_warmup600_discriminating_validation/predictions/full")
    parser.add_argument("--metadata", default="outputs/pm_native_geometry_warmup600_actions/28733449635_combined/results/run_metadata.csv")
    parser.add_argument("--outdir", default="outputs/pm_prop4_mathematical_audit_20260709")
    return parser


def main() -> None:
    cli = build_arg_parser().parse_args()
    assets = [asset.strip() for asset in cli.assets.split(",") if asset.strip()]
    metadata = pd.read_csv(cli.metadata)
    bandwidth_by_asset = {
        str(row["asset"]): float(row["selected_pm_bandwidth"])
        for _, row in metadata.loc[metadata["model_name"] == "PM_QDK_2"].iterrows()
    }
    args = runner_args()
    frames = []
    for asset in assets:
        print(f"Diagnosing {asset}", flush=True)
        frames.append(
            diagnose_asset(
                asset,
                args,
                Path(cli.panel_root),
                bandwidth_by_asset[asset],
                int(cli.sample_size),
                int(cli.seed),
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    outdir = Path(cli.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(outdir / "pm_action_geometry_origin_sample.csv", index=False)
    summarize(rows).to_csv(outdir / "pm_action_geometry_summary.csv", index=False)
    print(f"Wrote {len(rows)} fixed-origin diagnostics to {outdir}")


if __name__ == "__main__":
    main()
