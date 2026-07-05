#!/usr/bin/env python
"""Post-result validation for the PM-native warmup-600 geometry run.

This script intentionally reuses the Prop 4 runner's data loading, CP feature
construction, PM embeddings, validation-window logic, and SMAPE action.  The
validation-only placebo forecasts are not model-registry additions; they are
matched diagnostic kernels used to test whether PM_QDK_2's SMAPE win survives
random-residue, shuffled-lag, and generic Haar/shape geometries.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is installed in CI, fallback is for local minimal envs.
    stats = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_cp_repo_ops_hg_tests as runner  # noqa: E402


CORE_MODELS = ["CP_REPO_FRESH", "PM_QDK_2", "PHQO", "PM_QDK"]
PLACEBO_MODELS = [
    "PM_QDK_2_RANDOM_RESIDUE_PLACEBO",
    "PM_QDK_2_SHUFFLED_LAG_PLACEBO",
    "PM_QDK_2_HAAR_SHAPE_PLACEBO",
]
VALIDATION_MODELS = CORE_MODELS + PLACEBO_MODELS
LOSS_METRICS = ["SMAPE", "MAE", "MSE", "RMSE"]
ASSET_UNIVERSE = ["AAPL", "AMZN", "EEM", "FXI", "GLD", "GOOGL", "HYG", "QQQ", "SPY", "TLT"]
DEFAULT_SOURCE_RUN = Path("outputs/pm_native_geometry_warmup600_full")
DEFAULT_OUTDIR = Path("outputs/pm_native_geometry_warmup600_post_validation")


@dataclass
class PlaceboPrediction:
    model_name: str
    frame: pd.DataFrame
    selected_bandwidth: float
    cv_score: float
    phi_dim: int
    basis_description: str


def parse_assets(text: str) -> list[str]:
    if not text.strip():
        return ASSET_UNIVERSE
    return [part.strip().upper() for part in text.split(",") if part.strip()]


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def stable_seed_offset(value: object, modulo: int = 100000) -> int:
    text = json.dumps(value, sort_keys=True, default=str)
    total = 0
    for char in text:
        total = (total * 131 + ord(char)) % modulo
    return int(total)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def smape_array(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return 200.0 * np.abs(actual - pred) / np.maximum(eps, np.abs(actual) + np.abs(pred))


def model_loss_arrays(panel: pd.DataFrame, model_name: str) -> dict[str, np.ndarray]:
    pred_col = f"Predicted_{model_name}"
    if pred_col not in panel.columns:
        n = len(panel)
        return {metric: np.full(n, np.nan, dtype=float) for metric in LOSS_METRICS}
    actual = pd.to_numeric(panel["Actual"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(panel[pred_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred)
    err = actual - pred
    out = {
        "SMAPE": np.where(valid, smape_array(actual, pred), np.nan),
        "MAE": np.where(valid, np.abs(err), np.nan),
        "MSE": np.where(valid, np.square(err), np.nan),
    }
    rmse_scalar = math.sqrt(float(np.nanmean(out["MSE"]))) if np.isfinite(out["MSE"]).any() else np.nan
    out["RMSE"] = np.where(valid, np.sqrt(np.square(err)), np.nan)
    # The row array is absolute error for row-wise comparisons; aggregate rows use sqrt(mean MSE).
    out["_RMSE_SCALAR"] = np.array([rmse_scalar], dtype=float)
    return out


def aggregate_loss(values: np.ndarray, metric: str) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).any():
        return np.nan
    if metric == "RMSE":
        return float(np.sqrt(np.nanmean(np.square(arr))))
    return float(np.nanmean(arr))


def metric_value(actual: np.ndarray, pred: np.ndarray, metric: str) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return np.nan
    err = actual[mask] - pred[mask]
    if metric == "SMAPE":
        return float(np.nanmean(smape_array(actual[mask], pred[mask])))
    if metric == "MAE":
        return float(np.nanmean(np.abs(err)))
    if metric == "MSE":
        return float(np.nanmean(np.square(err)))
    if metric == "RMSE":
        return float(np.sqrt(np.nanmean(np.square(err))))
    raise ValueError(metric)


def row_loss(actual: np.ndarray, pred: np.ndarray, metric: str) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred)
    err = actual - pred
    if metric == "SMAPE":
        loss = smape_array(actual, pred)
    elif metric == "MAE":
        loss = np.abs(err)
    elif metric in {"MSE", "RMSE"}:
        loss = np.square(err)
    else:
        raise ValueError(metric)
    return np.where(valid, loss, np.nan)


def make_runner_args(args: argparse.Namespace, asset: str | None = None) -> SimpleNamespace:
    ns = runner.build_arg_parser().parse_args([])
    ns.mode = "full"
    ns.phase = "all"
    ns.assets = asset or ""
    ns.seed = int(args.seed)
    ns.n = int(args.n)
    ns.warmup = int(args.warmup)
    ns.local_dir = str(args.local_dir)
    ns.pm_tau_grid = args.pm_tau_grid
    ns.pm_eta_grid = args.pm_eta_grid
    ns.pm_bandwidth_grid = args.pm_bandwidth_grid
    ns.phqo_gamma_grid = args.phqo_gamma_grid
    ns.pm_low_modes = int(args.pm_low_modes)
    ns.pm_kernel_train_window = int(args.pm_kernel_train_window)
    ns.target_transform = "level"
    ns.log_eps = float(args.log_eps)
    ns.cv_mode = "inner"
    ns.pm_tau_grid_values = runner.parse_float_grid(ns.pm_tau_grid, runner.DEFAULT_PM_TAU_GRID)
    ns.pm_eta_grid_values = runner.parse_float_grid(ns.pm_eta_grid, runner.DEFAULT_PM_ETA_GRID)
    ns.pm_bandwidth_grid_values = runner.parse_float_grid(ns.pm_bandwidth_grid, runner.DEFAULT_PM_BANDWIDTH_GRID)
    ns.phqo_gamma_grid_values = runner.parse_float_grid(ns.phqo_gamma_grid, runner.DEFAULT_PHQO_GAMMA_GRID)
    ns.lambda_grid_values = runner.parse_float_grid(ns.lambda_grid, runner.DEFAULT_LAMBDA_GRID)
    ns.lambda_r_ratio_grid_values = runner.parse_float_grid(ns.lambda_r_ratio_grid, runner.DEFAULT_LAMBDA_R_RATIO_GRID)
    ns.model_registry = runner.build_model_registry(ns.n)
    return ns


def random_character_matrix(n: int, modes: list[tuple[int, int, int]], seed: int) -> np.ndarray:
    primes = runner.pm_primes(n)
    rng = np.random.default_rng(seed)
    lags = np.arange(1, n + 1, dtype=int)
    residue_by_prime = []
    for prime in primes:
        residues = np.mod(lags, prime)
        residue_by_prime.append(rng.permutation(residues))
    mat = np.empty((n, len(modes)), dtype=np.complex128)
    for idx, mode in enumerate(modes):
        phase = np.zeros(n, dtype=float)
        for dim, prime in enumerate(primes):
            phase += float(mode[dim]) * residue_by_prime[dim] / float(prime)
        mat[:, idx] = np.exp(2j * np.pi * phase)
    return mat


def harmonic_phi_from_chars(
    x: np.ndarray,
    chars: np.ndarray,
    mu: np.ndarray,
    tau_grid: list[float],
    log_eps: float,
) -> np.ndarray:
    z = np.log(np.clip(x, 0.0, None) + log_eps)
    centered = z - z.mean(axis=1)[:, None]
    base_coeff = centered @ np.conjugate(chars)
    pieces = [base_coeff * np.exp(-float(tau) * mu)[None, :] for tau in tau_grid]
    return runner.real_embedding(np.concatenate(pieces, axis=1))


def generic_haar_shape_phi(x: np.ndarray, args: SimpleNamespace, target_dim: int) -> tuple[np.ndarray, str]:
    blocks = runner.fallback_blocks(args.n)
    weights = []
    labels = []
    for name, weight in runner.haar_weights(args.n, blocks).items():
        weights.append(np.asarray(weight, dtype=float))
        labels.append(name)
    for name, weight in runner.opsc_weights(args.n, blocks).items():
        if len(weights) >= 2 * int(args.pm_low_modes):
            break
        weights.append(np.asarray(weight, dtype=float))
        labels.append(name)
    basis_count = 2 * int(args.pm_low_modes)
    if len(weights) < basis_count:
        raise RuntimeError(f"Need {basis_count} generic shape weights, found {len(weights)}")
    w = np.column_stack(weights[:basis_count])
    norms = np.linalg.norm(w, axis=0)
    norms = np.where(np.isfinite(norms) & (norms > 1e-12), norms, 1.0)
    w = w / norms
    z = np.log(np.clip(x, 0.0, None) + args.log_eps)
    centered = z - z.mean(axis=1)[:, None]
    base = centered @ w
    _, pm_mu = runner.pm_mode_subset(args.n, int(args.pm_low_modes))
    shape_mu = np.linspace(float(np.nanmin(pm_mu)), float(np.nanmax(pm_mu)), basis_count)
    pieces = [base * np.exp(-float(tau) * shape_mu)[None, :] for tau in args.pm_tau_grid_values]
    phi = np.concatenate(pieces, axis=1)
    if phi.shape[1] < target_dim:
        phi = np.pad(phi, ((0, 0), (0, target_dim - phi.shape[1])), mode="constant")
    elif phi.shape[1] > target_dim:
        phi = phi[:, :target_dim]
    return phi, f"Haar+OPSC generic shape basis ({','.join(labels[:basis_count])})"


def placebo_embeddings(frame: pd.DataFrame, args: SimpleNamespace, seed: int) -> dict[str, tuple[np.ndarray, str]]:
    x = runner.lag_matrix(frame, args.n)
    _, _, true_phi = runner.pm_qdk2_embedding(x, args.n, args.pm_tau_grid_values, args.log_eps, int(args.pm_low_modes))
    modes, mu = runner.pm_mode_subset(args.n, int(args.pm_low_modes))
    true_chars = runner.prime_character_matrix(args.n, modes, normalize=False)
    random_chars = random_character_matrix(args.n, modes, seed + 7001)
    rng = np.random.default_rng(seed + 7002)
    shuffled_x = x[:, rng.permutation(args.n)]
    generic_phi, generic_desc = generic_haar_shape_phi(x, args, true_phi.shape[1])
    return {
        "PM_QDK_2_RANDOM_RESIDUE_PLACEBO": (
            harmonic_phi_from_chars(x, random_chars, mu, args.pm_tau_grid_values, args.log_eps),
            "Random residue character geometry preserving residue group sizes",
        ),
        "PM_QDK_2_SHUFFLED_LAG_PLACEBO": (
            harmonic_phi_from_chars(shuffled_x, true_chars, mu, args.pm_tau_grid_values, args.log_eps),
            "True prime characters applied after deterministic lag shuffle",
        ),
        "PM_QDK_2_HAAR_SHAPE_PLACEBO": (
            generic_phi,
            generic_desc,
        ),
    }


def prediction_rows(frame: pd.DataFrame, model_name: str, values: list[tuple[int, float]]) -> pd.DataFrame:
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    rows = []
    for i, pred in values:
        if i + 1 >= len(frame):
            continue
        actual = float(y_next[i])
        if not (np.isfinite(actual) and np.isfinite(pred)):
            continue
        pred = float(max(pred, 0.0))
        rows.append(
            {
                "Date": frame.index[i + 1],
                "Actual": actual,
                f"Predicted_{model_name}": pred,
                f"Err_{model_name}": actual - pred,
                f"AbsErr_{model_name}": abs(actual - pred),
                f"SMAPE_{model_name}_pct": 200.0 * abs(actual - pred) / max(1e-12, abs(actual) + abs(pred)),
            }
        )
    return pd.DataFrame(rows)


def kernel_values_from_phi(
    frame: pd.DataFrame,
    args: SimpleNamespace,
    level: np.ndarray,
    c_state: np.ndarray,
    q: np.ndarray,
    valid: np.ndarray,
    bandwidth: float,
    start_i: int,
    stop_i: int,
    max_forecasts: int = 0,
) -> list[tuple[int, float]]:
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    scale_end = max(start_i, args.n + 25)
    level_scale = float(runner.robust_scale(level[:scale_end, None])[0])
    c_scale = runner.robust_scale(c_state[:scale_end])
    q_scale = runner.robust_scale(q[:scale_end])
    h = max(float(bandwidth), 1e-12)
    values = []
    finite_q = np.isfinite(q).all(axis=1)
    train_valid = valid & finite_q
    for i in range(start_i, min(stop_i, len(frame) - 1)):
        if not (bool(train_valid[i]) and np.isfinite(y_next[i])):
            continue
        train_idx = runner.candidate_train_indices(train_valid, y_next, i, args.pm_kernel_train_window)
        if train_idx.size < 5:
            continue
        dl = np.square((level[train_idx] - level[i]) / level_scale)
        dc = (c_state[train_idx] - c_state[i]) / c_scale
        dq = (q[train_idx] - q[i]) / q_scale
        d2 = dl + np.sum(dc * dc, axis=1) + np.sum(dq * dq, axis=1) / max(q.shape[1], 1)
        weights = runner.softmax_weights(-0.5 * d2 / (h * h))
        if weights.size == 0:
            continue
        pred = runner.weighted_smape_action(y_next[train_idx], weights)
        if np.isfinite(pred):
            values.append((i, float(pred)))
        if max_forecasts and len(values) >= max_forecasts:
            break
    return values


def generate_placebo_predictions(
    frame: pd.DataFrame,
    args: SimpleNamespace,
    seed: int,
    max_forecasts: int = 0,
) -> tuple[list[PlaceboPrediction], pd.DataFrame]:
    x = runner.lag_matrix(frame, args.n)
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1)
    level, c_state, _ = runner.pm_qdk2_embedding(x, args.n, args.pm_tau_grid_values, args.log_eps, int(args.pm_low_modes))
    embeddings = placebo_embeddings(frame, args, seed)
    val_start, first_oos = runner.validation_window_indices(frame, args.n, args.warmup)
    start_i = args.n + args.warmup
    stop_i = len(frame) - 1
    predictions = []
    param_rows = []
    for model_name, (phi, desc) in embeddings.items():
        q = runner.expanding_linear_residuals(phi, c_state, valid & np.isfinite(y_next))
        best = (np.inf, float(args.pm_bandwidth_grid_values[0]))
        for bandwidth in args.pm_bandwidth_grid_values:
            vals = kernel_values_from_phi(
                frame,
                args,
                level,
                c_state,
                q,
                valid,
                float(bandwidth),
                val_start,
                first_oos,
            )
            pred = prediction_rows(frame, f"{model_name}_TMP", vals)
            if pred.empty:
                score = np.inf
            else:
                score = float(np.nanmean(pd.to_numeric(pred[f"SMAPE_{model_name}_TMP_pct"], errors="coerce")))
            if np.isfinite(score) and score < best[0]:
                best = (score, float(bandwidth))
        vals = kernel_values_from_phi(
            frame,
            args,
            level,
            c_state,
            q,
            valid,
            best[1],
            start_i,
            stop_i,
            max_forecasts=max_forecasts,
        )
        pred = prediction_rows(frame, model_name, vals)
        predictions.append(
            PlaceboPrediction(
                model_name=model_name,
                frame=pred,
                selected_bandwidth=best[1],
                cv_score=safe_float(best[0]),
                phi_dim=int(phi.shape[1]),
                basis_description=desc,
            )
        )
        param_rows.append(
            {
                "model_name": model_name,
                "selected_bandwidth": best[1],
                "cv_score_smape": safe_float(best[0]),
                "phi_dim": int(phi.shape[1]),
                "tau_grid": ",".join(str(x) for x in args.pm_tau_grid_values),
                "bandwidth_grid": ",".join(str(x) for x in args.pm_bandwidth_grid_values),
                "pm_kernel_train_window": int(args.pm_kernel_train_window),
                "basis_description": desc,
            }
        )
    return predictions, pd.DataFrame(param_rows)


def merge_prediction_frames(base: pd.DataFrame, placebos: list[PlaceboPrediction]) -> pd.DataFrame:
    out = base.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Actual"] = pd.to_numeric(out["Actual"], errors="coerce")
    for pred in placebos:
        frame = pred.frame.copy()
        if frame.empty:
            continue
        frame["Date"] = pd.to_datetime(frame["Date"])
        keep = ["Date", f"Predicted_{pred.model_name}", f"Err_{pred.model_name}", f"AbsErr_{pred.model_name}", f"SMAPE_{pred.model_name}_pct"]
        out = out.merge(frame[keep], on="Date", how="left")
    return out.sort_values("Date").reset_index(drop=True)


def subset_masks(panel: pd.DataFrame, include_prop3: bool = True) -> dict[str, pd.Series]:
    actual = pd.to_numeric(panel["Actual"], errors="coerce")
    cp_abs = pd.to_numeric(panel.get("AbsErr_CP_REPO_FRESH", pd.Series(np.nan, index=panel.index)), errors="coerce")
    masks: dict[str, pd.Series] = {
        "all_observations": pd.Series(True, index=panel.index),
    }
    near_zero_cut = max(1e-12, float(actual.quantile(0.05))) if actual.notna().any() else 1e-12
    masks["near_zero_excluded"] = actual > near_zero_cut
    masks["high_vol_only"] = actual >= actual.quantile(0.75)
    masks["low_vol_only"] = actual <= actual.quantile(0.25)
    masks["top_decile_cp_error"] = cp_abs >= cp_abs.quantile(0.90)
    if include_prop3:
        lag_base = panel[["Date", "Actual"]].dropna().drop_duplicates("Date").sort_values("Date")
        lagged = runner.add_lagged_actuals(lag_base, 10)
        if not lagged.empty:
            conditions = runner.define_conditions(lagged)
            for name, mask in conditions.items():
                if name == "all_observations":
                    continue
                dates = set(pd.to_datetime(lagged.loc[mask, "Date"]))
                masks[f"prop3_{name}"] = panel["Date"].isin(dates)
    return {name: mask.fillna(False).astype(bool) for name, mask in masks.items()}


def compute_metric_robustness(panel: pd.DataFrame, models: list[str], asset: str) -> pd.DataFrame:
    rows = []
    masks = subset_masks(panel, include_prop3=True)
    actual_all = pd.to_numeric(panel["Actual"], errors="coerce").to_numpy(dtype=float)
    cp_all = pd.to_numeric(panel["Predicted_CP_REPO_FRESH"], errors="coerce").to_numpy(dtype=float)
    for subset, mask in masks.items():
        idx = mask.to_numpy(dtype=bool)
        actual = actual_all[idx]
        cp_pred = cp_all[idx]
        for model in models:
            pred_col = f"Predicted_{model}"
            if pred_col not in panel.columns:
                continue
            pred = pd.to_numeric(panel.loc[idx, pred_col], errors="coerce").to_numpy(dtype=float)
            for metric in LOSS_METRICS:
                model_loss = metric_value(actual, pred, metric)
                cp_loss = metric_value(actual, cp_pred, metric)
                cp_row_loss = row_loss(actual, cp_pred, metric)
                model_row_loss = row_loss(actual, pred, metric)
                valid = np.isfinite(cp_row_loss) & np.isfinite(model_row_loss)
                rows.append(
                    {
                        "asset": asset,
                        "model_name": model,
                        "benchmark_model": "CP_REPO_FRESH",
                        "subset": subset,
                        "loss_metric": metric,
                        "n_obs": int(valid.sum()),
                        "cp_loss": cp_loss,
                        "model_loss": model_loss,
                        "advantage_cp_minus_model": cp_loss - model_loss if np.isfinite(cp_loss) and np.isfinite(model_loss) else np.nan,
                        "win_rate_vs_cp": float(np.nanmean(model_row_loss[valid] < cp_row_loss[valid])) if valid.any() else np.nan,
                        "mean_row_loss_diff_cp_minus_model": float(np.nanmean(cp_row_loss[valid] - model_row_loss[valid])) if valid.any() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def compute_directional_accuracy(panel: pd.DataFrame, models: list[str], asset: str) -> pd.DataFrame:
    out = []
    actual = pd.to_numeric(panel["Actual"], errors="coerce")
    prev_actual = actual.shift(1)
    actual_dir = np.sign(actual - prev_actual)
    for model in models:
        pred_col = f"Predicted_{model}"
        if pred_col not in panel.columns:
            continue
        pred = pd.to_numeric(panel[pred_col], errors="coerce")
        pred_dir = np.sign(pred - prev_actual)
        mask = np.isfinite(actual_dir) & np.isfinite(pred_dir) & (actual_dir != 0)
        out.append(
            {
                "asset": asset,
                "model_name": model,
                "n_obs": int(mask.sum()),
                "directional_change_accuracy": float(np.nanmean(pred_dir[mask] == actual_dir[mask])) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(out)


def compute_error_decomposition(panel: pd.DataFrame, models: list[str], asset: str) -> pd.DataFrame:
    rows = []
    actual = pd.to_numeric(panel["Actual"], errors="coerce").to_numpy(dtype=float)
    cp_pred = pd.to_numeric(panel["Predicted_CP_REPO_FRESH"], errors="coerce").to_numpy(dtype=float)
    for model in models:
        if model == "CP_REPO_FRESH" or f"Predicted_{model}" not in panel.columns:
            continue
        pred = pd.to_numeric(panel[f"Predicted_{model}"], errors="coerce").to_numpy(dtype=float)
        for metric in LOSS_METRICS:
            cp_loss = row_loss(actual, cp_pred, metric)
            model_loss = row_loss(actual, pred, metric)
            valid = np.isfinite(cp_loss) & np.isfinite(model_loss)
            diff_model_minus_cp = model_loss[valid] - cp_loss[valid]
            adverse = diff_model_minus_cp[diff_model_minus_cp > 0]
            adverse_sorted = np.sort(adverse)[::-1]
            adverse_total = float(adverse_sorted.sum()) if adverse_sorted.size else 0.0
            top1_n = max(1, int(math.ceil(0.01 * adverse_sorted.size))) if adverse_sorted.size else 0
            top1_share = float(adverse_sorted[:top1_n].sum() / adverse_total) if adverse_total > 0 and top1_n else np.nan
            if adverse_total > 0:
                cum = np.cumsum(adverse_sorted)
                rows_for_80pct = int(np.searchsorted(cum, 0.8 * adverse_total) + 1)
                share_rows_for_80pct = rows_for_80pct / max(len(diff_model_minus_cp), 1)
            else:
                share_rows_for_80pct = np.nan
            rows.append(
                {
                    "asset": asset,
                    "model_name": model,
                    "loss_metric": metric,
                    "n_obs": int(valid.sum()),
                    "mean_advantage_cp_minus_model": float(np.nanmean(-diff_model_minus_cp)) if valid.any() else np.nan,
                    "median_advantage_cp_minus_model": float(np.nanmedian(-diff_model_minus_cp)) if valid.any() else np.nan,
                    "percent_rows_model_beats_cp": float(np.nanmean(diff_model_minus_cp < 0)) if valid.any() else np.nan,
                    "p90_model_minus_cp_loss": float(np.nanpercentile(diff_model_minus_cp, 90)) if valid.any() else np.nan,
                    "p95_model_minus_cp_loss": float(np.nanpercentile(diff_model_minus_cp, 95)) if valid.any() else np.nan,
                    "p99_model_minus_cp_loss": float(np.nanpercentile(diff_model_minus_cp, 99)) if valid.any() else np.nan,
                    "adverse_loss_top1pct_share": top1_share,
                    "row_share_explaining_80pct_adverse_loss": share_rows_for_80pct,
                    "broad_degradation_flag": bool(np.nanmean(diff_model_minus_cp > 0) > 0.5) if valid.any() else False,
                }
            )
    return pd.DataFrame(rows)


def paired_t_pvalue(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 3:
        return np.nan
    if stats is None:
        se = np.nanstd(diff, ddof=1) / math.sqrt(diff.size)
        if not np.isfinite(se) or se <= 0:
            return np.nan
        t = float(np.nanmean(diff) / se)
        # Normal approximation fallback.
        return float(1.0 - 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))
    res = stats.ttest_1samp(diff, popmean=0.0, alternative="greater", nan_policy="omit")
    return safe_float(res.pvalue)


def sign_test_pvalue(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if diff.size == 0:
        return np.nan
    wins = int(np.sum(diff > 0))
    n = int(diff.size)
    if stats is not None:
        return safe_float(stats.binomtest(wins, n=n, p=0.5, alternative="greater").pvalue)
    # Normal approximation fallback.
    mean = n * 0.5
    sd = math.sqrt(n * 0.25)
    z = (wins - mean) / sd if sd > 0 else 0.0
    return float(1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def compute_asset_statistical_tests(panel: pd.DataFrame, models: list[str], asset: str) -> pd.DataFrame:
    rows = []
    actual = pd.to_numeric(panel["Actual"], errors="coerce").to_numpy(dtype=float)
    cp_pred = pd.to_numeric(panel["Predicted_CP_REPO_FRESH"], errors="coerce").to_numpy(dtype=float)
    for model in models:
        if model == "CP_REPO_FRESH" or f"Predicted_{model}" not in panel.columns:
            continue
        pred = pd.to_numeric(panel[f"Predicted_{model}"], errors="coerce").to_numpy(dtype=float)
        for metric in LOSS_METRICS:
            cp_loss = row_loss(actual, cp_pred, metric)
            model_loss = row_loss(actual, pred, metric)
            diff = cp_loss - model_loss
            diff = diff[np.isfinite(diff)]
            rows.append(
                {
                    "scope": "asset",
                    "asset": asset,
                    "comparison": f"{model}_vs_CP_REPO_FRESH",
                    "loss_metric": metric,
                    "n_obs": int(diff.size),
                    "mean_advantage": float(np.nanmean(diff)) if diff.size else np.nan,
                    "median_advantage": float(np.nanmedian(diff)) if diff.size else np.nan,
                    "paired_t_p_value_mean_advantage_gt_0": paired_t_pvalue(diff),
                    "sign_test_p_value_win_rate_gt_50pct": sign_test_pvalue(diff),
                    "win_rate": float(np.nanmean(diff > 0)) if diff.size else np.nan,
                }
            )
    for placebo in PLACEBO_MODELS:
        if f"Predicted_{placebo}" not in panel.columns or "Predicted_PM_QDK_2" not in panel.columns:
            continue
        pm_pred = pd.to_numeric(panel["Predicted_PM_QDK_2"], errors="coerce").to_numpy(dtype=float)
        pl_pred = pd.to_numeric(panel[f"Predicted_{placebo}"], errors="coerce").to_numpy(dtype=float)
        for metric in LOSS_METRICS:
            pm_loss = row_loss(actual, pm_pred, metric)
            pl_loss = row_loss(actual, pl_pred, metric)
            diff = pl_loss - pm_loss
            diff = diff[np.isfinite(diff)]
            rows.append(
                {
                    "scope": "asset",
                    "asset": asset,
                    "comparison": f"PM_QDK_2_vs_{placebo}",
                    "loss_metric": metric,
                    "n_obs": int(diff.size),
                    "mean_advantage": float(np.nanmean(diff)) if diff.size else np.nan,
                    "median_advantage": float(np.nanmedian(diff)) if diff.size else np.nan,
                    "paired_t_p_value_mean_advantage_gt_0": paired_t_pvalue(diff),
                    "sign_test_p_value_win_rate_gt_50pct": sign_test_pvalue(diff),
                    "win_rate": float(np.nanmean(diff > 0)) if diff.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def date_to_origin_indices(frame: pd.DataFrame, dates: pd.Series) -> np.ndarray:
    idx = frame.index.get_indexer(pd.to_datetime(dates))
    return idx - 1


def nearest_smoothness(
    c_state: np.ndarray,
    q: np.ndarray,
    residual: np.ndarray,
    origin_idx: np.ndarray,
    train_window: int,
    seed: int,
    sample_size: int = 700,
    neighbors: int = 25,
) -> dict[str, float]:
    valid = origin_idx[(origin_idx > 0) & (origin_idx < len(residual))]
    valid = valid[np.isfinite(residual[valid]) & np.isfinite(c_state[valid]).all(axis=1) & np.isfinite(q[valid]).all(axis=1)]
    if valid.size < neighbors + 20:
        return {"cp_neighbor_abs_resid_diff": np.nan, "pm_neighbor_abs_resid_diff": np.nan, "smoothness_advantage": np.nan}
    rng = np.random.default_rng(seed)
    sample = rng.choice(valid, size=min(sample_size, valid.size), replace=False)
    c_scale = runner.robust_scale(c_state[valid])
    q_scale = runner.robust_scale(q[valid])
    cp_diffs = []
    pm_diffs = []
    for i in sample:
        start = max(0, int(i) - int(train_window)) if train_window > 0 else 0
        candidates = np.arange(start, int(i))
        candidates = candidates[np.isfinite(residual[candidates]) & np.isfinite(c_state[candidates]).all(axis=1) & np.isfinite(q[candidates]).all(axis=1)]
        if candidates.size < neighbors:
            continue
        dc = (c_state[candidates] - c_state[i]) / c_scale
        cp_d2 = np.sum(dc * dc, axis=1)
        dq = (q[candidates] - q[i]) / q_scale
        pm_d2 = cp_d2 + np.sum(dq * dq, axis=1) / max(q.shape[1], 1)
        cp_nn = candidates[np.argsort(cp_d2)[:neighbors]]
        pm_nn = candidates[np.argsort(pm_d2)[:neighbors]]
        cp_diffs.append(float(np.nanmean(np.abs(residual[cp_nn] - residual[i]))))
        pm_diffs.append(float(np.nanmean(np.abs(residual[pm_nn] - residual[i]))))
    cp_mean = float(np.nanmean(cp_diffs)) if cp_diffs else np.nan
    pm_mean = float(np.nanmean(pm_diffs)) if pm_diffs else np.nan
    return {
        "cp_neighbor_abs_resid_diff": cp_mean,
        "pm_neighbor_abs_resid_diff": pm_mean,
        "smoothness_advantage": cp_mean - pm_mean if np.isfinite(cp_mean) and np.isfinite(pm_mean) else np.nan,
    }


def theory_diagnostics_for_asset(
    panel: pd.DataFrame,
    frame: pd.DataFrame,
    args: SimpleNamespace,
    asset: str,
    placebo_param_rows: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    x = runner.lag_matrix(frame, args.n)
    valid = np.isfinite(x).all(axis=1)
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    level, c_state, pm_phi = runner.pm_qdk2_embedding(x, args.n, args.pm_tau_grid_values, args.log_eps, int(args.pm_low_modes))
    del level
    pm_q = runner.expanding_linear_residuals(pm_phi, c_state, valid & np.isfinite(y_next))
    embeddings = placebo_embeddings(frame, args, seed)
    origin_idx = date_to_origin_indices(frame, panel["Date"])
    cp_pred_by_date = pd.Series(pd.to_numeric(panel["Predicted_CP_REPO_FRESH"], errors="coerce").to_numpy(dtype=float), index=pd.to_datetime(panel["Date"]))
    cp_pred_origin = np.full(len(frame), np.nan, dtype=float)
    for date, value in cp_pred_by_date.items():
        pos = frame.index.get_indexer([date])[0] - 1
        if 0 <= pos < len(cp_pred_origin):
            cp_pred_origin[pos] = value
    residual = y_next - cp_pred_origin
    rows = []
    pm_smooth = nearest_smoothness(c_state, pm_q, residual, origin_idx, args.pm_kernel_train_window, seed + 8100)
    rows.append({"asset": asset, "diagnostic": "pm_geometry_residual_smoothness", **pm_smooth})
    for model_name, (phi, desc) in embeddings.items():
        q = runner.expanding_linear_residuals(phi, c_state, valid & np.isfinite(y_next))
        smooth = nearest_smoothness(c_state, q, residual, origin_idx, args.pm_kernel_train_window, seed + stable_seed_offset(model_name, 10000))
        rows.append({"asset": asset, "diagnostic": f"{model_name}_residual_smoothness", "basis_description": desc, **smooth})
    phqo_col = "Predicted_PHQO"
    if phqo_col in panel.columns:
        cp = pd.to_numeric(panel["Predicted_CP_REPO_FRESH"], errors="coerce").to_numpy(dtype=float)
        phqo = pd.to_numeric(panel[phqo_col], errors="coerce").to_numpy(dtype=float)
        valid_phqo = np.isfinite(cp) & np.isfinite(phqo)
        rows.append(
            {
                "asset": asset,
                "diagnostic": "phqo_safe_cp_deformation",
                "n_obs": int(valid_phqo.sum()),
                "mean_abs_phqo_minus_cp": float(np.nanmean(np.abs(phqo[valid_phqo] - cp[valid_phqo]))) if valid_phqo.any() else np.nan,
                "p99_abs_phqo_minus_cp": float(np.nanpercentile(np.abs(phqo[valid_phqo] - cp[valid_phqo]), 99)) if valid_phqo.any() else np.nan,
                "exact_or_near_cp_share_1e_12": float(np.nanmean(np.abs(phqo[valid_phqo] - cp[valid_phqo]) <= 1e-12)) if valid_phqo.any() else np.nan,
            }
        )
    for _, row in placebo_param_rows.iterrows():
        rows.append(
            {
                "asset": asset,
                "diagnostic": "placebo_matched_tuning_budget",
                "model_name": row.get("model_name"),
                "phi_dim": row.get("phi_dim"),
                "selected_bandwidth": row.get("selected_bandwidth"),
                "tau_grid": row.get("tau_grid"),
                "bandwidth_grid": row.get("bandwidth_grid"),
                "basis_description": row.get("basis_description"),
            }
        )
    return pd.DataFrame(rows)


def source_audits(source_run: Path, asset: str, frame: pd.DataFrame, feature_groups: dict[str, list[str]], args: SimpleNamespace, placebo_param_rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, fatal: bool, details: str) -> None:
        rows.append({"asset": asset, "audit_name": name, "passed": bool(passed), "fatal": bool(fatal), "details": details})

    result_dir = source_run / "results"
    repo_cp = pd.read_csv(result_dir / "repo_cp_reproduction_audit.csv") if (result_dir / "repo_cp_reproduction_audit.csv").exists() else pd.DataFrame()
    cp_pass_col = "passes" if "passes" in repo_cp.columns else "passed"
    cp_passed = (not repo_cp.empty) and cp_pass_col in repo_cp.columns and bool(repo_cp[cp_pass_col].astype(bool).all())
    add("cp_construction_reproduction", cp_passed, True, f"source rows={len(repo_cp)}")

    no_lookahead = pd.read_csv(result_dir / "no_lookahead_audit.csv") if (result_dir / "no_lookahead_audit.csv").exists() else pd.DataFrame()
    nl_passed = (not no_lookahead.empty) and "passed" in no_lookahead.columns and bool(no_lookahead["passed"].astype(bool).all())
    add("no_lookahead", nl_passed, True, f"source rows={len(no_lookahead)}")

    val_start, first_oos = runner.validation_window_indices(frame, args.n, args.warmup)
    hp_passed = val_start < first_oos and first_oos == args.n + args.warmup
    add("hyperparameter_selection_pre_oos", hp_passed, True, f"val_start={val_start}; first_oos_origin={first_oos}; first_oos_target={first_oos + 1}")

    meta = pd.read_csv(result_dir / "run_metadata.csv") if (result_dir / "run_metadata.csv").exists() else pd.DataFrame()
    grid_passed = True
    details = []
    if not meta.empty:
        for col, grid in [
            ("selected_pm_bandwidth", args.pm_bandwidth_grid_values),
            ("selected_pm_tau", args.pm_tau_grid_values),
            ("selected_pm_eta", args.pm_eta_grid_values),
            ("selected_phqo_gamma", args.phqo_gamma_grid_values),
        ]:
            if col in meta.columns:
                vals = pd.to_numeric(meta.loc[meta["asset"].astype(str) == asset, col], errors="coerce").dropna().unique()
                ok = all(any(abs(float(v) - float(g)) <= 1e-12 for g in grid) for v in vals)
                grid_passed = grid_passed and ok
                details.append(f"{col}={[float(v) for v in vals]}")
    add("selected_hyperparameters_in_declared_grids", grid_passed, True, "; ".join(details) if details else "no selected parameter rows")

    cp_cols = feature_groups.get("CP_REPO_FRESH", [])
    forbidden = [col for col in cp_cols if col.startswith(("CPB_B", "PM_", "OPS", "Z_", "GATE", "GATED", "RGATED", "RAND", "SHUF", "HAAR", "LRPM", "RSLOPE", "KOPS"))]
    allowed = all(col.startswith("RV") or col.startswith("CP_") for col in cp_cols)
    add("feature_leakage_cp_repo_fresh_contract", allowed and not forbidden, True, f"cp_feature_count={len(cp_cols)}; forbidden={forbidden[:5]}")

    align_passed = True
    align_details = []
    for model in ["PM_QDK_2", "PM_QDK"]:
        pred_col = f"Predicted_{model}"
        if pred_col not in panel.columns:
            align_passed = False
            align_details.append(f"missing {pred_col}")
            continue
        mask = pd.to_numeric(panel[pred_col], errors="coerce").notna() & pd.to_numeric(panel["Predicted_CP_REPO_FRESH"], errors="coerce").notna()
        align_details.append(f"{model}_overlap={int(mask.sum())}")
        if int(mask.sum()) != int(pd.to_numeric(panel["Predicted_CP_REPO_FRESH"], errors="coerce").notna().sum()):
            align_passed = False
    add("asset_oos_alignment_core_models", align_passed, True, "; ".join(align_details))

    pm_dim_expected = 2 * min(int(args.pm_low_modes), len(runner.prime_torus_modes(args.n, include_zero=False))) * len(args.pm_tau_grid_values)
    placebo_dims = pd.to_numeric(placebo_param_rows.get("phi_dim", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()
    matched = bool(placebo_dims) and all(dim == pm_dim_expected for dim in placebo_dims)
    add("placebo_matched_feature_count_smoothing_tuning_budget", matched, False, f"expected_phi_dim={pm_dim_expected}; placebo_dims={placebo_dims}; bandwidth_grid={args.pm_bandwidth_grid_values}")
    return pd.DataFrame(rows)


def run_asset(args: argparse.Namespace) -> None:
    if not args.asset:
        raise ValueError("--asset is required in asset mode")
    source_run = Path(args.source_run)
    source_pred = source_run / "predictions" / "full" / f"{args.asset}.csv"
    if not source_pred.exists():
        raise FileNotFoundError(source_pred)
    outdir = Path(args.outdir)
    runner_args = make_runner_args(args, args.asset)
    frame, feature_groups, _manifest, _meta, _diagnostics = runner.build_feature_frame(args.asset, runner_args)
    base = pd.read_csv(source_pred, parse_dates=["Date"])
    max_forecasts = int(args.max_forecasts or 0)
    placebos, placebo_params = generate_placebo_predictions(frame, runner_args, int(args.seed), max_forecasts=max_forecasts)
    panel = merge_prediction_frames(base, placebos)
    if max_forecasts:
        # Keep local smoke panels small while preserving normal full behavior in CI.
        first_valid_date = min((p.frame["Date"].min() for p in placebos if not p.frame.empty), default=panel["Date"].min())
        panel = panel.loc[pd.to_datetime(panel["Date"]) >= pd.Timestamp(first_valid_date)].head(max_forecasts).reset_index(drop=True)
    models = [model for model in VALIDATION_MODELS if f"Predicted_{model}" in panel.columns]
    write_csv(panel, outdir / "predictions" / "full" / f"{args.asset}_post_validation.csv")
    write_csv(placebo_params.assign(asset=args.asset), outdir / "results" / "placebo_parameter_audit.csv")
    metric = compute_metric_robustness(panel, models, args.asset)
    directional = compute_directional_accuracy(panel, models, args.asset)
    errors = compute_error_decomposition(panel, models, args.asset)
    stats_df = compute_asset_statistical_tests(panel, models, args.asset)
    theory = theory_diagnostics_for_asset(panel, frame, runner_args, args.asset, placebo_params, int(args.seed))
    audits = source_audits(source_run, args.asset, frame, feature_groups, runner_args, placebo_params, panel)
    write_csv(metric, outdir / "results" / "metric_robustness.csv")
    write_csv(directional, outdir / "results" / "directional_accuracy.csv")
    write_csv(errors, outdir / "results" / "error_decomposition.csv")
    write_csv(stats_df, outdir / "results" / "statistical_tests.csv")
    write_csv(theory, outdir / "results" / "theory_diagnostics.csv")
    write_csv(audits, outdir / "results" / "audit_summary.csv")
    failed = audits.loc[audits["fatal"].astype(bool) & ~audits["passed"].astype(bool)]
    status = "passed" if failed.empty else "fatal_audit_failed"
    (outdir / "README.md").write_text(
        "\n".join(
            [
                f"# PM-native post-validation: {args.asset}",
                "",
                f"Status: `{status}`",
                "",
                f"Source run: `{source_run}`",
                "",
                "Generated matched validation-only placebos:",
                *[f"- `{row.model_name}`: {row.basis_description}" for row in placebos],
                "",
            ]
        ),
        encoding="utf-8",
    )
    marker = "ASSET_VALIDATION_SUCCESS.txt" if failed.empty else "ASSET_VALIDATION_FATAL_AUDIT_FAILURE.txt"
    (outdir / marker).write_text(f"{args.asset} {status}\n", encoding="utf-8")
    if failed.empty:
        print(f"{args.asset} post-validation passed")
    else:
        print(f"{args.asset} fatal audit failure")
        print(failed.to_string(index=False))
        raise SystemExit(2)


def concat_artifact_csvs(artifacts_dir: Path, relpath: str) -> pd.DataFrame:
    frames = []
    for path in artifacts_dir.rglob(relpath):
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def copy_prediction_panels(artifacts_dir: Path, outdir: Path) -> None:
    for path in artifacts_dir.rglob("*_post_validation.csv"):
        dest = outdir / "predictions" / "full" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 500) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = np.nanmean(sample)
    return float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))


def aggregate_metric_rows(metric: pd.DataFrame) -> pd.DataFrame:
    if metric.empty:
        return metric
    all_rows = [metric]
    base = metric.loc[metric["subset"] == "all_observations"].copy()
    if not base.empty:
        grouped = (
            base.groupby(["model_name", "benchmark_model", "loss_metric"], dropna=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "asset": "POOLED",
                        "subset": "all_observations",
                        "n_obs": int(g["n_obs"].sum()),
                        "cp_loss": np.average(g["cp_loss"], weights=np.maximum(g["n_obs"], 1)),
                        "model_loss": np.average(g["model_loss"], weights=np.maximum(g["n_obs"], 1)),
                        "advantage_cp_minus_model": np.average(g["advantage_cp_minus_model"], weights=np.maximum(g["n_obs"], 1)),
                        "win_rate_vs_cp": np.average(g["win_rate_vs_cp"], weights=np.maximum(g["n_obs"], 1)),
                        "mean_row_loss_diff_cp_minus_model": np.average(g["mean_row_loss_diff_cp_minus_model"], weights=np.maximum(g["n_obs"], 1)),
                    }
                )
            )
            .reset_index()
        )
        equal = (
            base.groupby(["model_name", "benchmark_model", "loss_metric"], dropna=False)
            .agg(
                asset=("asset", lambda s: "EQUAL_WEIGHT_ASSET"),
                subset=("subset", "first"),
                n_obs=("n_obs", "sum"),
                cp_loss=("cp_loss", "mean"),
                model_loss=("model_loss", "mean"),
                advantage_cp_minus_model=("advantage_cp_minus_model", "mean"),
                win_rate_vs_cp=("win_rate_vs_cp", "mean"),
                mean_row_loss_diff_cp_minus_model=("mean_row_loss_diff_cp_minus_model", "mean"),
            )
            .reset_index()
        )
        all_rows.extend([grouped, equal])
    return pd.concat(all_rows, ignore_index=True, sort=False)


def placebo_comparison_from_metric(metric: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = metric.loc[metric["subset"] == "all_observations"].copy()
    for scope_asset, group in base.groupby("asset", dropna=False):
        for metric_name in LOSS_METRICS:
            pm = group.loc[(group["model_name"] == "PM_QDK_2") & (group["loss_metric"] == metric_name)]
            if pm.empty:
                continue
            pm_loss = float(pm["model_loss"].iloc[0])
            pm_n = int(pm["n_obs"].iloc[0])
            for placebo in PLACEBO_MODELS:
                pl = group.loc[(group["model_name"] == placebo) & (group["loss_metric"] == metric_name)]
                if pl.empty:
                    continue
                placebo_loss = float(pl["model_loss"].iloc[0])
                rows.append(
                    {
                        "asset": scope_asset,
                        "loss_metric": metric_name,
                        "pm_model": "PM_QDK_2",
                        "placebo_model": placebo,
                        "n_obs": min(pm_n, int(pl["n_obs"].iloc[0])),
                        "pm_qdk2_loss": pm_loss,
                        "placebo_loss": placebo_loss,
                        "advantage_pm_qdk2_minus_placebo": placebo_loss - pm_loss,
                        "pm_qdk2_beats_placebo": bool(pm_loss < placebo_loss) if np.isfinite(pm_loss) and np.isfinite(placebo_loss) else False,
                    }
                )
    return pd.DataFrame(rows)


def asset_failure_analysis(metric: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = metric.loc[(metric["subset"] == "all_observations") & (metric["model_name"] == "PM_QDK_2")].copy()
    for asset, group in base.groupby("asset", dropna=False):
        if asset in {"POOLED", "EQUAL_WEIGHT_ASSET"}:
            continue
        row = {"asset": asset}
        for metric_name in LOSS_METRICS:
            m = group.loc[group["loss_metric"] == metric_name]
            if not m.empty:
                row[f"{metric_name}_advantage_cp_minus_pm_qdk2"] = float(m["advantage_cp_minus_model"].iloc[0])
                row[f"{metric_name}_pm_qdk2_loss"] = float(m["model_loss"].iloc[0])
                row[f"{metric_name}_cp_loss"] = float(m["cp_loss"].iloc[0])
        row["failed_smape"] = bool(row.get("SMAPE_advantage_cp_minus_pm_qdk2", np.nan) <= 0)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "SMAPE_advantage_cp_minus_pm_qdk2" in out.columns:
        out = out.sort_values("SMAPE_advantage_cp_minus_pm_qdk2")
    return out


def pooled_statistical_tests_from_panels(outdir: Path, seed: int, n_boot: int) -> pd.DataFrame:
    panels = []
    for path in sorted((outdir / "predictions" / "full").glob("*_post_validation.csv")):
        asset = path.name.replace("_post_validation.csv", "")
        df = pd.read_csv(path)
        df["asset"] = asset
        panels.append(df)
    if not panels:
        return pd.DataFrame()
    all_panel = pd.concat(panels, ignore_index=True, sort=False)
    rows = []
    actual = pd.to_numeric(all_panel["Actual"], errors="coerce").to_numpy(dtype=float)
    cp_pred = pd.to_numeric(all_panel["Predicted_CP_REPO_FRESH"], errors="coerce").to_numpy(dtype=float)
    for model in [m for m in VALIDATION_MODELS if m != "CP_REPO_FRESH" and f"Predicted_{m}" in all_panel.columns]:
        pred = pd.to_numeric(all_panel[f"Predicted_{model}"], errors="coerce").to_numpy(dtype=float)
        for metric_name in LOSS_METRICS:
            cp_loss = row_loss(actual, cp_pred, metric_name)
            model_loss = row_loss(actual, pred, metric_name)
            diff = cp_loss - model_loss
            diff = diff[np.isfinite(diff)]
            ci_low, ci_high = bootstrap_ci(diff, seed + stable_seed_offset((model, metric_name)), n_boot=n_boot)
            asset_means = []
            for asset, g in all_panel.groupby("asset"):
                a = pd.to_numeric(g["Actual"], errors="coerce").to_numpy(dtype=float)
                cp = pd.to_numeric(g["Predicted_CP_REPO_FRESH"], errors="coerce").to_numpy(dtype=float)
                p = pd.to_numeric(g[f"Predicted_{model}"], errors="coerce").to_numpy(dtype=float)
                adiff = row_loss(a, cp, metric_name) - row_loss(a, p, metric_name)
                asset_means.append(float(np.nanmean(adiff)))
            eci_low, eci_high = bootstrap_ci(np.asarray(asset_means), seed + stable_seed_offset(("asset", model, metric_name)), n_boot=n_boot)
            rows.append(
                {
                    "scope": "pooled_rows",
                    "asset": "POOLED",
                    "comparison": f"{model}_vs_CP_REPO_FRESH",
                    "loss_metric": metric_name,
                    "n_obs": int(diff.size),
                    "mean_advantage": float(np.nanmean(diff)) if diff.size else np.nan,
                    "median_advantage": float(np.nanmedian(diff)) if diff.size else np.nan,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "equal_weight_asset_bootstrap_ci_low": eci_low,
                    "equal_weight_asset_bootstrap_ci_high": eci_high,
                    "paired_t_p_value_mean_advantage_gt_0": paired_t_pvalue(diff),
                    "sign_test_p_value_win_rate_gt_50pct": sign_test_pvalue(diff),
                    "win_rate": float(np.nanmean(diff > 0)) if diff.size else np.nan,
                }
            )
    for placebo in PLACEBO_MODELS:
        if f"Predicted_{placebo}" not in all_panel.columns:
            continue
        pm_pred = pd.to_numeric(all_panel["Predicted_PM_QDK_2"], errors="coerce").to_numpy(dtype=float)
        pl_pred = pd.to_numeric(all_panel[f"Predicted_{placebo}"], errors="coerce").to_numpy(dtype=float)
        for metric_name in LOSS_METRICS:
            pm_loss = row_loss(actual, pm_pred, metric_name)
            pl_loss = row_loss(actual, pl_pred, metric_name)
            diff = pl_loss - pm_loss
            diff = diff[np.isfinite(diff)]
            ci_low, ci_high = bootstrap_ci(diff, seed + stable_seed_offset(("placebo", placebo, metric_name)), n_boot=n_boot)
            rows.append(
                {
                    "scope": "pooled_rows",
                    "asset": "POOLED",
                    "comparison": f"PM_QDK_2_vs_{placebo}",
                    "loss_metric": metric_name,
                    "n_obs": int(diff.size),
                    "mean_advantage": float(np.nanmean(diff)) if diff.size else np.nan,
                    "median_advantage": float(np.nanmedian(diff)) if diff.size else np.nan,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "paired_t_p_value_mean_advantage_gt_0": paired_t_pvalue(diff),
                    "sign_test_p_value_win_rate_gt_50pct": sign_test_pvalue(diff),
                    "win_rate": float(np.nanmean(diff > 0)) if diff.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_figures(outdir: Path, metric: pd.DataFrame, errors: pd.DataFrame) -> None:
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    smape = metric.loc[
        (metric["subset"] == "all_observations")
        & (metric["loss_metric"] == "SMAPE")
        & (metric["model_name"].isin(["PM_QDK_2", "PM_QDK", "PHQO"]))
        & (~metric["asset"].isin(["POOLED", "EQUAL_WEIGHT_ASSET"]))
    ].copy()
    if not smape.empty:
        pivot = smape.pivot(index="asset", columns="model_name", values="advantage_cp_minus_model")
        ax = pivot.plot(kind="bar", figsize=(12, 5))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("CP minus model SMAPE")
        ax.set_title("SMAPE Advantage By Asset")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(fig_dir / "smape_advantage_by_asset.png", dpi=160)
        plt.close(fig)
    pm_err = errors.loc[(errors["model_name"] == "PM_QDK_2") & (errors["loss_metric"].isin(["SMAPE", "MAE", "MSE", "RMSE"]))]
    if not pm_err.empty:
        pivot = pm_err.pivot(index="asset", columns="loss_metric", values="percent_rows_model_beats_cp")
        ax = pivot.plot(kind="bar", figsize=(12, 5))
        ax.axhline(0.5, color="black", linewidth=0.8)
        ax.set_ylabel("Row win rate")
        ax.set_title("PM_QDK_2 Row Win Rates")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(fig_dir / "pm_qdk2_row_win_rates.png", dpi=160)
        plt.close(fig)


def summarize_validation(outdir: Path, metric: pd.DataFrame, placebo: pd.DataFrame, failures: pd.DataFrame, stats_df: pd.DataFrame) -> str:
    pooled_smape = metric.loc[
        (metric["asset"] == "POOLED")
        & (metric["model_name"] == "PM_QDK_2")
        & (metric["loss_metric"] == "SMAPE")
        & (metric["subset"] == "all_observations")
    ]
    smape_adv = safe_float(pooled_smape["advantage_cp_minus_model"].iloc[0]) if not pooled_smape.empty else np.nan
    asset_smape = metric.loc[
        (~metric["asset"].isin(["POOLED", "EQUAL_WEIGHT_ASSET"]))
        & (metric["model_name"] == "PM_QDK_2")
        & (metric["loss_metric"] == "SMAPE")
        & (metric["subset"] == "all_observations")
    ]
    assets_positive = int((asset_smape["advantage_cp_minus_model"] > 0).sum()) if not asset_smape.empty else 0
    n_assets = int(asset_smape["asset"].nunique()) if not asset_smape.empty else 0
    placebo_smape = placebo.loc[(placebo["asset"] == "POOLED") & (placebo["loss_metric"] == "SMAPE")]
    placebo_wins = int(placebo_smape["pm_qdk2_beats_placebo"].sum()) if not placebo_smape.empty else 0
    fatal_ok = failures.empty
    sig = stats_df.loc[
        (stats_df["scope"] == "pooled_rows")
        & (stats_df["comparison"] == "PM_QDK_2_vs_CP_REPO_FRESH")
        & (stats_df["loss_metric"] == "SMAPE")
    ]
    p_value = safe_float(sig["paired_t_p_value_mean_advantage_gt_0"].iloc[0]) if not sig.empty else np.nan
    ci_low = safe_float(sig["bootstrap_ci_low"].iloc[0]) if "bootstrap_ci_low" in sig.columns and not sig.empty else np.nan
    ci_high = safe_float(sig["bootstrap_ci_high"].iloc[0]) if "bootstrap_ci_high" in sig.columns and not sig.empty else np.nan
    consistency_floor = math.ceil(0.8 * n_assets) if n_assets else len(ASSET_UNIVERSE)
    survives = bool(fatal_ok and np.isfinite(smape_adv) and smape_adv > 0 and assets_positive >= consistency_floor and placebo_wins == len(PLACEBO_MODELS) and np.isfinite(ci_low) and ci_low > 0)
    lines = [
        "# PM-Native Geometry Post-Validation Summary",
        "",
        f"Fatal audits passed: `{fatal_ok}`.",
        f"`PM_QDK_2` pooled SMAPE advantage vs `CP_REPO_FRESH`: `{smape_adv:.6g}`.",
        f"`PM_QDK_2` positive SMAPE assets: `{assets_positive}/{n_assets}`.",
        f"`PM_QDK_2` pooled SMAPE bootstrap CI: `[{ci_low:.6g}, {ci_high:.6g}]`; paired one-sided p-value `{p_value:.6g}`.",
        f"`PM_QDK_2` beats matched SMAPE placebos pooled: `{placebo_wins}/{len(PLACEBO_MODELS)}`.",
        "",
        f"Conclusion: `PM_QDK_2` SMAPE win survives this validation: `{survives}`.",
        "",
        "Repo2 review status: ready only if fatal audits passed and validation package was copied by the promotion step.",
        "Repo3 status: keep out of repo3; this is diagnostic/internal-review material, not draft-facing.",
        "",
        "Required artifacts:",
        "- `metric_robustness.csv`",
        "- `placebo_comparison.csv`",
        "- `asset_failure_analysis.csv`",
        "- `error_decomposition.csv`",
        "- `statistical_tests.csv`",
        "- `audit_summary.csv`",
        "- `prop3_subset_performance.csv`",
        "- `theory_diagnostics.csv`",
    ]
    text = "\n".join(lines) + "\n"
    (outdir / "validation_summary.md").write_text(text, encoding="utf-8")
    (outdir / "README.md").write_text(text, encoding="utf-8")
    return text


def run_aggregate(args: argparse.Namespace) -> None:
    artifacts_dir = Path(args.artifacts_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    copy_prediction_panels(artifacts_dir, outdir)
    metric = concat_artifact_csvs(artifacts_dir, "metric_robustness.csv")
    directional = concat_artifact_csvs(artifacts_dir, "directional_accuracy.csv")
    errors = concat_artifact_csvs(artifacts_dir, "error_decomposition.csv")
    asset_stats = concat_artifact_csvs(artifacts_dir, "statistical_tests.csv")
    audits = concat_artifact_csvs(artifacts_dir, "audit_summary.csv")
    theory = concat_artifact_csvs(artifacts_dir, "theory_diagnostics.csv")
    params = concat_artifact_csvs(artifacts_dir, "placebo_parameter_audit.csv")

    metric_full = aggregate_metric_rows(metric)
    placebo = placebo_comparison_from_metric(metric_full)
    failures = audits.loc[audits.get("fatal", pd.Series(False, index=audits.index)).astype(bool) & ~audits.get("passed", pd.Series(False, index=audits.index)).astype(bool)] if not audits.empty else pd.DataFrame()
    failure_analysis = asset_failure_analysis(metric_full)
    pooled_stats = pooled_statistical_tests_from_panels(outdir, int(args.seed), int(args.bootstrap_samples))
    stats_all = pd.concat([asset_stats, pooled_stats], ignore_index=True, sort=False) if not asset_stats.empty else pooled_stats
    prop3 = metric_full.loc[metric_full["subset"].astype(str).str.startswith("prop3_")].copy() if not metric_full.empty else pd.DataFrame()

    write_csv(metric_full, outdir / "metric_robustness.csv")
    write_csv(placebo, outdir / "placebo_comparison.csv")
    write_csv(failure_analysis, outdir / "asset_failure_analysis.csv")
    write_csv(errors, outdir / "error_decomposition.csv")
    write_csv(stats_all, outdir / "statistical_tests.csv")
    write_csv(audits, outdir / "audit_summary.csv")
    write_csv(prop3, outdir / "prop3_subset_performance.csv")
    write_csv(theory, outdir / "theory_diagnostics.csv")
    write_csv(directional, outdir / "directional_accuracy.csv")
    write_csv(params, outdir / "placebo_parameter_audit.csv")
    make_figures(outdir, metric_full, errors)
    summary = summarize_validation(outdir, metric_full, placebo, failures, stats_all)
    if failures.empty:
        (outdir / "POST_VALIDATION_SUCCESS.txt").write_text("post validation passed\n", encoding="utf-8")
    else:
        report = outdir / "failure_report.md"
        report.write_text("# PM-native post-validation fatal audit failure\n\n" + failures.to_markdown(index=False) + "\n", encoding="utf-8")
        print(summary)
        print(failures.to_string(index=False))
        raise SystemExit(2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PM-native geometry warmup-600 results")
    parser.add_argument("--mode", choices=["asset", "aggregate"], required=True)
    parser.add_argument("--asset", default="")
    parser.add_argument("--assets", default=",".join(ASSET_UNIVERSE))
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--artifacts-dir", default="")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=22)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--local-dir", default=str(runner.DEFAULT_LOCAL_DIR))
    parser.add_argument("--pm-tau-grid", default=runner.DEFAULT_PM_TAU_GRID)
    parser.add_argument("--pm-eta-grid", default=runner.DEFAULT_PM_ETA_GRID)
    parser.add_argument("--pm-bandwidth-grid", default=runner.DEFAULT_PM_BANDWIDTH_GRID)
    parser.add_argument("--phqo-gamma-grid", default=runner.DEFAULT_PHQO_GAMMA_GRID)
    parser.add_argument("--pm-low-modes", type=int, default=runner.DEFAULT_PM_LOW_MODES)
    parser.add_argument("--pm-kernel-train-window", type=int, default=runner.DEFAULT_PM_KERNEL_TRAIN_WINDOW)
    parser.add_argument("--log-eps", type=float, default=runner.LOG_EPS_DEFAULT)
    parser.add_argument("--max-forecasts", type=int, default=0, help="Debug cap for local smoke only; CI uses 0/full.")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "asset":
        run_asset(args)
    elif args.mode == "aggregate":
        if not args.artifacts_dir:
            raise ValueError("--artifacts-dir is required in aggregate mode")
        run_aggregate(args)
    else:  # pragma: no cover
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
