#!/usr/bin/env python
"""Tune, run, and aggregate PM-CAST-D without using final OOS outcomes for selection."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER_DIR = PROJECT_ROOT / "outputs" / "cp_repo_ops_hg_tests" / "scripts"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import run_cp_repo_ops_hg_tests as base  # noqa: E402


DEFAULT_ASSETS = ["AAPL", "AMZN", "EEM", "FXI", "GLD", "GOOGL", "HYG", "QQQ", "SPY", "TLT"]
MODEL_FAMILIES = {
    "PM_CAST_D": "pm",
    "SPECTRAL_RANDOM_CAST_D": "spectral_random",
    "CONTIGUOUS_CAST_D": "contiguous",
    "CAST_D_CALIBRATION_ONLY": "calibration_only",
}
ALL_MODELS = ["CP_REPO_FRESH", *MODEL_FAMILIES]
METRICS = ["SMAPE", "MAE", "MSE", "RMSE"]
TAIL_RULES = {
    "none": (1.1, 1.0),
    "q90_half": (0.90, 0.5),
    "q90_zero": (0.90, 0.0),
}
EPS = 1e-12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def runner_args(n: int, warmup: int, seed: int):
    args = base.build_arg_parser().parse_args([])
    args.n = int(n)
    args.warmup = int(warmup)
    args.seed = int(seed)
    args.model_registry = base.build_model_registry(args.n)
    args.pm_tau_grid_values = base.parse_float_grid(args.pm_tau_grid, base.DEFAULT_PM_TAU_GRID)
    args.pm_eta_grid_values = base.parse_float_grid(args.pm_eta_grid, base.DEFAULT_PM_ETA_GRID)
    args.pm_bandwidth_grid_values = base.parse_float_grid(args.pm_bandwidth_grid, base.DEFAULT_PM_BANDWIDTH_GRID)
    args.phqo_gamma_grid_values = base.parse_float_grid(args.phqo_gamma_grid, base.DEFAULT_PHQO_GAMMA_GRID)
    args.lambda_grid_values = base.parse_float_grid(args.lambda_grid, base.DEFAULT_LAMBDA_GRID)
    args.lambda_r_ratio_grid_values = base.parse_float_grid(args.lambda_r_ratio_grid, base.DEFAULT_LAMBDA_R_RATIO_GRID)
    return args


def pm_heat_operator(n: int, tau: float, low_modes: int) -> tuple[np.ndarray, np.ndarray]:
    c_mat, m_c = base.cp_orthogonal_projection(n)
    modes, mu = base.pm_mode_subset(n, low_modes)
    psi = base.prime_character_matrix(n, modes, normalize=True)
    heat = np.real(psi @ np.diag(np.exp(-float(tau) * mu)) @ np.conjugate(psi).T @ m_c)
    return heat, c_mat


def _nonzero_svd(operator: np.ndarray) -> tuple[np.ndarray, int]:
    singular = np.linalg.svd(operator, compute_uv=False)
    if singular.size == 0:
        return singular, 0
    rank = int(np.sum(singular > max(operator.shape) * singular[0] * np.finfo(float).eps))
    return singular[:rank], rank


def spectral_random_operator(pm_operator: np.ndarray, seed: int) -> np.ndarray:
    singular, rank = _nonzero_svd(pm_operator)
    if rank == 0:
        return np.zeros_like(pm_operator)
    _, m_c = base.cp_orthogonal_projection(pm_operator.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(m_c)
    quotient_basis = eigenvectors[:, eigenvalues > 0.5]
    rng = np.random.default_rng(int(seed))
    left = np.linalg.qr(rng.normal(size=(pm_operator.shape[0], rank)))[0][:, :rank]
    right_coeff = np.linalg.qr(rng.normal(size=(quotient_basis.shape[1], rank)))[0][:, :rank]
    right = quotient_basis @ right_coeff
    return left @ np.diag(singular) @ right.T


def contiguous_operator(pm_operator: np.ndarray) -> np.ndarray:
    singular, rank = _nonzero_svd(pm_operator)
    n = pm_operator.shape[0]
    if rank == 0:
        return np.zeros_like(pm_operator)
    laplacian = np.zeros((n, n), dtype=float)
    for idx in range(n - 1):
        laplacian[idx, idx] += 1.0
        laplacian[idx + 1, idx + 1] += 1.0
        laplacian[idx, idx + 1] -= 1.0
        laplacian[idx + 1, idx] -= 1.0
    _, path_modes = np.linalg.eigh(laplacian)
    left = path_modes[:, 1 : rank + 1]
    _, m_c = base.cp_orthogonal_projection(n)
    projected = m_c @ path_modes[:, 1:]
    right, _, _ = np.linalg.svd(projected, full_matrices=False)
    right = right[:, :rank]
    return left @ np.diag(singular) @ right.T


def operator_for_family(family: str, n: int, tau: float, low_modes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pm_operator, c_mat = pm_heat_operator(n, tau, low_modes)
    if family == "pm":
        return pm_operator, c_mat
    if family == "spectral_random":
        return spectral_random_operator(pm_operator, seed + int(round(tau * 10000))), c_mat
    if family == "contiguous":
        return contiguous_operator(pm_operator), c_mat
    if family == "calibration_only":
        return np.zeros_like(pm_operator), c_mat
    raise ValueError(f"Unknown operator family: {family}")


def expanding_prior_rank(values: np.ndarray) -> np.ndarray:
    ranks = np.full(len(values), 0.5, dtype=float)
    prior: list[float] = []
    for idx, value in enumerate(np.asarray(values, dtype=float)):
        if np.isfinite(value) and prior:
            ranks[idx] = bisect.bisect_right(prior, float(value)) / len(prior)
        if np.isfinite(value):
            bisect.insort(prior, float(value))
    return ranks


def transport_log_ratio(
    x: np.ndarray,
    operator: np.ndarray,
    kappa: float,
    gamma: float,
    return_weight_audit: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x).all(axis=1)
    safe_x = np.where(valid[:, None], np.clip(x, 0.0, None), 0.0)
    z = np.log(safe_x + EPS)
    z -= z.mean(axis=1, keepdims=True)
    energy = z @ operator.T
    scalar_scale = np.sqrt(np.mean(np.square(energy), axis=1))
    normalized = energy / np.maximum(scalar_scale[:, None], EPS)
    log_beta = -float(kappa) * np.arange(x.shape[1], dtype=float)
    log_beta -= np.max(log_beta)
    beta = np.exp(log_beta)
    beta /= beta.sum()
    logits = np.clip(float(gamma) * normalized, -20.0, 20.0) + np.log(beta)[None, :]
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    transported = np.sum(weights * safe_x, axis=1)
    reference = safe_x @ beta
    ratio = np.log((transported + EPS) / (reference + EPS))
    ratio[~valid] = np.nan
    audit = {}
    if return_weight_audit:
        audit = {
            "max_abs_weight_sum_error": float(np.max(np.abs(weights[valid].sum(axis=1) - 1.0))) if valid.any() else np.nan,
            "minimum_weight": float(np.min(weights[valid])) if valid.any() else np.nan,
            "mean_weight_entropy": float(np.mean(-np.sum(weights[valid] * np.log(np.maximum(weights[valid], EPS)), axis=1))) if valid.any() else np.nan,
        }
    return ratio, audit


def apply_action(cp: np.ndarray, log_ratio: np.ndarray, prior_rank: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    cp = np.asarray(cp, dtype=float)
    log_ratio = np.asarray(log_ratio, dtype=float)
    threshold, tail_factor = TAIL_RULES[str(config["tail_rule"])]
    gate = np.where(np.asarray(prior_rank, dtype=float) >= threshold, tail_factor, 1.0)
    correction = float(config["intercept"]) + float(config["alpha"]) * log_ratio
    correction = gate * np.clip(correction, -float(config["clip"]), float(config["clip"]))
    pred = cp.copy()
    valid = np.isfinite(cp) & np.isfinite(log_ratio) & (cp > 0.0)
    pred[valid] = cp[valid] * np.exp(correction[valid])
    return pred, correction


def loss_values(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[valid], pred[valid]
    error = actual - pred
    mse = float(np.mean(np.square(error)))
    return {
        "SMAPE": float(np.mean(200.0 * np.abs(error) / np.maximum(np.abs(actual) + np.abs(pred), EPS))),
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
    }


def build_asset_data(asset: str, args, history_warmup: int, stop_origin: int | None = None) -> dict:
    frame, feature_groups, _, _, _ = base.build_feature_frame(asset, args)
    x = base.lag_matrix(frame, args.n)
    actual = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    cp_map = base.cp_prediction_map(frame, feature_groups, args, None, warmup_override=history_warmup)
    origins = []
    dates = []
    cp_values = []
    max_origin = len(frame) - 1 if stop_origin is None else min(int(stop_origin), len(frame) - 1)
    for origin in range(args.n + int(history_warmup), max_origin):
        date = pd.Timestamp(frame.index[origin + 1])
        cp_value = cp_map.get(date, np.nan)
        if np.isfinite(cp_value) and np.isfinite(actual[origin]) and np.isfinite(x[origin]).all():
            origins.append(origin)
            dates.append(date)
            cp_values.append(float(cp_value))
    origin_array = np.asarray(origins, dtype=int)
    return {
        "asset": asset,
        "frame": frame,
        "feature_groups": feature_groups,
        "origin": origin_array,
        "Date": pd.to_datetime(dates),
        "Actual": actual[origin_array],
        "CP": np.asarray(cp_values, dtype=float),
        "X": x[origin_array],
        "prior_rank": expanding_prior_rank(np.asarray(cp_values, dtype=float)),
    }


def fold_ids(origins: np.ndarray, eval_start: int, first_oos: int, folds: int) -> np.ndarray:
    width = max(1, int(math.ceil((first_oos - eval_start) / folds)))
    return np.clip((np.asarray(origins) - eval_start) // width, 0, folds - 1).astype(int)


def candidate_score(actual: np.ndarray, cp: np.ndarray, pred: np.ndarray, folds: np.ndarray, n_folds: int) -> dict:
    cp_all = loss_values(actual, cp)
    model_all = loss_values(actual, pred)
    pooled_norm = {metric: (cp_all[metric] - model_all[metric]) / max(abs(cp_all[metric]), EPS) for metric in ["SMAPE", "MAE", "MSE"]}
    fold_norm: dict[str, list[float]] = {metric: [] for metric in ["SMAPE", "MAE", "MSE"]}
    for fold in range(n_folds):
        mask = folds == fold
        if not mask.any():
            continue
        cp_loss = loss_values(actual[mask], cp[mask])
        model_loss = loss_values(actual[mask], pred[mask])
        for metric in fold_norm:
            fold_norm[metric].append((cp_loss[metric] - model_loss[metric]) / max(abs(cp_loss[metric]), EPS))
    robust = {
        metric: float(np.mean(values) - 0.25 * np.std(values, ddof=0)) if values else -np.inf
        for metric, values in fold_norm.items()
    }
    all_fold_values = [value for values in fold_norm.values() for value in values]
    return {
        "selection_score": float(min(robust.values())),
        "pooled_min_normalized_gain": float(min(pooled_norm.values())),
        "worst_fold_metric_normalized_gain": float(min(all_fold_values)) if all_fold_values else -np.inf,
        "positive_fold_metric_count": int(sum(value > 0 for value in all_fold_values)),
        "fold_metric_count": int(len(all_fold_values)),
        **{f"pooled_{metric.lower()}_gain": cp_all[metric] - model_all[metric] for metric in ["SMAPE", "MAE", "MSE"]},
        **{f"pooled_{metric.lower()}_normalized_gain": pooled_norm[metric] for metric in ["SMAPE", "MAE", "MSE"]},
    }


def zero_inflated_residual_calibration(actual: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred) & (actual >= 0.0) & (pred > 0.0)
    actual, pred = actual[valid], pred[valid]
    if actual.size == 0:
        raise ValueError("No valid pre-OOS rows for distribution calibration")
    zero_probability = float(np.mean(actual <= EPS))
    positive = actual > EPS
    positive_residual = np.log((actual[positive] + EPS) / (pred[positive] + EPS))
    if positive_residual.size == 0:
        multipliers = {probability: 0.0 for probability in [0.05, 0.50, 0.95]}
        positive_mean = np.nan
        positive_std = np.nan
    else:
        multipliers = {}
        for probability in [0.05, 0.50, 0.95]:
            if probability <= zero_probability:
                multipliers[probability] = 0.0
            else:
                conditional_probability = (probability - zero_probability) / max(1.0 - zero_probability, EPS)
                conditional_probability = float(np.clip(conditional_probability, 0.0, 1.0))
                multipliers[probability] = float(np.exp(np.quantile(positive_residual, conditional_probability)))
        positive_mean = float(np.mean(positive_residual))
        positive_std = float(np.std(positive_residual, ddof=1)) if positive_residual.size > 1 else 0.0
    return {
        "n": int(actual.size),
        "n_positive": int(positive.sum()),
        "zero_probability": zero_probability,
        "multiplier_q05": multipliers[0.05],
        "multiplier_q50": multipliers[0.50],
        "multiplier_q95": multipliers[0.95],
        "positive_log_residual_mean": positive_mean,
        "positive_log_residual_std": positive_std,
    }
def tune(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    assets = parse_csv(args.assets)
    runner = runner_args(args.n, args.warmup, args.seed)
    first_oos = args.n + args.warmup
    eval_start = args.n + args.tuning_min_history
    data = {
        asset: build_asset_data(asset, runner, args.history_warmup, stop_origin=first_oos)
        for asset in assets
    }
    manifest_rows = []
    for asset, panel in data.items():
        eval_mask = (panel["origin"] >= eval_start) & (panel["origin"] < first_oos)
        manifest_rows.append(
            {
                "asset": asset,
                "history_first_origin": int(panel["origin"].min()),
                "tuning_first_origin": int(panel["origin"][eval_mask].min()),
                "tuning_last_origin": int(panel["origin"][eval_mask].max()),
                "first_oos_origin": first_oos,
                "n_tuning_rows": int(eval_mask.sum()),
                "strictly_pre_oos": bool(panel["origin"][eval_mask].max() < first_oos),
            }
        )
    write_csv(pd.DataFrame(manifest_rows), outdir / "tuning_data_manifest.csv")

    tau_grid = parse_csv(args.tau_grid, float)
    kappa_grid = parse_csv(args.kappa_grid, float)
    gamma_grid = parse_csv(args.gamma_grid, float)
    intercept_grid = parse_csv(args.intercept_grid, float)
    alpha_grid = parse_csv(args.alpha_grid, float)
    clip_grid = parse_csv(args.clip_grid, float)
    tail_rules = parse_csv(args.tail_rules)
    candidate_rows = []
    selected: dict[str, dict] = {}

    for model_name, family in MODEL_FAMILIES.items():
        family_rows = []
        family_taus = [tau_grid[0]] if family == "calibration_only" else tau_grid
        family_kappas = [kappa_grid[0]] if family == "calibration_only" else kappa_grid
        family_gammas = [0.0] if family == "calibration_only" else gamma_grid
        for tau in family_taus:
            operator, _ = operator_for_family(family, args.n, tau, args.low_modes, args.seed)
            for kappa in family_kappas:
                for gamma in family_gammas:
                    parts = []
                    for asset, panel in data.items():
                        if family == "calibration_only":
                            log_ratio = np.zeros(len(panel["origin"]), dtype=float)
                        else:
                            log_ratio, _ = transport_log_ratio(panel["X"], operator, kappa, gamma)
                        mask = (panel["origin"] >= eval_start) & (panel["origin"] < first_oos)
                        parts.append(
                            pd.DataFrame(
                                {
                                    "asset": asset,
                                    "origin": panel["origin"][mask],
                                    "Actual": panel["Actual"][mask],
                                    "CP": panel["CP"][mask],
                                    "log_ratio": log_ratio[mask],
                                    "prior_rank": panel["prior_rank"][mask],
                                    "fold": fold_ids(panel["origin"][mask], eval_start, first_oos, args.folds),
                                }
                            )
                        )
                    joined = pd.concat(parts, ignore_index=True)
                    actual = joined["Actual"].to_numpy(dtype=float)
                    cp = joined["CP"].to_numpy(dtype=float)
                    ratio = joined["log_ratio"].to_numpy(dtype=float)
                    ranks = joined["prior_rank"].to_numpy(dtype=float)
                    folds = joined["fold"].to_numpy(dtype=int)
                    for intercept in intercept_grid:
                        for alpha in alpha_grid:
                            for clip in clip_grid:
                                for tail_rule in tail_rules:
                                    config = {
                                        "family": family,
                                        "tau": tau,
                                        "kappa": kappa,
                                        "gamma": gamma,
                                        "intercept": intercept,
                                        "alpha": alpha,
                                        "clip": clip,
                                        "tail_rule": tail_rule,
                                    }
                                    pred, _ = apply_action(cp, ratio, ranks, config)
                                    score = candidate_score(actual, cp, pred, folds, args.folds)
                                    row = {"model_name": model_name, **config, **score}
                                    family_rows.append(row)
        family_frame = pd.DataFrame(family_rows).sort_values(
            ["selection_score", "pooled_min_normalized_gain", "worst_fold_metric_normalized_gain", "alpha"],
            ascending=[False, False, False, True],
        )
        best = family_frame.iloc[0].to_dict()
        selected[model_name] = {
            key: (float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, np.integer) else value)
            for key, value in best.items()
            if key not in {"model_name"}
        }
        candidate_rows.append(family_frame)

    candidates = pd.concat(candidate_rows, ignore_index=True)
    write_csv(candidates, outdir / "tuning_candidates.csv")
    pm_config = selected["PM_CAST_D"]
    pm_operator, _ = operator_for_family(
        "pm", args.n, float(pm_config["tau"]), args.low_modes, args.seed
    )
    distribution_calibration = {}
    for asset, panel in data.items():
        log_ratio, _ = transport_log_ratio(
            panel["X"], pm_operator, float(pm_config["kappa"]), float(pm_config["gamma"])
        )
        pred, _ = apply_action(panel["CP"], log_ratio, panel["prior_rank"], pm_config)
        mask = (
            (panel["origin"] >= eval_start)
            & (panel["origin"] < first_oos)
            & np.isfinite(panel["Actual"])
            & np.isfinite(pred)
            & (panel["Actual"] >= 0.0)
            & (pred > 0.0)
        )
        distribution_calibration[asset] = zero_inflated_residual_calibration(
            panel["Actual"][mask], pred[mask]
        )
    config = {
        "model_name": "PM_CAST_D",
        "created_utc": utc_now(),
        "n": args.n,
        "warmup": args.warmup,
        "seed": args.seed,
        "low_modes": args.low_modes,
        "history_warmup": args.history_warmup,
        "tuning_min_history": args.tuning_min_history,
        "tuning_first_origin": eval_start,
        "tuning_last_origin": first_oos - 1,
        "first_oos_origin": first_oos,
        "assets": assets,
        "folds": args.folds,
        "selection_rule": "maximize min_metric(mean_fold_normalized_gain - 0.25*fold_sd)",
        "models": selected,
        "distribution_calibration": distribution_calibration,
        "grids": {
            "tau": tau_grid,
            "kappa": kappa_grid,
            "gamma": gamma_grid,
            "intercept": intercept_grid,
            "alpha": alpha_grid,
            "clip": clip_grid,
            "tail_rules": tail_rules,
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (outdir / "frozen_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (outdir / "TUNING_COMPLETE.txt").write_text("PM-CAST-D tuning used strictly pre-OOS origins only\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in cfg.items() if key in {"family", "tau", "kappa", "gamma", "intercept", "alpha", "clip", "tail_rule", "selection_score"}} for name, cfg in selected.items()}, indent=2))


def scalar_transport_log_ratio(x: np.ndarray, operator: np.ndarray, kappa: float, gamma: float) -> float:
    values, _ = transport_log_ratio(np.asarray(x, dtype=float)[None, :], operator, kappa, gamma)
    return float(values[0])


def run_asset(args) -> None:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if int(config["warmup"]) != int(args.warmup):
        raise ValueError(f"Config warmup {config['warmup']} does not match requested warmup {args.warmup}")
    runner = runner_args(args.n, args.warmup, args.seed)
    panel = build_asset_data(args.asset, runner, int(config["history_warmup"]))
    first_oos = args.n + args.warmup
    output_mask = panel["origin"] >= first_oos
    output = pd.DataFrame(
        {
            "asset": args.asset,
            "origin": panel["origin"][output_mask],
            "Date": panel["Date"][output_mask],
            "Actual": panel["Actual"][output_mask],
            "Predicted_CP_REPO_FRESH": panel["CP"][output_mask],
        }
    )
    audit_rows = []
    operators: dict[str, np.ndarray] = {}
    for model_name, family in MODEL_FAMILIES.items():
        model_config = config["models"][model_name]
        operator, c_mat = operator_for_family(family, args.n, float(model_config["tau"]), int(config["low_modes"]), args.seed)
        operators[model_name] = operator
        if family == "calibration_only":
            log_ratio = np.zeros(len(panel["origin"]), dtype=float)
            weight_audit = {"max_abs_weight_sum_error": 0.0, "minimum_weight": np.nan, "mean_weight_entropy": np.nan}
        else:
            log_ratio, weight_audit = transport_log_ratio(
                panel["X"], operator, float(model_config["kappa"]), float(model_config["gamma"]), return_weight_audit=True
            )
        pred, correction = apply_action(panel["CP"], log_ratio, panel["prior_rank"], model_config)
        output[f"Predicted_{model_name}"] = pred[output_mask]
        output[f"LogTransport_{model_name}"] = log_ratio[output_mask]
        output[f"LogCorrection_{model_name}"] = correction[output_mask]
        cp_null = float(np.max(np.abs(operator @ c_mat.T)))
        sample_indices = np.flatnonzero(output_mask)[: min(8, int(output_mask.sum()))]
        scalar = np.array(
            [scalar_transport_log_ratio(panel["X"][idx], operator, float(model_config["kappa"]), float(model_config["gamma"])) for idx in sample_indices]
        ) if family != "calibration_only" else np.zeros(len(sample_indices))
        fast_diff = float(np.max(np.abs(scalar - log_ratio[sample_indices]))) if len(sample_indices) else 0.0
        audit_rows.extend(
            [
                {"asset": args.asset, "model_name": model_name, "check": "finite_predictions", "value": int(np.isfinite(pred[output_mask]).all()), "threshold": 1.0, "passed": bool(np.isfinite(pred[output_mask]).all())},
                {"asset": args.asset, "model_name": model_name, "check": "cp_quotient_annihilation", "value": cp_null, "threshold": 1e-10, "passed": cp_null <= 1e-10},
                {"asset": args.asset, "model_name": model_name, "check": "weight_sum", "value": weight_audit["max_abs_weight_sum_error"], "threshold": 1e-12, "passed": weight_audit["max_abs_weight_sum_error"] <= 1e-12},
                {"asset": args.asset, "model_name": model_name, "check": "fast_scalar_equivalence", "value": fast_diff, "threshold": 1e-12, "passed": fast_diff <= 1e-12},
            ]
        )

    distribution = config["distribution_calibration"][args.asset]
    pm_point = output["Predicted_PM_CAST_D"].to_numpy(dtype=float)
    lower = pm_point.copy()
    median = pm_point.copy()
    upper = pm_point.copy()
    positive = np.isfinite(pm_point) & (pm_point > 0.0)
    lower[positive] = pm_point[positive] * float(distribution["multiplier_q05"])
    median[positive] = pm_point[positive] * float(distribution["multiplier_q50"])
    upper[positive] = pm_point[positive] * float(distribution["multiplier_q95"])
    output["Predicted_PM_CAST_D_Q05"] = lower
    output["Predicted_PM_CAST_D_Q50"] = median
    output["Predicted_PM_CAST_D_Q95"] = upper
    interval_pass = bool(
        np.isfinite(lower).all()
        and np.isfinite(median).all()
        and np.isfinite(upper).all()
        and np.all(lower <= median)
        and np.all(median <= upper)
    )
    audit_rows.append(
        {
            "asset": args.asset,
            "model_name": "PM_CAST_D",
            "check": "distribution_interval_monotone_finite",
            "value": int(interval_pass),
            "threshold": 1.0,
            "passed": interval_pass,
        }
    )

    cp = panel["CP"][output_mask]
    nested, _ = apply_action(cp, np.zeros_like(cp), panel["prior_rank"][output_mask], {"intercept": 0.0, "alpha": 0.0, "clip": 0.1, "tail_rule": "none"})
    nesting_diff = float(np.max(np.abs(nested - cp))) if len(cp) else 0.0
    tuning_last = int(config["tuning_last_origin"])
    audit_rows.extend(
        [
            {"asset": args.asset, "model_name": "PM_CAST_D", "check": "exact_cp_nesting", "value": nesting_diff, "threshold": 0.0, "passed": nesting_diff == 0.0},
            {"asset": args.asset, "model_name": "PM_CAST_D", "check": "tuning_strictly_pre_oos", "value": tuning_last, "threshold": first_oos - 1, "passed": tuning_last < first_oos},
            {"asset": args.asset, "model_name": "PM_CAST_D", "check": "timestamp_unique", "value": int(output["Date"].duplicated().sum()), "threshold": 0.0, "passed": not output["Date"].duplicated().any()},
        ]
    )
    for control in ["SPECTRAL_RANDOM_CAST_D", "CONTIGUOUS_CAST_D"]:
        control_tau = float(config["models"][control]["tau"])
        matched_pm, _ = pm_heat_operator(args.n, control_tau, int(config["low_modes"]))
        matched_singular = np.linalg.svd(matched_pm, compute_uv=False)
        difference = float(np.max(np.abs(matched_singular - np.linalg.svd(operators[control], compute_uv=False))))
        audit_rows.append({"asset": args.asset, "model_name": control, "check": "singular_spectrum_match", "value": difference, "threshold": 1e-10, "passed": difference <= 1e-10})

    final_cp = base.cp_prediction_map(panel["frame"], panel["feature_groups"], runner, None, warmup_override=args.warmup)
    direct_cp = np.array([final_cp.get(pd.Timestamp(date), np.nan) for date in output["Date"]], dtype=float)
    cp_diff = float(np.nanmax(np.abs(direct_cp - output["Predicted_CP_REPO_FRESH"].to_numpy(dtype=float))))
    audit_rows.append({"asset": args.asset, "model_name": "CP_REPO_FRESH", "check": "history_warmup_cp_invariance", "value": cp_diff, "threshold": 1e-12, "passed": cp_diff <= 1e-12})

    outdir = Path(args.outdir)
    write_csv(output, outdir / "predictions" / f"{args.asset}.csv")
    asset_metric_rows = []
    for model_name in ALL_MODELS:
        losses = loss_values(output["Actual"].to_numpy(dtype=float), output[f"Predicted_{model_name}"].to_numpy(dtype=float))
        for metric, loss in losses.items():
            asset_metric_rows.append({"asset": args.asset, "model_name": model_name, "loss_metric": metric, "n_obs": len(output), "loss": loss})
    write_csv(pd.DataFrame(asset_metric_rows), outdir / "results" / "asset_metrics.csv")
    audits = pd.DataFrame(audit_rows)
    write_csv(audits, outdir / "results" / "audit.csv")
    metadata = pd.DataFrame(
        [
            {
                "asset": args.asset,
                "n": args.n,
                "warmup": args.warmup,
                "seed": args.seed,
                "n_oos": len(output),
                "first_oos_origin": first_oos,
                "tuning_last_origin": tuning_last,
                "all_audits_pass": bool(audits["passed"].all()),
                "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "created_utc": utc_now(),
            }
        ]
    )
    write_csv(metadata, outdir / "results" / "metadata.csv")
    if not bool(audits["passed"].all()):
        raise RuntimeError(f"{args.asset}: PM-CAST-D audit failure")
    (outdir / "ASSET_COMPLETE.txt").write_text(f"{args.asset} PM-CAST-D run complete\n", encoding="utf-8")
    print(f"{args.asset}: wrote {len(output)} OOS rows")


def discover_asset_panels(artifacts: Path, assets: list[str]) -> dict[str, Path]:
    found = {}
    for path in artifacts.rglob("predictions/*.csv"):
        if path.stem in assets and path.stem not in found:
            found[path.stem] = path
    return found


def metric_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    groups = list(panel.groupby("asset", sort=True)) + [("POOLED", panel)]
    for asset, group in groups:
        actual = group["Actual"].to_numpy(dtype=float)
        cp_losses = loss_values(actual, group["Predicted_CP_REPO_FRESH"].to_numpy(dtype=float))
        for model_name in ALL_MODELS:
            losses = loss_values(actual, group[f"Predicted_{model_name}"].to_numpy(dtype=float))
            for metric in METRICS:
                rows.append(
                    {
                        "asset": asset,
                        "model_name": model_name,
                        "benchmark_model": "CP_REPO_FRESH",
                        "loss_metric": metric,
                        "n_obs": len(group),
                        "CP_loss": cp_losses[metric],
                        "model_loss": losses[metric],
                        "advantage_CP_minus_model": cp_losses[metric] - losses[metric],
                        "relative_advantage": (cp_losses[metric] - losses[metric]) / max(abs(cp_losses[metric]), EPS),
                    }
                )
    metrics = pd.DataFrame(rows)
    by_asset = metrics[metrics["asset"] != "POOLED"].copy()
    consistency_rows = []
    for (model, metric), group in by_asset.groupby(["model_name", "loss_metric"]):
        advantage = group["advantage_CP_minus_model"]
        consistency_rows.append(
            {
                "model_name": model,
                "loss_metric": metric,
                "assets_positive": int((advantage > 0).sum()),
                "assets_zero": int((advantage == 0).sum()),
                "assets_negative": int((advantage < 0).sum()),
                "n_assets": int(len(group)),
            }
        )
    consistency = pd.DataFrame(consistency_rows)
    pm = metrics[metrics["model_name"] == "PM_CAST_D"]
    controls = metrics[metrics["model_name"].isin(["CAST_D_CALIBRATION_ONLY", "SPECTRAL_RANDOM_CAST_D", "CONTIGUOUS_CAST_D"])]
    comparison = pm.merge(controls, on=["asset", "loss_metric"], suffixes=("_pm", "_control"))
    comparison["advantage_PM_minus_control"] = comparison["model_loss_control"] - comparison["model_loss_pm"]
    comparison["PM_beats_control"] = comparison["advantage_PM_minus_control"] > 0
    return metrics, consistency, comparison


def distribution_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = list(panel.groupby("asset", sort=True)) + [("POOLED", panel)]
    for asset, group in groups:
        actual = group["Actual"].to_numpy(dtype=float)
        lower = group["Predicted_PM_CAST_D_Q05"].to_numpy(dtype=float)
        median = group["Predicted_PM_CAST_D_Q50"].to_numpy(dtype=float)
        upper = group["Predicted_PM_CAST_D_Q95"].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(median) & np.isfinite(upper)
        rows.append(
            {
                "asset": asset,
                "n_obs": int(valid.sum()),
                "nominal_coverage": 0.90,
                "empirical_coverage": float(np.mean((actual[valid] >= lower[valid]) & (actual[valid] <= upper[valid]))),
                "mean_interval_width": float(np.mean(upper[valid] - lower[valid])),
                "median_interval_width": float(np.median(upper[valid] - lower[valid])),
                "median_absolute_error": float(np.mean(np.abs(actual[valid] - median[valid]))),
            }
        )
    return pd.DataFrame(rows)


def daily_sufficient_statistics(panel: pd.DataFrame, model_name: str) -> dict[str, pd.DataFrame]:
    frame = panel.copy()
    frame["day"] = pd.to_datetime(frame["Date"]).dt.date
    actual = frame["Actual"].to_numpy(dtype=float)
    cp = frame["Predicted_CP_REPO_FRESH"].to_numpy(dtype=float)
    pred = frame[f"Predicted_{model_name}"].to_numpy(dtype=float)
    frame["smape_diff"] = 200.0 * np.abs(actual - cp) / np.maximum(np.abs(actual) + np.abs(cp), EPS) - 200.0 * np.abs(actual - pred) / np.maximum(np.abs(actual) + np.abs(pred), EPS)
    frame["abs_diff"] = np.abs(actual - cp) - np.abs(actual - pred)
    frame["cp_sq"] = np.square(actual - cp)
    frame["model_sq"] = np.square(actual - pred)
    grouped = frame.groupby(["asset", "day"], sort=True).agg(
        n=("Actual", "size"), smape_diff=("smape_diff", "sum"), abs_diff=("abs_diff", "sum"), cp_sq=("cp_sq", "sum"), model_sq=("model_sq", "sum")
    ).reset_index()
    return {asset: group.reset_index(drop=True) for asset, group in grouped.groupby("asset", sort=True)}


def moving_block_bootstrap(panel: pd.DataFrame, model_name: str, reps: int, block_days: int, seed: int) -> pd.DataFrame:
    daily = daily_sufficient_statistics(panel, model_name)
    assets = sorted(daily)
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(int(reps)):
        totals = {key: 0.0 for key in ["n", "smape_diff", "abs_diff", "cp_sq", "model_sq"]}
        for sampled_asset in rng.choice(assets, size=len(assets), replace=True):
            frame = daily[str(sampled_asset)]
            n_days = len(frame)
            selected = []
            while len(selected) < n_days:
                start = int(rng.integers(0, max(1, n_days - block_days + 1)))
                selected.extend(range(start, min(start + block_days, n_days)))
            sample = frame.iloc[selected[:n_days]]
            for key in totals:
                totals[key] += float(sample[key].sum())
        n = max(totals["n"], 1.0)
        draws["SMAPE"].append(totals["smape_diff"] / n)
        draws["MAE"].append(totals["abs_diff"] / n)
        draws["MSE"].append((totals["cp_sq"] - totals["model_sq"]) / n)
        draws["RMSE"].append(math.sqrt(totals["cp_sq"] / n) - math.sqrt(totals["model_sq"] / n))
    rows = []
    for metric, values in draws.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "model_name": model_name,
                "loss_metric": metric,
                "bootstrap_reps": reps,
                "block_days": block_days,
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
                "bootstrap_mean_advantage": float(array.mean()),
                "one_sided_p_advantage_le_zero": float((1 + np.sum(array <= 0)) / (len(array) + 1)),
            }
        )
    return pd.DataFrame(rows)


def write_figures(metrics: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pooled = metrics[(metrics["asset"] == "POOLED") & (metrics["model_name"] != "CP_REPO_FRESH")]
    pivot = pooled.pivot(index="model_name", columns="loss_metric", values="relative_advantage") * 100.0
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot[METRICS].plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Relative loss improvement vs CP (%)")
    ax.set_title("PM-CAST-D Practitioner Metric Advantages")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(outdir / "pooled_metric_advantages.png", dpi=180)
    plt.close(fig)

    asset = metrics[(metrics["asset"] != "POOLED") & (metrics["model_name"] == "PM_CAST_D")]
    heat = asset.pivot(index="asset", columns="loss_metric", values="relative_advantage").reindex(columns=METRICS) * 100.0
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-max(0.01, np.nanmax(np.abs(heat.to_numpy()))), vmax=max(0.01, np.nanmax(np.abs(heat.to_numpy()))))
    ax.set_xticks(range(len(heat.columns)), heat.columns)
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("PM-CAST-D Relative Advantage by Asset (%)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "asset_metric_advantages.png", dpi=180)
    plt.close(fig)


def aggregate(args) -> None:
    assets = parse_csv(args.assets)
    roots = discover_asset_panels(Path(args.artifacts_dir), assets)
    missing = sorted(set(assets) - set(roots))
    if missing:
        raise FileNotFoundError(f"Missing asset panels: {missing}")
    frames = []
    audit_frames = []
    metadata_frames = []
    for asset in assets:
        frame = pd.read_csv(roots[asset], parse_dates=["Date"])
        frames.append(frame)
        asset_root = roots[asset].parents[1]
        audit_frames.append(pd.read_csv(asset_root / "results" / "audit.csv"))
        metadata_frames.append(pd.read_csv(asset_root / "results" / "metadata.csv"))
    panel = pd.concat(frames, ignore_index=True)
    audits = pd.concat(audit_frames, ignore_index=True)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    metrics, consistency, comparison = metric_tables(panel)
    distribution = distribution_table(panel)
    bootstrap = pd.concat(
        [moving_block_bootstrap(panel, model, args.bootstrap_reps, args.block_days, args.seed + idx * 1000) for idx, model in enumerate(MODEL_FAMILIES)],
        ignore_index=True,
    )
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pm_pooled = metrics[(metrics["asset"] == "POOLED") & (metrics["model_name"] == "PM_CAST_D")].set_index("loss_metric")
    pm_consistency = consistency[consistency["model_name"] == "PM_CAST_D"].set_index("loss_metric")
    pm_bootstrap = bootstrap[bootstrap["model_name"] == "PM_CAST_D"].set_index("loss_metric")
    pm_control = comparison[(comparison["asset"] == "POOLED") & (comparison["model_name_control"] == "CAST_D_CALIBRATION_ONLY")].set_index("loss_metric")
    checks = [
        {"check": "all_asset_audits", "passed": bool(audits["passed"].all()), "detail": f"{int(audits['passed'].sum())}/{len(audits)}"},
        {"check": "pooled_all_metrics_beat_cp", "passed": bool((pm_pooled["advantage_CP_minus_model"] > 0).all()), "detail": ",".join(metric for metric in METRICS if pm_pooled.loc[metric, "advantage_CP_minus_model"] > 0)},
        {"check": "at_least_7_assets_each_metric", "passed": bool((pm_consistency["assets_positive"] >= 7).all()), "detail": json.dumps({metric: int(pm_consistency.loc[metric, "assets_positive"]) for metric in METRICS})},
        {"check": "bootstrap_ci_positive_all_metrics", "passed": bool((pm_bootstrap["ci_low"] > 0).all()), "detail": json.dumps({metric: float(pm_bootstrap.loc[metric, "ci_low"]) for metric in METRICS})},
        {"check": "beats_calibration_only_all_metrics", "passed": bool((pm_control["advantage_PM_minus_control"] > 0).all()), "detail": ",".join(metric for metric in METRICS if pm_control.loc[metric, "advantage_PM_minus_control"] > 0)},
        {"check": "pm_transport_selected", "passed": bool(abs(float(config["models"]["PM_CAST_D"]["alpha"])) > 0), "detail": f"alpha={config['models']['PM_CAST_D']['alpha']}"},
    ]
    success = pd.DataFrame(checks)
    outdir = Path(args.outdir)
    write_csv(panel, outdir / "predictions" / "combined.csv")
    write_csv(metrics, outdir / "results" / "metrics.csv")
    write_csv(consistency, outdir / "results" / "asset_consistency.csv")
    write_csv(comparison, outdir / "results" / "matched_control_comparison.csv")
    write_csv(distribution, outdir / "results" / "distribution_calibration.csv")
    write_csv(bootstrap, outdir / "results" / "moving_block_bootstrap.csv")
    write_csv(audits, outdir / "results" / "audit.csv")
    write_csv(metadata, outdir / "results" / "metadata.csv")
    write_csv(success, outdir / "results" / "success_bar.csv")
    (outdir / "frozen_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_figures(metrics, outdir / "figures")

    report = [
        "# PM-CAST-D Warmup-600 Practitioner Results",
        "",
        f"Generated: {utc_now()}",
        "",
        "The configuration was frozen using origins strictly before the first warmup-600 OOS forecast.",
        "Positive advantage means lower loss than `CP_REPO_FRESH`.",
        "",
        "## Pooled Metrics",
        "",
        "```text",
        metrics[metrics["asset"] == "POOLED"][["model_name", "loss_metric", "CP_loss", "model_loss", "advantage_CP_minus_model", "relative_advantage"]].to_string(index=False),
        "```",
        "",
        "## Asset Consistency",
        "",
        "```text",
        consistency[consistency["model_name"] == "PM_CAST_D"].to_string(index=False),
        "```",
        "",
        "## Moving-Block Inference",
        "",
        "```text",
        bootstrap[bootstrap["model_name"] == "PM_CAST_D"].to_string(index=False),
        "```",
        "",
        "## Distribution Calibration",
        "",
        "```text",
        distribution.to_string(index=False),
        "```",
        "",
        "## Success Bar",
        "",
        "```text",
        success.to_string(index=False),
        "```",
        "",
        "Application value is established only by the observed OOS and audit checks; prime specificity still requires PM to beat the matched operators.",
    ]
    (outdir / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if bool(success["passed"].all()):
        (outdir / "SUCCESS.txt").write_text("PM-CAST-D passed every practitioner and audit success criterion\n", encoding="utf-8")
    else:
        (outdir / "RUN_COMPLETE_NEEDS_REVIEW.txt").write_text("PM-CAST-D run complete; one or more practitioner criteria failed\n", encoding="utf-8")
    print(success.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PM-CAST-D warmup-600 runner")
    parser.add_argument("--mode", choices=["tune", "run-asset", "aggregate"], required=True)
    parser.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    parser.add_argument("--asset", default="AAPL")
    parser.add_argument("--n", type=int, default=22)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-modes", type=int, default=12)
    parser.add_argument("--history-warmup", type=int, default=50)
    parser.add_argument("--tuning-min-history", type=int, default=200)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--tau-grid", default="0.1,0.5,1.0")
    parser.add_argument("--kappa-grid", default="0.07,0.12")
    parser.add_argument("--gamma-grid", default="-1,-0.5,-0.25,0.25,0.5")
    parser.add_argument("--intercept-grid", default="-0.02,0,0.02,0.05,0.07")
    parser.add_argument("--alpha-grid", default="0,0.2,0.35,0.5,0.75")
    parser.add_argument("--clip-grid", default="0.1,0.2")
    parser.add_argument("--tail-rules", default="none,q90_half,q90_zero")
    parser.add_argument("--config", default="outputs/pm_cast_d_warmup600/tuning/frozen_config.json")
    parser.add_argument("--artifacts-dir", default="_pm_cast_d_artifacts")
    parser.add_argument("--outdir", default="outputs/pm_cast_d_warmup600/run")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup != 600:
        raise ValueError("PM-CAST-D is frozen to warmup=600; pass --warmup 600")
    if args.mode == "tune":
        tune(args)
    elif args.mode == "run-asset":
        run_asset(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
