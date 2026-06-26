"""
CP-OPS-HG reproducible testing ladder.

This runner is intentionally standalone. It reuses the repository data loading,
existing CP/PM feature builders, expanding-window target alignment, and prior
condition definitions, while adding the CP-orthogonal OPS/shape ladder requested
for the CP-OPS-HG research specification.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.data_utils import calculate_intraday_realized_volatility, fetch_intraday_data  # noqa: E402
from utils.models_utils import add_prime_modulo_terms, build_minimal_primes, contig_prime_modulo  # noqa: E402


DEFAULT_ASSETS = ["AAPL", "AMZN", "EEM", "FXI", "GLD", "GOOGL", "HYG", "QQQ", "SPY", "TLT"]
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "cp_ops_hg_tests"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "data" / "market_data" / "clean"
DEFAULT_PRIOR_PRED_DIR = PROJECT_ROOT / "run_results" / "current_intraday" / "predictions"
PRIOR_INCREMENTAL_DIR = PROJECT_ROOT / "outputs" / "cp_pm_incremental_tests"

N_LAGS_DEFAULT = 22
WARMUP_DEFAULT = 600
NUMERIC_RIDGE = 1e-12
ZERO_SUM_TOL = 1e-10
LAMBDA_SHAPE = 1e-3
LAMBDA_AR = 1e-2
LAMBDA_R_RATIO = 10.0

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

MODEL_ORDER = [
    "HAR_RV",
    "CP_FRESH",
    "RAW_PM",
    "RAW_CP_PLUS_PM",
    "CP_OPS_R",
    "CP_RIDGE_OPS_R",
    "CP_LRPM",
    "CP_RECENT_SLOPE",
    "CP_HAAR_SHAPE",
    "CP_OPS_C",
    "CP_RIDGE_OPS_C",
    "CP_GATED_RIDGE_OPS_C",
    "CP_OPS_HG",
    "RANDOM_RESIDUES_PLACEBO",
    "SHUFFLED_LAG_PM_PLACEBO",
    "RANDOM_GATE_PLACEBO",
    "RIDGE_AR22",
    "CP_OPS_K",
]

SHAPE_MODELS = {
    "CP_OPS_R",
    "CP_RIDGE_OPS_R",
    "CP_LRPM",
    "CP_RECENT_SLOPE",
    "CP_HAAR_SHAPE",
    "CP_OPS_C",
    "CP_RIDGE_OPS_C",
    "CP_GATED_RIDGE_OPS_C",
    "CP_OPS_HG",
    "RANDOM_RESIDUES_PLACEBO",
    "SHUFFLED_LAG_PM_PLACEBO",
    "RANDOM_GATE_PLACEBO",
    "CP_OPS_K",
}

PHASE_MODELS = {
    "phase1": [
        "HAR_RV",
        "CP_FRESH",
        "RAW_PM",
        "RAW_CP_PLUS_PM",
        "CP_OPS_R",
        "CP_OPS_C",
        "CP_RIDGE_OPS_C",
        "CP_GATED_RIDGE_OPS_C",
    ],
    "phase2": [
        "CP_RIDGE_OPS_R",
        "CP_LRPM",
        "CP_RECENT_SLOPE",
        "CP_HAAR_SHAPE",
        "CP_OPS_HG",
        "RANDOM_RESIDUES_PLACEBO",
        "SHUFFLED_LAG_PM_PLACEBO",
        "RANDOM_GATE_PLACEBO",
        "RIDGE_AR22",
    ],
}
PHASE_MODELS["linear"] = list(dict.fromkeys(PHASE_MODELS["phase1"] + PHASE_MODELS["phase2"]))
PHASE_MODELS["all"] = list(dict.fromkeys(PHASE_MODELS["linear"] + ["CP_OPS_K"]))

REUSABLE_PRIOR_MODELS = {
    "HAR_RV": "HAR",
    "CP_FRESH": "CP",
    "RAW_PM": "PM",
}


@dataclass
class ModelSpec:
    name: str
    features: list[str]
    ridge_penalty: dict[str, float]
    family: str
    scaffolded: bool = False
    scaffold_reason: str = ""


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def marker_path_for_run(outdir: Path, args, results: dict[str, pd.DataFrame] | None = None, failures: pd.DataFrame | None = None) -> Path:
    if args.mode == "smoke":
        return outdir / "SMOKE_SUCCESS.txt"
    if args.mode == "tables-only":
        return outdir / "TABLES_SUCCESS.txt"
    if args.mode == "full" and args.phase == "all":
        paper_main = pd.DataFrame() if results is None else results.get("ablation", pd.DataFrame())
        if not paper_main.empty and "is_paper_eligible" in paper_main.columns:
            paper_main = paper_main[paper_main["is_paper_eligible"] == True]  # noqa: E712
        no_failures = failures is None or failures.empty
        if no_failures and not paper_main.empty:
            return outdir / "SUCCESS.txt"
    return outdir / f"{args.mode.upper()}_{args.phase.upper()}_READY.txt"


def clear_stale_success_markers(outdir: Path, keep: Path) -> None:
    success = outdir / "SUCCESS.txt"
    if keep.name != "SUCCESS.txt" and success.exists():
        success.unlink()


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def unique_in_order(values: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def ensure_dirs(outdir: Path) -> dict[str, Path]:
    dirs = {
        "root": outdir,
        "results": outdir / "results",
        "paper_tables": outdir / "paper_tables",
        "figures": outdir / "figures",
        "predictions": outdir / "predictions",
        "logs": outdir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def fallback_blocks(n: int) -> list[list[int]]:
    if n != 22:
        # Scale only if the user explicitly changes n. The locked first run uses n=22.
        cuts = [max(1, round(n * 3 / 22)), max(2, round(n * 8 / 22)), max(3, round(n * 14 / 22)), n]
        starts = [1, cuts[0] + 1, cuts[1] + 1, cuts[2] + 1]
        return [list(range(starts[i], cuts[i] + 1)) for i in range(4)]
    return [list(range(1, 4)), list(range(4, 9)), list(range(9, 15)), list(range(15, 23))]


def select_cols(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return sorted([col for col in frame.columns if col.startswith(prefixes)])


def smape(actual: pd.Series, pred: pd.Series) -> pd.Series:
    denom = actual.abs() + pred.abs()
    return np.where(denom > 0, 200.0 * (actual - pred).abs() / denom, np.nan)


def load_volatility(ticker: str, local_dir: Path) -> pd.DataFrame:
    raw = fetch_intraday_data(ticker, use_local=True, local_dir=str(local_dir))
    vol = calculate_intraday_realized_volatility(raw)
    return vol.replace([np.inf, -np.inf], np.nan)


def add_lag_columns(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    out = frame.copy()
    for lag in range(1, n + 1):
        out[f"lag{lag}"] = out["RV_d"].shift(lag)
    return out


def lag_matrix(frame: pd.DataFrame, n: int) -> np.ndarray:
    return frame[[f"lag{lag}" for lag in range(1, n + 1)]].to_numpy(dtype=float)


def block_sum(weights: np.ndarray, block: list[int]) -> float:
    return float(np.sum([weights[lag - 1] for lag in block]))


def project_zero_blocks(weights: np.ndarray, blocks: list[list[int]]) -> np.ndarray:
    out = np.asarray(weights, dtype=float).copy()
    for block in blocks:
        idx = [lag - 1 for lag in block]
        out[idx] -= out[idx].mean()
    return out


def manifest_row(
    feature_name: str,
    feature_family: str,
    weights: np.ndarray,
    blocks: list[list[int]],
    source_model: str,
    extra: dict | None = None,
) -> dict:
    sums = {f"B{idx + 1}": block_sum(weights, block) for idx, block in enumerate(blocks)}
    max_abs = max(abs(value) for value in sums.values()) if sums else 0.0
    row = {
        "feature_name": feature_name,
        "feature_family": feature_family,
        "source_model": source_model,
        "n_lags": int(len(weights)),
        "zero_block_sum_pass": bool(max_abs <= ZERO_SUM_TOL),
        "max_abs_block_sum": float(max_abs),
        "block_sums_json": json.dumps(sums, sort_keys=True),
        "weights_json": json.dumps([float(x) for x in weights]),
    }
    if extra:
        row.update(extra)
    return row


def add_weight_features(
    frame: pd.DataFrame,
    weights: dict[str, np.ndarray],
    n: int,
) -> pd.DataFrame:
    if not weights:
        return frame
    out = frame.copy()
    x = lag_matrix(out, n)
    for name, weight in weights.items():
        out[name] = x @ weight
    return out


def residue_weights(n: int, blocks: list[list[int]]) -> dict[str, np.ndarray]:
    weights = {}
    for prime in build_minimal_primes(n):
        for remainder in range(prime):
            lags = [lag for lag in range(1, n + 1) if lag % prime == remainder]
            if not lags:
                continue
            w = np.zeros(n, dtype=float)
            for lag in lags:
                w[lag - 1] = 1.0 / len(lags)
            w = project_zero_blocks(w, blocks)
            if np.linalg.norm(w) > 1e-14:
                weights[f"OPSR_p{prime}_r{remainder}"] = w
    return weights


def random_residue_weights(n: int, blocks: list[list[int]], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = {}
    all_lags = np.arange(1, n + 1)
    for prime in build_minimal_primes(n):
        sizes = [sum(1 for lag in all_lags if lag % prime == remainder) for remainder in range(prime)]
        permuted = rng.permutation(all_lags)
        offset = 0
        for remainder, size in enumerate(sizes):
            group = sorted(int(x) for x in permuted[offset : offset + size])
            offset += size
            if not group:
                continue
            w = np.zeros(n, dtype=float)
            for lag in group:
                w[lag - 1] = 1.0 / len(group)
            w = project_zero_blocks(w, blocks)
            if np.linalg.norm(w) > 1e-14:
                weights[f"RANDR_p{prime}_r{remainder}"] = w
    return weights


def shuffled_pm_weights(n: int, blocks: list[list[int]], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled_labels = dict(zip(range(1, n + 1), rng.permutation(np.arange(1, n + 1))))
    weights = {}
    for prime in build_minimal_primes(n):
        for remainder in range(prime):
            group = [lag for lag in range(1, n + 1) if shuffled_labels[lag] % prime == remainder]
            if not group:
                continue
            w = np.zeros(n, dtype=float)
            for lag in group:
                w[lag - 1] = 1.0 / len(group)
            w = project_zero_blocks(w, blocks)
            if np.linalg.norm(w) > 1e-14:
                weights[f"SHUFPM_p{prime}_r{remainder}"] = w
    return weights


def lrpm_weights(n: int, blocks: list[list[int]]) -> dict[str, np.ndarray]:
    weights = {}
    for block_idx, block in enumerate(blocks, start=1):
        m = len(block)
        local_primes = build_minimal_primes(m)
        local_positions = dict(zip(block, range(1, m + 1)))
        for prime in local_primes:
            for remainder in range(prime):
                group = [lag for lag in block if local_positions[lag] % prime == remainder]
                if not group:
                    continue
                w = np.zeros(n, dtype=float)
                for lag in group:
                    w[lag - 1] += 1.0 / len(group)
                for lag in block:
                    w[lag - 1] -= 1.0 / m
                if np.linalg.norm(w) > 1e-14:
                    weights[f"LRPM_B{block_idx}_p{prime}_r{remainder}"] = w
    return weights


def recent_slope_weights(n: int, blocks: list[list[int]]) -> dict[str, np.ndarray]:
    weights = {}
    for block_idx, block in enumerate(blocks[:2], start=1):
        if len(block) < 2:
            continue
        split = max(1, len(block) // 2)
        recent = block[:split]
        older = block[split:]
        if not older:
            continue
        w = np.zeros(n, dtype=float)
        for lag in recent:
            w[lag - 1] = 1.0 / len(recent)
        for lag in older:
            w[lag - 1] = -1.0 / len(older)
        weights[f"RSLOPE_B{block_idx}"] = w
    return weights


def haar_weights(n: int, blocks: list[list[int]]) -> dict[str, np.ndarray]:
    weights = {}

    def add_node(block: list[int], block_idx: int, path: str, depth: int):
        if len(block) < 2 or depth > 3:
            return
        mid = len(block) // 2
        left, right = block[:mid], block[mid:]
        if not left or not right:
            return
        w = np.zeros(n, dtype=float)
        for lag in left:
            w[lag - 1] = 1.0 / len(left)
        for lag in right:
            w[lag - 1] = -1.0 / len(right)
        weights[f"HAAR_B{block_idx}_{path}"] = w
        add_node(left, block_idx, f"{path}L", depth + 1)
        add_node(right, block_idx, f"{path}R", depth + 1)

    for block_idx, block in enumerate(blocks, start=1):
        add_node(block, block_idx, "R", 1)
    return weights


def helmert_contrasts(q: int) -> list[np.ndarray]:
    contrasts = []
    for k in range(1, q):
        h = np.zeros(q, dtype=float)
        h[:k] = 1.0 / k
        h[k] = -1.0
        h = h / max(np.linalg.norm(h), 1e-12)
        contrasts.append(h)
    return contrasts


def opsc_weights(n: int, blocks: list[list[int]]) -> dict[str, np.ndarray]:
    weights = {}
    for block_idx, block in enumerate(blocks, start=1):
        m = len(block)
        for q in [2, 3, 5]:
            if q > m:
                continue
            groups = [list(map(int, group)) for group in np.array_split(np.array(block), q)]
            for contrast_idx, contrast in enumerate(helmert_contrasts(q), start=1):
                w = np.zeros(n, dtype=float)
                for group_idx, group in enumerate(groups):
                    if not group:
                        continue
                    for lag in group:
                        w[lag - 1] += contrast[group_idx] / len(group)
                if np.linalg.norm(w) > 1e-14:
                    weights[f"OPSC_B{block_idx}_q{q}_c{contrast_idx}"] = w
    return weights


def kops_weights(n: int, blocks: list[list[int]]) -> dict[str, np.ndarray]:
    weights = {}
    for block_idx, block in enumerate(blocks, start=1):
        m = len(block)
        if m < 3:
            continue
        primes = build_minimal_primes(m)
        positions = np.arange(1, m + 1, dtype=float)
        sig_cols = []
        for prime in primes:
            sig_cols.append(np.cos(2.0 * np.pi * positions / prime))
            sig_cols.append(np.sin(2.0 * np.pi * positions / prime))
        signature = np.column_stack(sig_cols)
        diffs = signature[:, None, :] - signature[None, :, :]
        dist2 = np.sum(diffs * diffs, axis=2)
        nonzero = dist2[dist2 > 1e-12]
        tau2 = float(np.median(nonzero)) if nonzero.size else 1.0
        kernel = np.exp(-dist2 / max(2.0 * tau2, 1e-12))
        center = np.eye(m) - np.ones((m, m)) / m
        centered = center @ kernel @ center
        evals, evecs = np.linalg.eigh(centered)
        order = np.argsort(evals)[::-1]
        kept = 0
        for eig_idx in order:
            if kept >= min(2, m - 1):
                break
            if evals[eig_idx] <= 1e-10:
                continue
            vec = evecs[:, eig_idx].astype(float)
            vec = vec - vec.mean()
            norm = np.linalg.norm(vec)
            if norm <= 1e-12:
                continue
            vec = vec / norm
            first_nonzero = np.flatnonzero(np.abs(vec) > 1e-10)
            if first_nonzero.size and vec[first_nonzero[0]] < 0:
                vec = -vec
            w = np.zeros(n, dtype=float)
            for pos, lag in enumerate(block):
                w[lag - 1] = vec[pos]
            kept += 1
            weights[f"KOPS_B{block_idx}_e{kept}"] = w
    return weights


def online_standardize(frame: pd.DataFrame, columns: list[str], min_periods: int = 25) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    z_cols = []
    for col in columns:
        z_col = f"Z_{col}"
        series = pd.to_numeric(out[col], errors="coerce").astype(float)
        mean = series.expanding(min_periods=min_periods).mean().shift(1)
        std = series.expanding(min_periods=min_periods).std(ddof=0).shift(1)
        z = (series - mean) / std.replace(0.0, np.nan)
        out[z_col] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        z_cols.append(z_col)
    return out, z_cols


def add_gate_columns(frame: pd.DataFrame, n: int, blocks: list[list[int]], seed: int) -> pd.DataFrame:
    out = frame.copy()
    eps = 1e-12
    lags = [f"lag{lag}" for lag in range(1, n + 1)]
    x = out[lags].to_numpy(dtype=float)
    row_mean = np.nanmean(x, axis=1)
    row_scale = np.nanmean(np.abs(x - row_mean[:, None]), axis=1)
    row_scale = np.where(np.isfinite(row_scale) & (row_scale > eps), row_scale, eps)

    recent = out[[f"lag{lag}" for lag in range(1, 4)]].mean(axis=1)
    prior = out[[f"lag{lag}" for lag in range(4, 9)]].mean(axis=1)
    out["GATE_RECENT_SLOPE"] = (recent - prior) / row_scale

    max_pos = np.nanargmax(np.where(np.isfinite(x), x, -np.inf), axis=1) + 1
    out["GATE_SPIKE_RECENCY"] = 1.0 - (max_pos - 1.0) / max(n - 1.0, 1.0)

    block1 = out[[f"lag{lag}" for lag in blocks[0]]].mean(axis=1)
    block2 = out[[f"lag{lag}" for lag in blocks[1]]].mean(axis=1)
    out["GATE_EXIT_PRESSURE"] = np.maximum((block2 - block1) / row_scale, 0.0)

    older_lag_cols = [f"lag{lag}" for lag in range(4, n + 1)]
    older = out[older_lag_cols].to_numpy(dtype=float)
    older_mean = np.nanmean(older, axis=1)
    older_mad = np.nanmean(np.abs(older - older_mean[:, None]), axis=1)
    out["GATE_DISPERSION"] = older_mad / row_scale

    gate_components = ["GATE_RECENT_SLOPE", "GATE_SPIKE_RECENCY", "GATE_EXIT_PRESSURE", "GATE_DISPERSION"]
    out, z_gate_cols = online_standardize(out, gate_components, min_periods=25)
    z_slope, z_recency, z_exit, z_disp = z_gate_cols
    score = out[z_slope] + out[z_recency] - out[z_exit] - out[z_disp]
    out["GATE_REAL"] = 1.0 / (1.0 + np.exp(-score.clip(-50, 50)))
    rng = np.random.default_rng(seed)
    out["GATE_RANDOM"] = rng.uniform(0.0, 1.0, size=len(out))
    return out


def build_feature_frame(ticker: str, args) -> tuple[pd.DataFrame, dict[str, list[str]], list[dict], dict]:
    started = time.perf_counter()
    vol = load_volatility(ticker, Path(args.local_dir))
    cp_frame = contig_prime_modulo(vol.copy(), args.n, per_day_normalize=False)
    pm_frame = add_prime_modulo_terms(vol.copy(), args.n)

    frame = cp_frame.copy()
    for col in select_cols(pm_frame, ("PM_",)):
        frame[col] = pm_frame[col]
    frame = add_lag_columns(frame, args.n)

    blocks = fallback_blocks(args.n)
    manifest = []
    shape_weights = {}
    weight_groups = {
        "OPSR": (residue_weights(args.n, blocks), "OPS_R", "CP_OPS_R"),
        "RANDR": (random_residue_weights(args.n, blocks, args.seed + 101), "RANDOM_RESIDUES", "RANDOM_RESIDUES_PLACEBO"),
        "SHUFPM": (shuffled_pm_weights(args.n, blocks, args.seed + 202), "SHUFFLED_PM", "SHUFFLED_LAG_PM_PLACEBO"),
        "LRPM": (lrpm_weights(args.n, blocks), "LRPM", "CP_LRPM"),
        "RSLOPE": (recent_slope_weights(args.n, blocks), "RECENT_SLOPE", "CP_RECENT_SLOPE"),
        "HAAR": (haar_weights(args.n, blocks), "HAAR", "CP_HAAR_SHAPE"),
        "OPSC": (opsc_weights(args.n, blocks), "OPS_C", "CP_OPS_C"),
        "KOPS": (kops_weights(args.n, blocks), "OPS_K", "CP_OPS_K"),
    }
    for _, (weights, family, source_model) in weight_groups.items():
        for feature_name, weight in weights.items():
            shape_weights[feature_name] = weight
            manifest.append(manifest_row(feature_name, family, weight, blocks, source_model))

    frame = add_weight_features(frame, shape_weights, args.n)
    shape_cols = list(shape_weights.keys())
    frame, z_shape_cols = online_standardize(frame, shape_cols)
    frame = add_gate_columns(frame, args.n, blocks, args.seed + 303)

    z_by_raw = dict(zip(shape_cols, z_shape_cols))
    z_groups = {
        prefix: [z_by_raw[col] for col in shape_cols if col.startswith(prefix)]
        for prefix in ["OPSR", "RANDR", "SHUFPM", "LRPM", "RSLOPE", "HAAR", "OPSC", "KOPS"]
    }
    for col in z_groups["OPSC"]:
        frame[f"GATED_{col}"] = frame[col] * frame["GATE_REAL"]
        frame[f"RGATED_{col}"] = frame[col] * frame["GATE_RANDOM"]
    for col in z_groups["OPSR"]:
        frame[f"GATED_{col}"] = frame[col] * frame["GATE_REAL"]
    for col in z_groups["KOPS"]:
        frame[f"GATED_{col}"] = frame[col] * frame["GATE_REAL"]

    rv_cols = ["RV_d", "RV_w", "RV_m"]
    cp_cols = rv_cols + select_cols(frame, ("CP_",))
    pm_cols = select_cols(frame, ("PM_",))
    lag_cols = [f"lag{lag}" for lag in range(1, args.n + 1)]

    features = {
        "HAR_RV": rv_cols,
        "CP_FRESH": cp_cols,
        "RAW_PM": rv_cols + pm_cols,
        "RAW_CP_PLUS_PM": cp_cols + pm_cols,
        "CP_OPS_R": cp_cols + z_groups["OPSR"],
        "CP_RIDGE_OPS_R": cp_cols + z_groups["OPSR"],
        "CP_LRPM": cp_cols + z_groups["LRPM"],
        "CP_RECENT_SLOPE": cp_cols + z_groups["RSLOPE"],
        "CP_HAAR_SHAPE": cp_cols + z_groups["HAAR"],
        "CP_OPS_C": cp_cols + z_groups["OPSC"],
        "CP_RIDGE_OPS_C": cp_cols + z_groups["OPSC"],
        "CP_GATED_RIDGE_OPS_C": cp_cols + [f"GATED_{col}" for col in z_groups["OPSC"]],
        "CP_OPS_HG": cp_cols + [f"GATED_{col}" for col in z_groups["OPSC"]] + [f"GATED_{col}" for col in z_groups["OPSR"]],
        "RANDOM_RESIDUES_PLACEBO": cp_cols + z_groups["RANDR"],
        "SHUFFLED_LAG_PM_PLACEBO": cp_cols + z_groups["SHUFPM"],
        "RANDOM_GATE_PLACEBO": cp_cols + [f"RGATED_{col}" for col in z_groups["OPSC"]],
        "RIDGE_AR22": lag_cols,
        "CP_OPS_K": cp_cols + [f"GATED_{col}" for col in z_groups["KOPS"]],
    }

    meta = {
        "asset": ticker,
        "raw_rows": int(len(vol)),
        "feature_rows": int(len(frame)),
        "elapsed_feature_seconds": time.perf_counter() - started,
        "cp_feature_count": int(len(cp_cols)),
        "pm_feature_count": int(len(pm_cols)),
        "shape_feature_count": int(len(shape_cols)),
        "cp_compatibility": "existing contig_prime_modulo benchmark; OPS zero-sum checked against fallback disjoint blocks",
        "fallback_blocks_json": json.dumps(blocks),
    }
    return frame, features, manifest, meta


def model_spec(model_name: str, features: list[str]) -> ModelSpec:
    if model_name == "CP_OPS_K" and not features:
        return ModelSpec(model_name, features, {}, "advanced", scaffolded=True, scaffold_reason="No KOPS features generated")

    ridge = {}
    family = "linear"
    if model_name in {"CP_RIDGE_OPS_R", "RANDOM_RESIDUES_PLACEBO", "SHUFFLED_LAG_PM_PLACEBO"}:
        ridge = {col: LAMBDA_SHAPE for col in features if col.startswith("Z_") and not col.startswith("Z_CP")}
        family = "ridge_shape"
    elif model_name in {"CP_RIDGE_OPS_C", "CP_GATED_RIDGE_OPS_C", "RANDOM_GATE_PLACEBO", "CP_OPS_K"}:
        ridge = {col: LAMBDA_SHAPE for col in features if "OPSC" in col or "KOPS" in col}
        family = "ridge_shape"
    elif model_name == "CP_OPS_HG":
        for col in features:
            if "OPSC" in col:
                ridge[col] = LAMBDA_SHAPE
            elif "OPSR" in col:
                ridge[col] = LAMBDA_SHAPE * LAMBDA_R_RATIO
        family = "hierarchical_gated_ridge"
    elif model_name == "RIDGE_AR22":
        ridge = {col: LAMBDA_AR for col in features if col.startswith("lag")}
        family = "ridge_ar"
    return ModelSpec(model_name, features, ridge, family)


def fast_expanding_predict(
    frame: pd.DataFrame,
    spec: ModelSpec,
    n: int,
    warmup: int,
    max_forecasts: int | None = None,
) -> pd.DataFrame:
    if not spec.features:
        return pd.DataFrame()
    x_df = frame[spec.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_values = x_df.to_numpy(dtype=float)
    t_obs = len(frame)
    if t_obs < n + warmup + 2:
        return pd.DataFrame()
    x_all = np.column_stack([np.ones(t_obs, dtype=float), x_values])
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    valid_x = np.isfinite(x_all).all(axis=1)
    valid_y = np.isfinite(y_next)

    p = x_all.shape[1]
    penalty = np.zeros(p, dtype=float)
    for idx, col in enumerate(spec.features, start=1):
        penalty[idx] = float(spec.ridge_penalty.get(col, 0.0))
    penalty_matrix = np.diag(penalty + NUMERIC_RIDGE)
    penalty_matrix[0, 0] = 0.0

    min_train_obs = max(5, p + 2)
    xtx = np.zeros((p, p), dtype=float)
    xty = np.zeros(p, dtype=float)
    train_count = 0
    rows = []
    for i in range(0, t_obs - 1):
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
            beta = np.linalg.solve(xtx + penalty_matrix, xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtx + penalty_matrix, rcond=1e-10) @ xty
        pred = float(x_all[i] @ beta)
        actual = float(y_next[i])
        if not (np.isfinite(pred) and np.isfinite(actual)):
            continue
        err = actual - pred
        denom = max(1e-12, abs(actual) + abs(pred))
        rows.append(
            {
                "Date": frame.index[i + 1],
                "Actual": actual,
                f"Predicted_{spec.name}": pred,
                f"Err_{spec.name}": err,
                f"AbsErr_{spec.name}": abs(err),
                f"SMAPE_{spec.name}_pct": 200.0 * abs(err) / denom,
            }
        )
        if max_forecasts is not None and len(rows) >= max_forecasts:
            break
    return pd.DataFrame(rows)


def load_prior_prediction(ticker: str, model_name: str, prior_dir: Path, max_rows: int | None) -> pd.DataFrame:
    prior_model = REUSABLE_PRIOR_MODELS[model_name]
    path = prior_dir / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["Date"])
    pred_col = f"Predicted_{prior_model}"
    if pred_col not in frame.columns or "Actual" not in frame.columns:
        return pd.DataFrame()
    out = frame[["Date", "Actual", pred_col]].rename(columns={pred_col: f"Predicted_{model_name}"}).copy()
    err = out["Actual"] - out[f"Predicted_{model_name}"]
    out[f"Err_{model_name}"] = err
    out[f"AbsErr_{model_name}"] = err.abs()
    out[f"SMAPE_{model_name}_pct"] = smape(out["Actual"], out[f"Predicted_{model_name}"])
    if max_rows is not None:
        out = out.head(max_rows)
    return out


def append_prediction(pred_path: Path, pred: pd.DataFrame, model_name: str, force: bool) -> None:
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    if pred.empty:
        return
    new = pred.copy()
    new["Date"] = pd.to_datetime(new["Date"])
    new = new.set_index("Date").sort_index()
    if pred_path.exists():
        acc = pd.read_csv(pred_path, parse_dates=["Date"]).set_index("Date").sort_index()
        drop_cols = [
            f"Predicted_{model_name}",
            f"Err_{model_name}",
            f"AbsErr_{model_name}",
            f"SMAPE_{model_name}_pct",
        ]
        if force:
            acc = acc.drop(columns=[col for col in drop_cols if col in acc.columns], errors="ignore")
        if "Actual" in acc.columns and "Actual" in new.columns:
            new = new.drop(columns=["Actual"])
        joined = acc.join(new, how="outer")
    else:
        joined = new
    joined.reset_index().to_csv(pred_path, index=False)


def prediction_has_model(pred_path: Path, model_name: str) -> bool:
    if not pred_path.exists():
        return False
    try:
        cols = pd.read_csv(pred_path, nrows=0).columns
    except Exception:
        return False
    return f"Predicted_{model_name}" in cols


def add_lagged_actuals(frame: pd.DataFrame, max_lag: int) -> pd.DataFrame:
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

    return {name: mask.fillna(False).astype(bool) for name, mask in conditions.items()}


def metric_row(
    frame: pd.DataFrame,
    model_name: str,
    asset: str,
    condition: str,
    provenance: dict,
) -> dict:
    if frame.empty:
        return {
            "model_name": model_name,
            "benchmark_model": "CP_FRESH",
            "asset": asset,
            "condition": condition,
            "n_obs": 0,
        }
    model_smape_col = f"SMAPE_{model_name}_pct"
    cp_smape_col = "SMAPE_CP_FRESH_pct"
    model_abs_col = f"AbsErr_{model_name}"
    cp_abs_col = "AbsErr_CP_FRESH"
    advantage = frame[cp_smape_col].astype(float) - frame[model_smape_col].astype(float)
    abs_advantage = frame[cp_abs_col].astype(float) - frame[model_abs_col].astype(float)
    row = {
        "model_name": model_name,
        "benchmark_model": "CP_FRESH",
        "asset": asset,
        "condition": condition,
        "n_obs": int(len(frame)),
        "model_mean_SMAPE": safe_float(frame[model_smape_col].mean()),
        "CP_mean_SMAPE": safe_float(frame[cp_smape_col].mean()),
        "mean_advantage_vs_CP": safe_float(advantage.mean()),
        "median_advantage_vs_CP": safe_float(advantage.median()),
        "win_rate_vs_CP": safe_float((advantage > 0).mean()),
        "mean_abs_error_model": safe_float(frame[model_abs_col].mean()),
        "mean_abs_error_CP": safe_float(frame[cp_abs_col].mean()),
        "mean_abs_error_advantage_vs_CP": safe_float(abs_advantage.mean()),
    }
    row.update(provenance)
    return row


def load_metadata_map(metadata_path: Path) -> dict[tuple[str, str], dict]:
    if not metadata_path.exists():
        return {}
    try:
        meta = pd.read_csv(metadata_path)
    except Exception:
        return {}
    if meta.empty:
        return {}
    out = {}
    for _, row in meta.iterrows():
        out[(str(row.get("asset")), str(row.get("model_name")))] = row.to_dict()
    return out


def provenance_for(meta_map: dict, asset: str, model: str, cp_meta: dict | None = None) -> dict:
    meta = meta_map.get((asset, model), {})
    cp_meta = cp_meta or meta_map.get((asset, "CP_FRESH"), {})
    source = str(meta.get("source", "unknown"))
    run_type = str(meta.get("run_type", "unknown"))
    cp_source = str(cp_meta.get("source", "unknown"))
    eligible = bool(
        run_type == "full"
        and source == "generated_current"
        and cp_source == "generated_current"
        and not bool(meta.get("scaffolded", False))
    )
    return {
        "run_type": run_type,
        "model_status": str(meta.get("status", "unknown")),
        "source": source,
        "benchmark_source": cp_source,
        "source_path": str(meta.get("source_path", "")),
        "seed": meta.get("seed", np.nan),
        "phase": str(meta.get("phase", "")),
        "is_paper_eligible": eligible,
    }


def compute_results(outdir: Path, mode: str, assets: list[str], models: list[str], n: int) -> dict[str, pd.DataFrame]:
    pred_dir = outdir / "predictions" / mode
    metadata_path = outdir / "results" / "run_metadata.csv"
    meta_map = load_metadata_map(metadata_path)
    overall_rows = []
    conditional_rows = []
    pooled_parts = []
    model_list = unique_in_order(["CP_FRESH"] + models)

    for asset in assets:
        pred_path = pred_dir / f"{asset}.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path, parse_dates=["Date"]).sort_values("Date")
        if "Predicted_CP_FRESH" not in pred.columns:
            continue
        actual = pd.to_numeric(pred["Actual"], errors="coerce")
        cp_pred = pd.to_numeric(pred["Predicted_CP_FRESH"], errors="coerce")
        pred["AbsErr_CP_FRESH"] = (actual - cp_pred).abs()
        pred["SMAPE_CP_FRESH_pct"] = smape(actual, cp_pred)

        for model_name in model_list:
            pred_col = f"Predicted_{model_name}"
            if pred_col not in pred.columns:
                continue
            model_cols = unique_in_order(["Date", "Actual", "Predicted_CP_FRESH", pred_col])
            model_frame = pred[model_cols].copy()
            model_frame = model_frame.dropna(subset=["Actual", "Predicted_CP_FRESH", pred_col])
            model_frame[f"AbsErr_{model_name}"] = (model_frame["Actual"] - model_frame[pred_col]).abs()
            model_frame[f"SMAPE_{model_name}_pct"] = smape(model_frame["Actual"], model_frame[pred_col])
            model_frame["AbsErr_CP_FRESH"] = (model_frame["Actual"] - model_frame["Predicted_CP_FRESH"]).abs()
            model_frame["SMAPE_CP_FRESH_pct"] = smape(model_frame["Actual"], model_frame["Predicted_CP_FRESH"])
            prov = provenance_for(meta_map, asset, model_name)
            overall_rows.append(metric_row(model_frame, model_name, asset, "all_observations", prov))

            lagged = add_lagged_actuals(model_frame, n)
            if lagged.empty:
                continue
            conditions = define_conditions(lagged)
            for condition in CONDITION_ORDER:
                subset = lagged.loc[conditions[condition]].copy()
                conditional_rows.append(metric_row(subset, model_name, asset, condition, prov))
                if not subset.empty:
                    pooled_piece = subset.copy()
                    pooled_piece["asset"] = asset
                    pooled_piece["model_name"] = model_name
                    pooled_piece["condition"] = condition
                    pooled_parts.append(pooled_piece)

    overall = pd.DataFrame(overall_rows)
    conditional = pd.DataFrame(conditional_rows)

    pooled_rows = []
    if pooled_parts:
        pooled_all = pd.concat(pooled_parts, ignore_index=True)
        for model_name in sorted(pooled_all["model_name"].unique(), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999):
            for condition in CONDITION_ORDER:
                subset = pooled_all[(pooled_all["model_name"] == model_name) & (pooled_all["condition"] == condition)].copy()
                if subset.empty:
                    continue
                assets_for_model = sorted(subset["asset"].unique())
                prov = provenance_for(meta_map, assets_for_model[0], model_name) if assets_for_model else {}
                pooled_rows.append(metric_row(subset, model_name, "POOLED", condition, prov))
    pooled = pd.DataFrame(pooled_rows)

    if not pooled.empty:
        ablation = pooled[pooled["condition"] == "all_observations"].copy()
        ablation["ladder_order"] = ablation["model_name"].map({name: idx + 1 for idx, name in enumerate(MODEL_ORDER)})
        ablation = ablation.sort_values(["ladder_order", "model_name"])
    else:
        ablation = pd.DataFrame()

    placebo_names = ["RANDOM_RESIDUES_PLACEBO", "SHUFFLED_LAG_PM_PLACEBO", "RANDOM_GATE_PLACEBO", "RIDGE_AR22"]
    placebo = ablation[ablation["model_name"].isin(placebo_names)].copy() if not ablation.empty else pd.DataFrame()

    return {
        "overall": overall,
        "conditional": conditional,
        "pooled": pooled,
        "ablation": ablation,
        "placebo": placebo,
    }


def write_csv(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None or frame.empty:
        fallback_columns = columns
        if fallback_columns is None and frame is not None:
            fallback_columns = list(frame.columns)
        pd.DataFrame(columns=fallback_columns or []).to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)


def write_figures(results: dict[str, pd.DataFrame], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    ablation = results["ablation"].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    if ablation.empty:
        ax.text(0.5, 0.5, "No ablation rows available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(ablation["model_name"], ablation["mean_advantage_vs_CP"])
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Mean SMAPE advantage vs CP")
        ax.tick_params(axis="x", rotation=60)
        ax.set_title("CP-OPS-HG Ablation Ladder")
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_ladder.png", dpi=180)
    plt.close(fig)

    pooled = results["pooled"].copy()
    focus = pooled[pooled["model_name"].isin(["RAW_CP_PLUS_PM", "CP_GATED_RIDGE_OPS_C", "CP_OPS_HG"])] if not pooled.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(12, 6))
    if focus.empty:
        ax.text(0.5, 0.5, "No conditional rows available", ha="center", va="center")
        ax.set_axis_off()
    else:
        pivot = focus.pivot_table(index="condition", columns="model_name", values="mean_advantage_vs_CP", aggfunc="mean")
        pivot = pivot.reindex(CONDITION_ORDER)
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Mean SMAPE advantage vs CP")
        ax.set_title("Conditional Advantages")
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(fig_dir / "conditional_advantages.png", dpi=180)
    plt.close(fig)


def write_readme(outdir: Path, args, results: dict[str, pd.DataFrame], feature_manifest: pd.DataFrame, failures: pd.DataFrame) -> None:
    ablation_preview = "[empty]"
    if not results["ablation"].empty:
        cols = ["model_name", "condition", "n_obs", "mean_advantage_vs_CP", "win_rate_vs_CP", "is_paper_eligible"]
        ablation_preview = results["ablation"][[col for col in cols if col in results["ablation"].columns]].to_string(index=False)

    zero_failures = int((~feature_manifest["zero_block_sum_pass"]).sum()) if not feature_manifest.empty else 0
    failure_preview = failures.to_string(index=False) if failures is not None and not failures.empty else "[none]"
    content = f"""# CP-OPS-HG Tests

Generated: {utc_now()}

## Run

- Mode: `{args.mode}`
- Phase: `{args.phase}`
- Assets: `{','.join(args.assets_resolved)}`
- Models requested: `{','.join(args.models_resolved)}`
- Seed: `{args.seed}`
- Reuse prior baselines: `{bool(args.reuse_prior_baselines)}`
- Skip existing: `{bool(args.skip_existing)}`
- Force rerun: `{bool(args.force)}`
- Command: `{args.command_line}`
- Paper eligibility rule: rows are paper eligible only when both the model and CP benchmark were generated in the current full run. Smoke and reused-prior rows are excluded.

## Compatibility Note

The repository CP benchmark is `contig_prime_modulo`, not an explicit disjoint CP-block projection. This runner preserves that CP benchmark for `CP_FRESH` and raw CP+PM reproduction, while OPS/shape zero-sum checks use the locked fallback blocks `{fallback_blocks(args.n)}`.

## Commands

Smoke test:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode smoke --assets AAPL --phase all --seed 42 --force
```

Phase 1 run:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase phase1 --seed 42 --skip-existing
```

Phase 2 / full linear ladder:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase linear --seed 42 --skip-existing
```

Final all-model full run:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase all --seed 42 --skip-existing
```

Regenerate paper tables from existing outputs:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode tables-only --outdir outputs/cp_ops_hg_tests
```

## Interpretation Rules

- If gated OPS-C beats CP, temporal path shape matters after CP.
- If CP-OPS-HG beats gated OPS-C, original PM residue structure adds value beyond local shape.
- If OPS-R fails against random/shuffled residues, PM-specific residue structure is not supported.
- If Haar or recent slope matches OPS-C, the result is generic path shape rather than PM-specific.
- If improvements are mostly positive mean advantage with sub-50% win rate, describe them as gain-size effects, not dominance.
- If exit/high-dispersion losses disappear, the model has solved the raw PM noise problem.
- If CP-OPS-HG loses to gated OPS-C, OPS-R is contaminating the final hybrid and should be reported as rejected.

## Zero-Block-Sum Check

Shape feature failures: `{zero_failures}`

## Ablation Preview

```text
{ablation_preview}
```

## Failures

```text
{failure_preview}
```
"""
    (outdir / "README.md").write_text(content, encoding="utf-8")


def write_results(outdir: Path, args, mode_for_tables: str, assets: list[str], models: list[str]) -> dict[str, pd.DataFrame]:
    results = compute_results(outdir, mode_for_tables, assets, models, args.n)
    result_dir = outdir / "results"
    paper_dir = outdir / "paper_tables"
    write_csv(results["overall"], result_dir / "model_overall_by_asset.csv")
    write_csv(results["conditional"], result_dir / "model_conditional_by_asset.csv")
    write_csv(results["pooled"], result_dir / "model_conditional_pooled.csv")
    write_csv(results["ablation"], result_dir / "ablation_ladder_summary.csv")
    write_csv(results["placebo"], result_dir / "placebo_summary.csv")

    paper_main = results["ablation"]
    if not paper_main.empty and "is_paper_eligible" in paper_main.columns:
        paper_main = paper_main[paper_main["is_paper_eligible"] == True].copy()  # noqa: E712
    paper_cond = results["pooled"]
    if not paper_cond.empty and "is_paper_eligible" in paper_cond.columns:
        paper_cond = paper_cond[paper_cond["is_paper_eligible"] == True].copy()  # noqa: E712
    paper_placebo = results["placebo"]
    if not paper_placebo.empty and "is_paper_eligible" in paper_placebo.columns:
        paper_placebo = paper_placebo[paper_placebo["is_paper_eligible"] == True].copy()  # noqa: E712
    write_csv(paper_main, paper_dir / "main_model_summary.csv")
    write_csv(paper_cond, paper_dir / "conditional_summary.csv")
    write_csv(paper_placebo, paper_dir / "placebo_summary.csv")
    write_figures(results, outdir / "figures")
    return results


def process_asset(asset: str, models: list[str], args, dirs: dict[str, Path]) -> tuple[list[dict], list[dict], list[dict]]:
    pred_dir = dirs["predictions"] / args.mode
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{asset}.csv"
    metadata_rows = []
    failure_rows = []
    manifest_rows = []
    feature_frame = None
    feature_groups = None
    feature_meta = {}

    for model_name in models:
        started = time.perf_counter()
        source_path = ""
        try:
            if args.skip_existing and not args.force and prediction_has_model(pred_path, model_name):
                metadata_rows.append(
                    {
                        "time_utc": utc_now(),
                        "asset": asset,
                        "model_name": model_name,
                        "status": "skipped_existing",
                        "source": "existing_output",
                        "source_path": str(pred_path),
                        "run_type": args.mode,
                        "phase": args.phase,
                        "seed": args.seed,
                        "n_obs": np.nan,
                        "feature_count": np.nan,
                        "seconds": time.perf_counter() - started,
                        "scaffolded": False,
                        **feature_meta,
                    }
                )
                continue

            max_rows = args.smoke_max_forecasts if args.mode == "smoke" else args.max_forecasts
            if args.reuse_prior_baselines and model_name in REUSABLE_PRIOR_MODELS:
                prior = load_prior_prediction(asset, model_name, Path(args.prior_pred_dir), max_rows)
                if not prior.empty:
                    source_path = str(Path(args.prior_pred_dir) / f"{asset}.csv")
                    append_prediction(pred_path, prior, model_name, force=args.force)
                    metadata_rows.append(
                        {
                            "time_utc": utc_now(),
                            "asset": asset,
                            "model_name": model_name,
                            "status": "reused_prior",
                            "source": "reused_prior",
                            "source_path": source_path,
                            "run_type": args.mode,
                            "phase": args.phase,
                            "seed": args.seed,
                            "n_obs": int(len(prior)),
                            "feature_count": np.nan,
                            "seconds": time.perf_counter() - started,
                            "scaffolded": False,
                            **feature_meta,
                        }
                    )
                    continue

            if feature_frame is None or feature_groups is None:
                feature_frame, feature_groups, manifest_rows, feature_meta = build_feature_frame(asset, args)

            spec = model_spec(model_name, feature_groups.get(model_name, []))
            if spec.scaffolded:
                metadata_rows.append(
                    {
                        "time_utc": utc_now(),
                        "asset": asset,
                        "model_name": model_name,
                        "status": "scaffolded",
                        "source": "scaffolded",
                        "source_path": "",
                        "run_type": args.mode,
                        "phase": args.phase,
                        "seed": args.seed,
                        "n_obs": 0,
                        "feature_count": 0,
                        "seconds": time.perf_counter() - started,
                        "scaffolded": True,
                        "scaffold_reason": spec.scaffold_reason,
                        **feature_meta,
                    }
                )
                continue

            pred = fast_expanding_predict(feature_frame, spec, args.n, args.warmup, max_forecasts=max_rows)
            if pred.empty:
                raise RuntimeError(f"{model_name} produced no predictions")
            append_prediction(pred_path, pred, model_name, force=args.force)
            metadata_rows.append(
                {
                    "time_utc": utc_now(),
                    "asset": asset,
                    "model_name": model_name,
                    "status": "complete",
                    "source": "generated_current",
                    "source_path": str(pred_path),
                    "run_type": args.mode,
                    "phase": args.phase,
                    "seed": args.seed,
                    "n_obs": int(len(pred)),
                    "feature_count": int(len(spec.features)),
                    "seconds": time.perf_counter() - started,
                    "scaffolded": False,
                    "model_family": spec.family,
                    **feature_meta,
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "time_utc": utc_now(),
                    "asset": asset,
                    "model_name": model_name,
                    "stage": "process_asset",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            metadata_rows.append(
                {
                    "time_utc": utc_now(),
                    "asset": asset,
                    "model_name": model_name,
                    "status": "failed",
                    "source": "failed",
                    "source_path": source_path,
                    "run_type": args.mode,
                    "phase": args.phase,
                    "seed": args.seed,
                    "n_obs": 0,
                    "feature_count": np.nan,
                    "seconds": time.perf_counter() - started,
                    "scaffolded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    **feature_meta,
                }
            )
    if not manifest_rows and any(model in SHAPE_MODELS for model in models):
        try:
            if feature_frame is None or feature_groups is None:
                feature_frame, feature_groups, manifest_rows, feature_meta = build_feature_frame(asset, args)
        except Exception as exc:
            failure_rows.append(
                {
                    "time_utc": utc_now(),
                    "asset": asset,
                    "model_name": "FEATURE_MANIFEST",
                    "stage": "manifest_rebuild",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    for row in manifest_rows:
        row["asset"] = asset
        row["run_type"] = args.mode
        row["seed"] = args.seed
    return metadata_rows, failure_rows, manifest_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CP-OPS-HG testing ladder")
    parser.add_argument("--mode", choices=["smoke", "full", "tables-only"], default="smoke")
    parser.add_argument("--phase", choices=["phase1", "phase2", "linear", "all"], default="all")
    parser.add_argument("--assets", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-prior-baselines", action="store_true")
    parser.add_argument("--n", type=int, default=N_LAGS_DEFAULT)
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument("--prior-pred-dir", default=str(DEFAULT_PRIOR_PRED_DIR))
    parser.add_argument("--smoke-max-forecasts", type=int, default=120)
    parser.add_argument("--max-forecasts", type=int, default=0, help="Optional cap for full/debug runs; 0 means no cap.")
    return parser


def write_run_files(
    outdir: Path,
    args,
    metadata_rows: list[dict],
    failure_rows: list[dict],
    manifest_rows: list[dict],
    results: dict[str, pd.DataFrame],
) -> None:
    result_dir = outdir / "results"
    metadata = pd.DataFrame(metadata_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["time_utc", "asset", "model_name", "stage", "message", "traceback"],
    )
    manifest = pd.DataFrame(manifest_rows)

    if not metadata.empty:
        metadata["python_version"] = platform.python_version()
        metadata["platform"] = platform.platform()
        metadata["project_root"] = str(PROJECT_ROOT)
        metadata["prior_incremental_dir"] = str(PRIOR_INCREMENTAL_DIR)
        metadata["command"] = args.command_line
        metadata["outdir"] = str(outdir)
    write_csv(metadata, result_dir / "run_metadata.csv")
    write_csv(failures, result_dir / "run_failures.csv")
    write_csv(manifest, result_dir / "model_feature_manifest.csv")
    write_readme(outdir, args, results, manifest, failures)

    required = [
        result_dir / "model_overall_by_asset.csv",
        result_dir / "model_conditional_by_asset.csv",
        result_dir / "model_conditional_pooled.csv",
        result_dir / "ablation_ladder_summary.csv",
        result_dir / "placebo_summary.csv",
        result_dir / "model_feature_manifest.csv",
        result_dir / "run_metadata.csv",
        result_dir / "run_failures.csv",
        outdir / "paper_tables" / "main_model_summary.csv",
        outdir / "paper_tables" / "conditional_summary.csv",
        outdir / "paper_tables" / "placebo_summary.csv",
        outdir / "figures" / "ablation_ladder.png",
        outdir / "figures" / "conditional_advantages.png",
        outdir / "README.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    zero_failures = 0 if manifest.empty else int((~manifest["zero_block_sum_pass"].astype(bool)).sum())
    if missing:
        raise RuntimeError(f"Required outputs missing: {missing}")
    if zero_failures:
        raise RuntimeError(f"Zero-block-sum validation failed for {zero_failures} features")
    marker = marker_path_for_run(outdir, args, results=results, failures=failures)
    clear_stale_success_markers(outdir, marker)
    marker.write_text(
        f"CP-OPS-HG {args.mode} {args.phase} run completed successfully at {utc_now()}\n",
        encoding="utf-8",
    )
    if args.mode == "smoke":
        (outdir / "IMPLEMENTATION_READY.txt").write_text(
            f"CP-OPS-HG implementation-ready smoke verification completed at {utc_now()}\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.command_line = " ".join(sys.argv)
    args.max_forecasts = None if int(args.max_forecasts or 0) <= 0 else int(args.max_forecasts)
    args.assets_resolved = parse_csv(args.assets, DEFAULT_ASSETS)
    phase_models = PHASE_MODELS[args.phase]
    requested = parse_csv(args.models, phase_models) if args.models else list(phase_models)
    unknown = [model for model in requested if model not in MODEL_ORDER]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")
    args.models_resolved = unique_in_order(["CP_FRESH"] + requested)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    dirs = ensure_dirs(outdir)

    if args.mode == "tables-only":
        source_mode = "full" if (outdir / "predictions" / "full").exists() else "smoke"
        results = write_results(outdir, args, source_mode, args.assets_resolved, args.models_resolved)
        metadata_path = outdir / "results" / "run_metadata.csv"
        manifest_path = outdir / "results" / "model_feature_manifest.csv"
        failures_path = outdir / "results" / "run_failures.csv"
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
        metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
        write_readme(outdir, args, results, manifest, failures)
        if metadata.empty:
            write_csv(metadata, metadata_path)
        marker = marker_path_for_run(outdir, args, results=results, failures=failures)
        clear_stale_success_markers(outdir, marker)
        marker.write_text(
            f"CP-OPS-HG tables-only run completed successfully at {utc_now()}\n",
            encoding="utf-8",
        )
        return

    print(f"CP-OPS-HG runner: mode={args.mode} phase={args.phase} assets={args.assets_resolved}")
    print(f"Models: {args.models_resolved}")
    metadata_rows: list[dict] = []
    failure_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for asset in args.assets_resolved:
        print(f"[ASSET] {asset}")
        meta, failures, manifest = process_asset(asset, args.models_resolved, args, dirs)
        metadata_rows.extend(meta)
        failure_rows.extend(failures)
        manifest_rows.extend(manifest)

    # Metadata is needed before result aggregation so paper eligibility and provenance are available.
    result_dir = outdir / "results"
    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        metadata["python_version"] = platform.python_version()
        metadata["platform"] = platform.platform()
        metadata["project_root"] = str(PROJECT_ROOT)
        metadata["prior_incremental_dir"] = str(PRIOR_INCREMENTAL_DIR)
        metadata["command"] = args.command_line
        metadata["outdir"] = str(outdir)
    write_csv(metadata, result_dir / "run_metadata.csv")
    write_csv(pd.DataFrame(failure_rows), result_dir / "run_failures.csv")
    write_csv(pd.DataFrame(manifest_rows), result_dir / "model_feature_manifest.csv")

    results = write_results(outdir, args, args.mode, args.assets_resolved, args.models_resolved)
    write_run_files(outdir, args, metadata_rows, failure_rows, manifest_rows, results)
    print(f"Outputs written to {outdir}")


if __name__ == "__main__":
    main()
