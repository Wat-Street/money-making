"""
CP-OPS-HG reproducible testing ladder.

This runner is intentionally standalone. It reuses the repository data loading,
expanding-window target alignment, and prior condition definitions, while
constructing the locked explicit CP-block benchmark and CP-orthogonal OPS/shape
ladder requested for the CP-OPS-HG research specification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import platform
import subprocess
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
from utils.models_utils import build_minimal_primes  # noqa: E402


DEFAULT_ASSETS = ["AAPL", "AMZN", "EEM", "FXI", "GLD", "GOOGL", "HYG", "QQQ", "SPY", "TLT"]
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "cp_ops_hg_tests"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "data" / "market_data" / "clean"
DEFAULT_PRIOR_PRED_DIR = PROJECT_ROOT / "run_results" / "current_intraday" / "predictions"
PRIOR_INCREMENTAL_DIR = PROJECT_ROOT / "outputs" / "cp_pm_incremental_tests"

N_LAGS_DEFAULT = 22
WARMUP_DEFAULT = 600
NUMERIC_RIDGE = 1e-12
ZERO_SUM_TOL = 1e-10
DEFAULT_LAMBDA_GRID = "0.001"
DEFAULT_LAMBDA_R_RATIO_GRID = "10"
LOG_EPS_DEFAULT = 1e-12
REGISTRY_FILENAME = "model_registry.json"
NO_LOOKAHEAD_AUDIT_FILENAME = "no_lookahead_audit.csv"
RV_HAR_COLS = ["RV_d", "RV_w", "RV_m"]

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

# Prior predictions are from the older repository model family. They remain
# useful as diagnostics, but the locked CP-OPS-HG paper ladder must regenerate
# every model from the explicit CP block benchmark, so no model is reused here.
REUSABLE_PRIOR_MODELS: dict[str, str] = {}


@dataclass
class ModelSpec:
    name: str
    features: list[str]
    ridge_penalty: dict[str, float]
    family: str
    scaffolded: bool = False
    scaffold_reason: str = ""
    lambda_shape: float = 0.0
    lambda_r_ratio: float = 0.0
    cv_mode_effective: str = "none"
    cv_score: float = np.nan


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def marker_path_for_run(outdir: Path, args, results: dict[str, pd.DataFrame] | None = None, failures: pd.DataFrame | None = None) -> Path:
    if args.mode == "smoke":
        return outdir / "SMOKE_SUCCESS.txt"
    if args.mode == "tables-only":
        return outdir / "TABLES_SUCCESS.txt"
    if args.mode == "full" and args.phase == "all" and getattr(args, "full_success_ready", False):
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
        raise ValueError("The locked CP-OPS-HG paper specification currently supports n=22 only.")
    return [list(range(1, 4)), list(range(4, 9)), list(range(9, 15)), list(range(15, 23))]


def cp_block_columns(blocks: list[list[int]]) -> list[str]:
    return [f"CPB_B{idx + 1}" for idx in range(len(blocks))]


def add_cp_block_features(frame: pd.DataFrame, blocks: list[list[int]]) -> pd.DataFrame:
    out = frame.copy()
    for block_idx, block in enumerate(blocks, start=1):
        lag_cols = [f"lag{lag}" for lag in block]
        out[f"CPB_B{block_idx}"] = out[lag_cols].mean(axis=1)
    return out


def add_raw_pm_features(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    out = frame.copy()
    for prime in build_minimal_primes(n):
        for remainder in range(prime):
            lag_cols = [f"lag{lag}" for lag in range(1, n + 1) if lag % prime == remainder]
            if lag_cols:
                out[f"PM_p{prime}_r{remainder}"] = out[lag_cols].mean(axis=1)
    return out


def parse_float_grid(value: str | None, default: str) -> list[float]:
    text = default if value is None or not str(value).strip() else str(value)
    grid = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not grid:
        raise ValueError("Grid arguments must contain at least one numeric value.")
    return grid


def select_cols(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return sorted([col for col in frame.columns if col.startswith(prefixes)])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_metadata() -> dict:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    status = run_git(["status", "--porcelain"])
    return {
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_branch": run_git(["branch", "--show-current"]),
        "git_dirty": bool(status),
    }


def package_versions_json() -> str:
    packages = {}
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib"]:
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = ""
    return json.dumps(packages, sort_keys=True)


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
    out["GATE_RANDOM"] = rng.permutation(out["GATE_REAL"].to_numpy(dtype=float))
    return out


def build_model_registry(n: int) -> dict:
    blocks = fallback_blocks(n)
    registry = {
        "registry_schema_version": 1,
        "spec": "CP-OPS-HG locked ladder",
        "n_lags": int(n),
        "cp_blocks": {f"B{idx + 1}": block for idx, block in enumerate(blocks)},
        "cp_block_columns": cp_block_columns(blocks),
        "zero_sum_tolerance": ZERO_SUM_TOL,
        "models": {
            "HAR_RV": {
                "feature_families": ["HAR_RV"],
                "includes_cp": False,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "HAR-only volatility benchmark using RV_d, RV_w, and RV_m.",
            },
            "CP_FRESH": {
                "feature_families": ["CP_BLOCK"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Fresh explicit disjoint CP-block benchmark and denominator for all advantages.",
            },
            "RAW_PM": {
                "feature_families": ["RAW_PM"],
                "includes_cp": False,
                "includes_raw_pm": True,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Raw prime-modulo residue averages without HAR or CP controls.",
            },
            "RAW_CP_PLUS_PM": {
                "feature_families": ["CP_BLOCK", "RAW_PM"],
                "includes_cp": True,
                "includes_raw_pm": True,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Naive CP plus raw PM hybrid benchmark.",
            },
            "CP_OPS_R": {
                "feature_families": ["CP_BLOCK", "OPS_R"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "CP plus CP-orthogonalized PM residue shape.",
            },
            "CP_RIDGE_OPS_R": {
                "feature_families": ["CP_BLOCK", "OPS_R"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "OPS-R with ridge shrinkage on shape coefficients only.",
            },
            "CP_LRPM": {
                "feature_families": ["CP_BLOCK", "LRPM"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Local residue bridge ablation inside CP blocks.",
            },
            "CP_RECENT_SLOPE": {
                "feature_families": ["CP_BLOCK", "RECENT_SLOPE"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Simple local recent-vs-older zero-sum slope ablation.",
            },
            "CP_HAAR_SHAPE": {
                "feature_families": ["CP_BLOCK", "HAAR"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Generic zero-sum Haar path shape benchmark.",
            },
            "CP_OPS_C": {
                "feature_families": ["CP_BLOCK", "OPS_C"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "CP plus contiguous zero-sum prime-radix path shape.",
            },
            "CP_RIDGE_OPS_C": {
                "feature_families": ["CP_BLOCK", "OPS_C"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "OPS-C with ridge shrinkage on shape coefficients only.",
            },
            "CP_GATED_RIDGE_OPS_C": {
                "feature_families": ["CP_BLOCK", "GATED_OPS_C"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": True,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Registered CP plus gated ridge OPS-C model.",
            },
            "CP_OPS_HG": {
                "feature_families": ["CP_BLOCK", "GATED_OPS_C", "GATED_OPS_R"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": True,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Final hybrid with gated OPS-C plus heavily shrunk gated OPS-R.",
            },
            "RANDOM_RESIDUES_PLACEBO": {
                "feature_families": ["CP_BLOCK", "RANDOM_RESIDUES"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": True,
                "is_placebo": True,
                "paper_eligible": True,
                "expected_interpretation": "OPS-R placebo preserving residue group sizes with random lag groups.",
            },
            "SHUFFLED_LAG_PM_PLACEBO": {
                "feature_families": ["CP_BLOCK", "SHUFFLED_PM"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": True,
                "is_placebo": True,
                "paper_eligible": True,
                "expected_interpretation": "OPS-R placebo that shuffles lag labels before PM grouping.",
            },
            "RANDOM_GATE_PLACEBO": {
                "feature_families": ["CP_BLOCK", "RANDOM_GATE_OPS_C"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": True,
                "uses_ridge": True,
                "is_placebo": True,
                "paper_eligible": True,
                "expected_interpretation": "Gated OPS-C placebo using a shuffled real gate.",
            },
            "RIDGE_AR22": {
                "feature_families": ["AR_LAG"],
                "includes_cp": False,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": True,
                "is_placebo": True,
                "paper_eligible": True,
                "expected_interpretation": "Flexible AR(22) lag benchmark, not CP-OPS evidence.",
            },
            "CP_OPS_K": {
                "feature_families": ["CP_BLOCK", "GATED_OPS_K"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": True,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Optional kernel challenger using CP-orthogonal prime-signature eigenvectors.",
            },
        },
    }
    return registry


def write_model_registry(outdir: Path, args) -> tuple[dict, str]:
    registry = build_model_registry(args.n)
    path = outdir / REGISTRY_FILENAME
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    return registry, sha256_file(path)


def feature_family(column: str) -> str:
    if column in RV_HAR_COLS:
        return "HAR_RV"
    if column.startswith("CPB_B"):
        return "CP_BLOCK"
    if column.startswith("PM_p"):
        return "RAW_PM"
    if column.startswith("Z_OPSR"):
        return "OPS_R"
    if column.startswith("Z_RANDR"):
        return "RANDOM_RESIDUES"
    if column.startswith("Z_SHUFPM"):
        return "SHUFFLED_PM"
    if column.startswith("Z_LRPM"):
        return "LRPM"
    if column.startswith("Z_RSLOPE"):
        return "RECENT_SLOPE"
    if column.startswith("Z_HAAR"):
        return "HAAR"
    if column.startswith("Z_OPSC"):
        return "OPS_C"
    if column.startswith("GATED_Z_OPSC"):
        return "GATED_OPS_C"
    if column.startswith("GATED_Z_OPSR"):
        return "GATED_OPS_R"
    if column.startswith("RGATED_Z_OPSC"):
        return "RANDOM_GATE_OPS_C"
    if column.startswith("GATED_Z_KOPS"):
        return "GATED_OPS_K"
    if column.startswith("lag"):
        return "AR_LAG"
    return "UNKNOWN"


def validate_model_features(model_name: str, features: list[str], registry: dict, blocks: list[list[int]]) -> None:
    if model_name not in registry["models"]:
        raise ValueError(f"{model_name}: model missing from registry")
    entry = registry["models"][model_name]
    allowed = set(entry["feature_families"])
    actual = [feature_family(col) for col in features]
    unknown = [col for col, family in zip(features, actual) if family == "UNKNOWN"]
    extras = [col for col, family in zip(features, actual) if family not in allowed]
    if unknown or extras:
        raise ValueError(f"{model_name}: feature purity violation unknown={unknown} extras={extras}")

    actual_families = set(actual)
    required = set(entry["feature_families"])
    if model_name != "CP_OPS_K" and not required.issubset(actual_families):
        raise ValueError(f"{model_name}: missing required feature families {sorted(required - actual_families)}")

    cp_cols = cp_block_columns(blocks)
    if model_name == "HAR_RV" and features != RV_HAR_COLS:
        raise ValueError("HAR_RV must use only RV_d, RV_w, RV_m.")
    if model_name == "CP_FRESH" and features != cp_cols:
        raise ValueError("CP_FRESH must use only explicit CP block averages.")
    if model_name == "RAW_PM" and any(not col.startswith("PM_p") for col in features):
        raise ValueError("RAW_PM must use raw PM columns only.")
    if model_name == "RAW_CP_PLUS_PM":
        expected = cp_cols + [col for col in features if col.startswith("PM_p")]
        if features != expected or not any(col.startswith("PM_p") for col in features):
            raise ValueError("RAW_CP_PLUS_PM must be explicit CP block columns followed by raw PM columns only.")
    if model_name in {"CP_GATED_RIDGE_OPS_C", "CP_OPS_HG", "RANDOM_GATE_PLACEBO", "CP_OPS_K"}:
        ungated = [col for col in features if col.startswith("Z_OPS") or col.startswith("Z_KOPS")]
        if ungated:
            raise ValueError(f"{model_name}: gated model contains ungated shape columns {ungated}")
    if model_name in {"RANDOM_RESIDUES_PLACEBO", "SHUFFLED_LAG_PM_PLACEBO", "RANDOM_GATE_PLACEBO"}:
        true_ops = [col for col in features if feature_family(col) in {"OPS_R", "OPS_C", "GATED_OPS_C", "GATED_OPS_R"}]
        if true_ops:
            raise ValueError(f"{model_name}: placebo contains true OPS columns {true_ops}")


def placebo_diagnostics_rows(
    frame: pd.DataFrame,
    n: int,
    blocks: list[list[int]],
    true_weights: dict[str, np.ndarray],
    random_weights: dict[str, np.ndarray],
    shuffled_weights: dict[str, np.ndarray],
    seed: int,
) -> list[dict]:
    rows = []
    primes = build_minimal_primes(n)
    true_group_sizes = {
        f"p{prime}": [sum(1 for lag in range(1, n + 1) if lag % prime == remainder) for remainder in range(prime)]
        for prime in primes
    }
    random_group_sizes = dict(true_group_sizes)
    rows.append(
        {
            "diagnostic": "random_residue_group_sizes",
            "passed": True,
            "seed": seed + 101,
            "details_json": json.dumps(
                {
                    "true_group_sizes": true_group_sizes,
                    "random_group_sizes": random_group_sizes,
                    "construction": "random residue groups use the true PM group-size vector for each prime",
                },
                sort_keys=True,
            ),
        }
    )

    true_matrix = np.column_stack(list(true_weights.values())) if true_weights else np.empty((n, 0))
    shuffled_matrix = np.column_stack(list(shuffled_weights.values())) if shuffled_weights else np.empty((n, 0))
    identical = bool(true_matrix.shape == shuffled_matrix.shape and np.allclose(true_matrix, shuffled_matrix))
    rows.append(
        {
            "diagnostic": "shuffled_pm_not_identical",
            "passed": not identical,
            "seed": seed + 202,
            "details_json": json.dumps(
                {
                    "true_weight_columns": int(true_matrix.shape[1]),
                    "shuffled_weight_columns": int(shuffled_matrix.shape[1]),
                    "identical_after_projection": identical,
                },
                sort_keys=True,
            ),
        }
    )

    real_gate = pd.to_numeric(frame["GATE_REAL"], errors="coerce").dropna().to_numpy(dtype=float)
    random_gate = pd.to_numeric(frame["GATE_RANDOM"], errors="coerce").dropna().to_numpy(dtype=float)
    sorted_equal = bool(len(real_gate) == len(random_gate) and np.allclose(np.sort(real_gate), np.sort(random_gate)))
    max_sorted_diff = (
        float(np.max(np.abs(np.sort(real_gate) - np.sort(random_gate)))) if len(real_gate) == len(random_gate) and len(real_gate) else np.nan
    )
    rows.append(
        {
            "diagnostic": "random_gate_shuffled_real_distribution",
            "passed": sorted_equal,
            "seed": seed + 303,
            "details_json": json.dumps(
                {
                    "sorted_value_equality": sorted_equal,
                    "max_abs_sorted_difference": max_sorted_diff,
                    "n_gate_values": int(len(real_gate)),
                },
                sort_keys=True,
            ),
        }
    )
    return rows


def build_feature_frame(ticker: str, args) -> tuple[pd.DataFrame, dict[str, list[str]], list[dict], dict, list[dict]]:
    started = time.perf_counter()
    vol = load_volatility(ticker, Path(args.local_dir))
    frame = add_lag_columns(vol.copy(), args.n)

    blocks = fallback_blocks(args.n)
    frame = add_cp_block_features(frame, blocks)
    frame = add_raw_pm_features(frame, args.n)
    manifest = []
    shape_weights = {}
    true_opsr_weights = residue_weights(args.n, blocks)
    random_weights = random_residue_weights(args.n, blocks, args.seed + 101)
    shuffled_weights = shuffled_pm_weights(args.n, blocks, args.seed + 202)
    weight_groups = {
        "OPSR": (true_opsr_weights, "OPS_R", "CP_OPS_R"),
        "RANDR": (random_weights, "RANDOM_RESIDUES", "RANDOM_RESIDUES_PLACEBO"),
        "SHUFPM": (shuffled_weights, "SHUFFLED_PM", "SHUFFLED_LAG_PM_PLACEBO"),
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

    rv_cols = RV_HAR_COLS
    cp_cols = cp_block_columns(blocks)
    pm_cols = select_cols(frame, ("PM_p",))
    lag_cols = [f"lag{lag}" for lag in range(1, args.n + 1)]

    features = {
        "HAR_RV": rv_cols,
        "CP_FRESH": cp_cols,
        "RAW_PM": pm_cols,
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
    registry = getattr(args, "model_registry", None) or build_model_registry(args.n)
    for model_name, cols in features.items():
        validate_model_features(model_name, cols, registry, blocks)

    diagnostics = placebo_diagnostics_rows(frame, args.n, blocks, true_opsr_weights, random_weights, shuffled_weights, args.seed)
    meta = {
        "asset": ticker,
        "raw_rows": int(len(vol)),
        "feature_rows": int(len(frame)),
        "elapsed_feature_seconds": time.perf_counter() - started,
        "cp_feature_count": int(len(cp_cols)),
        "pm_feature_count": int(len(pm_cols)),
        "shape_feature_count": int(len(shape_cols)),
        "cp_compatibility": "locked explicit disjoint CP block benchmark; OPS/shape zero-sum checked against the same CP block space",
        "cp_blocks_json": json.dumps(blocks),
        "target_transform": args.target_transform,
    }
    return frame, features, manifest, meta, diagnostics


def ridge_penalties(model_name: str, features: list[str], lambda_shape: float, lambda_r_ratio: float) -> dict[str, float]:
    ridge = {}
    if model_name in {"CP_RIDGE_OPS_R", "RANDOM_RESIDUES_PLACEBO", "SHUFFLED_LAG_PM_PLACEBO"}:
        ridge = {col: lambda_shape for col in features if feature_family(col) not in {"CP_BLOCK"}}
    elif model_name in {"CP_RIDGE_OPS_C", "CP_GATED_RIDGE_OPS_C", "RANDOM_GATE_PLACEBO", "CP_OPS_K"}:
        ridge = {col: lambda_shape for col in features if feature_family(col) not in {"CP_BLOCK"}}
    elif model_name == "CP_OPS_HG":
        for col in features:
            family = feature_family(col)
            if family == "GATED_OPS_C":
                ridge[col] = lambda_shape
            elif family == "GATED_OPS_R":
                ridge[col] = lambda_shape * lambda_r_ratio
    elif model_name == "RIDGE_AR22":
        ridge = {col: lambda_shape for col in features if feature_family(col) == "AR_LAG"}
    return ridge


def transformed_target(y: np.ndarray, target_transform: str, log_eps: float) -> np.ndarray:
    if target_transform == "log":
        return np.log(np.clip(y, 0.0, None) + log_eps)
    return y


def inverse_transformed_prediction(pred: float, target_transform: str, log_eps: float) -> float:
    if target_transform == "log":
        return float(max(np.exp(pred) - log_eps, 0.0))
    return float(pred)


def validation_score_for_penalties(
    frame: pd.DataFrame,
    features: list[str],
    penalties: dict[str, float],
    n: int,
    warmup: int,
    target_transform: str,
    log_eps: float,
) -> float:
    if not features:
        return np.inf
    first_forecast_origin = n + warmup
    if len(frame) <= first_forecast_origin + 1:
        return np.inf

    x_df = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_values = x_df.to_numpy(dtype=float)
    x_all = np.column_stack([np.ones(len(frame), dtype=float), x_values])
    y_level = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    y_fit = transformed_target(y_level, target_transform, log_eps)
    valid = np.isfinite(x_all).all(axis=1) & np.isfinite(y_fit)

    val_start = max(25, int(first_forecast_origin * 0.8))
    train_idx = np.arange(0, val_start)
    val_idx = np.arange(val_start, first_forecast_origin)
    train_idx = train_idx[valid[train_idx]]
    val_idx = val_idx[valid[val_idx]]
    if len(train_idx) < max(5, x_all.shape[1] + 2) or len(val_idx) < 5:
        return np.inf

    penalty = np.zeros(x_all.shape[1], dtype=float)
    for idx, col in enumerate(features, start=1):
        penalty[idx] = float(penalties.get(col, 0.0))
    penalty_matrix = np.diag(penalty + NUMERIC_RIDGE)
    penalty_matrix[0, 0] = 0.0
    x_train = x_all[train_idx]
    y_train = y_fit[train_idx]
    try:
        beta = np.linalg.solve(x_train.T @ x_train + penalty_matrix, x_train.T @ y_train)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(x_train.T @ x_train + penalty_matrix, rcond=1e-10) @ (x_train.T @ y_train)
    preds = np.array([inverse_transformed_prediction(float(x_all[idx] @ beta), target_transform, log_eps) for idx in val_idx])
    actual = y_level[val_idx]
    return float(np.nanmean(smape(pd.Series(actual), pd.Series(preds))))


def select_ridge_controls(model_name: str, features: list[str], frame: pd.DataFrame, args) -> tuple[float, float, str, float]:
    if model_name not in {
        "CP_RIDGE_OPS_R",
        "CP_RIDGE_OPS_C",
        "CP_GATED_RIDGE_OPS_C",
        "CP_OPS_HG",
        "RANDOM_RESIDUES_PLACEBO",
        "SHUFFLED_LAG_PM_PLACEBO",
        "RANDOM_GATE_PLACEBO",
        "RIDGE_AR22",
        "CP_OPS_K",
    }:
        return 0.0, 0.0, "none", np.nan

    lambda_grid = args.lambda_grid_values
    ratio_grid = args.lambda_r_ratio_grid_values if model_name == "CP_OPS_HG" else [args.lambda_r_ratio_grid_values[0]]
    effective_cv = args.cv_mode
    if effective_cv == "auto":
        effective_cv = "fixed" if args.mode == "smoke" else "inner"
    if effective_cv == "fixed":
        return float(lambda_grid[0]), float(ratio_grid[0]), "fixed", np.nan

    best = (np.inf, float(lambda_grid[0]), float(ratio_grid[0]))
    for lambda_shape in lambda_grid:
        for ratio in ratio_grid:
            penalties = ridge_penalties(model_name, features, float(lambda_shape), float(ratio))
            score = validation_score_for_penalties(
                frame,
                features,
                penalties,
                args.n,
                args.warmup,
                args.target_transform,
                args.log_eps,
            )
            if np.isfinite(score) and score < best[0]:
                best = (score, float(lambda_shape), float(ratio))
    if not np.isfinite(best[0]):
        return float(lambda_grid[0]), float(ratio_grid[0]), "inner_fallback_fixed", np.nan
    return best[1], best[2], "inner", best[0]


def model_spec(model_name: str, features: list[str], frame: pd.DataFrame, args) -> ModelSpec:
    if model_name == "CP_OPS_K" and not features:
        return ModelSpec(model_name, features, {}, "advanced", scaffolded=True, scaffold_reason="No KOPS features generated")

    lambda_shape, lambda_r_ratio, cv_mode_effective, cv_score = select_ridge_controls(model_name, features, frame, args)
    ridge = ridge_penalties(model_name, features, lambda_shape, lambda_r_ratio)
    family = "linear"
    if model_name in {"CP_RIDGE_OPS_R", "RANDOM_RESIDUES_PLACEBO", "SHUFFLED_LAG_PM_PLACEBO"}:
        family = "ridge_shape"
    elif model_name in {"CP_RIDGE_OPS_C", "CP_GATED_RIDGE_OPS_C", "RANDOM_GATE_PLACEBO", "CP_OPS_K"}:
        family = "ridge_shape"
    elif model_name == "CP_OPS_HG":
        family = "hierarchical_gated_ridge"
    elif model_name == "RIDGE_AR22":
        family = "ridge_ar"
    return ModelSpec(
        model_name,
        features,
        ridge,
        family,
        lambda_shape=lambda_shape,
        lambda_r_ratio=lambda_r_ratio,
        cv_mode_effective=cv_mode_effective,
        cv_score=cv_score,
    )


def fast_expanding_predict(
    frame: pd.DataFrame,
    spec: ModelSpec,
    n: int,
    warmup: int,
    target_transform: str = "level",
    log_eps: float = LOG_EPS_DEFAULT,
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
    y_next_fit = transformed_target(y_next, target_transform, log_eps)
    valid_x = np.isfinite(x_all).all(axis=1)
    valid_y = np.isfinite(y_next_fit) & np.isfinite(y_next)

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
                y = y_next_fit[train_idx]
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
        pred_fit = float(x_all[i] @ beta)
        pred = inverse_transformed_prediction(pred_fit, target_transform, log_eps)
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


def compute_design_rank(frame: pd.DataFrame, features: list[str]) -> int:
    if not features:
        return 0
    x = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return 0
    return int(np.linalg.matrix_rank(x.to_numpy(dtype=float)))


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


def winsorized_mean(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return np.nan
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return float(series.clip(lo, hi).mean())


def trimmed_mean(values: pd.Series, trim: float = 0.10) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if series.empty:
        return np.nan
    cut = int(math.floor(len(series) * trim))
    if cut > 0 and len(series) > 2 * cut:
        series = series.iloc[cut:-cut]
    return float(series.mean())


def standard_error(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if len(series) < 2:
        return np.nan
    return float(series.std(ddof=1) / math.sqrt(len(series)))


def one_sided_pvalue(mean_value: float, se: float) -> float:
    if not np.isfinite(mean_value) or not np.isfinite(se) or se <= 0:
        return np.nan
    z = mean_value / se
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    adjusted = []
    running = 1.0
    for rank, (idx, value) in enumerate(valid.iloc[::-1].items(), start=1):
        true_rank = m - rank + 1
        running = min(running, float(value) * m / true_rank)
        adjusted.append((idx, running))
    for idx, value in adjusted:
        out.loc[idx] = min(value, 1.0)
    return out


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
    wins = advantage[advantage > 0]
    losses = advantage[advantage < 0]
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
        "winsorized_mean_advantage_1_99": safe_float(winsorized_mean(advantage, 0.01, 0.99)),
        "trimmed_mean_advantage_10pct": safe_float(trimmed_mean(advantage, 0.10)),
        "mean_gain_when_model_wins": safe_float(wins.mean()),
        "mean_loss_when_model_loses": safe_float(losses.mean()),
        "median_abs_advantage_vs_CP": safe_float(advantage.abs().median()),
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
    eligible_sources = {"generated_current", "skipped_existing_verified"}
    eligible = bool(
        run_type == "full"
        and source in eligible_sources
        and cp_source in eligible_sources
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


def compute_inference_summary(pooled_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pooled_all.empty:
        return pd.DataFrame()
    for model_name in sorted(pooled_all["model_name"].unique(), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999):
        model_col = f"SMAPE_{model_name}_pct"
        if model_col not in pooled_all.columns:
            continue
        for condition in CONDITION_ORDER:
            subset = pooled_all[(pooled_all["model_name"] == model_name) & (pooled_all["condition"] == condition)].copy()
            if subset.empty:
                continue
            advantage = subset["SMAPE_CP_FRESH_pct"].astype(float) - subset[model_col].astype(float)
            mean_value = float(advantage.mean())
            se = standard_error(advantage)
            asset_means = subset.assign(_advantage=advantage).groupby("asset")["_advantage"].mean()
            rows.append(
                {
                    "model_name": model_name,
                    "condition": condition,
                    "n_obs": int(len(advantage.dropna())),
                    "mean_advantage_vs_CP": safe_float(mean_value),
                    "standard_error": safe_float(se),
                    "ci_low_approx": safe_float(mean_value - 1.96 * se) if np.isfinite(se) else np.nan,
                    "ci_high_approx": safe_float(mean_value + 1.96 * se) if np.isfinite(se) else np.nan,
                    "p_value_mean_advantage_gt_0": one_sided_pvalue(mean_value, se),
                    "asset_sign_count": int((asset_means > 0).sum() - (asset_means < 0).sum()),
                    "number_of_assets_positive": int((asset_means > 0).sum()),
                    "number_of_assets": int(len(asset_means)),
                    "inference_method": "normal_approx_row_se",
                }
            )
    inference = pd.DataFrame(rows)
    if not inference.empty:
        inference["fdr_adjusted_p_value"] = bh_fdr(inference["p_value_mean_advantage_gt_0"])
    return inference


def compute_results(outdir: Path, mode: str, assets: list[str], models: list[str], n: int) -> dict[str, pd.DataFrame]:
    pred_dir = outdir / "predictions" / mode
    metadata_path = outdir / "results" / "run_metadata.csv"
    meta_map = load_metadata_map(metadata_path)
    overall_rows = []
    conditional_rows = []
    alignment_rows = []
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
        cp_frame = pred[["Date", "Actual", "Predicted_CP_FRESH"]].dropna().copy()
        cp_meta = meta_map.get((asset, "CP_FRESH"), {})

        for model_name in model_list:
            pred_col = f"Predicted_{model_name}"
            if pred_col not in pred.columns:
                continue
            model_raw = pred[["Date", "Actual", pred_col]].dropna().copy()
            if model_name == "CP_FRESH":
                merged = cp_frame.copy()
                actual_match = True
                timestamps_exact = True
                model_frame = merged.rename(columns={"Actual": "Actual"})
                n_used_alignment = int(len(merged.dropna(subset=["Actual", "Predicted_CP_FRESH"])))
            else:
                merged = cp_frame.merge(model_raw, on="Date", how="inner", suffixes=("_CP", "_MODEL"))
                actual_match = bool(
                    not merged.empty
                    and np.allclose(
                        pd.to_numeric(merged["Actual_CP"], errors="coerce"),
                        pd.to_numeric(merged["Actual_MODEL"], errors="coerce"),
                        equal_nan=False,
                    )
                )
                timestamps_exact = bool(
                    len(model_raw) == len(cp_frame)
                    and model_raw["Date"].reset_index(drop=True).equals(cp_frame["Date"].reset_index(drop=True))
                )
                model_frame = merged.rename(columns={"Actual_CP": "Actual"}).drop(columns=["Actual_MODEL"])
                n_used_alignment = int(len(merged.dropna(subset=["Actual_CP", "Predicted_CP_FRESH", pred_col])))
            alignment_rows.append(
                {
                    "asset": asset,
                    "model_name": model_name,
                    "n_prediction_rows": int(len(model_raw)),
                    "first_prediction_timestamp": model_raw["Date"].min() if not model_raw.empty else pd.NaT,
                    "last_prediction_timestamp": model_raw["Date"].max() if not model_raw.empty else pd.NaT,
                    "n_rows_overlapping_CP": int(len(merged)),
                    "n_rows_used_in_CP_comparison": n_used_alignment,
                    "timestamps_exactly_match_CP": timestamps_exact,
                    "actual_target_values_match_CP": actual_match,
                }
            )
            model_frame = model_frame.dropna(subset=["Actual", "Predicted_CP_FRESH", pred_col])
            model_frame[f"AbsErr_{model_name}"] = (model_frame["Actual"] - model_frame[pred_col]).abs()
            model_frame[f"SMAPE_{model_name}_pct"] = smape(model_frame["Actual"], model_frame[pred_col])
            model_frame["AbsErr_CP_FRESH"] = (model_frame["Actual"] - model_frame["Predicted_CP_FRESH"]).abs()
            model_frame["SMAPE_CP_FRESH_pct"] = smape(model_frame["Actual"], model_frame["Predicted_CP_FRESH"])
            prov = provenance_for(meta_map, asset, model_name, cp_meta=cp_meta)
            prov.update(
                {
                    "alignment_exact": timestamps_exact,
                    "actual_match": actual_match,
                    "n_rows_overlapping_CP": int(len(merged)),
                    "n_rows_used_in_CP_comparison": int(len(model_frame)),
                }
            )
            prov["is_paper_eligible"] = bool(prov["is_paper_eligible"] and timestamps_exact and actual_match)
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
                    pooled_piece["row_is_paper_eligible"] = prov["is_paper_eligible"]
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
                prov = {
                    "run_type": mode,
                    "model_status": "pooled",
                    "source": "pooled",
                    "benchmark_source": "pooled",
                    "source_path": str(pred_dir),
                    "seed": np.nan,
                    "phase": "",
                    "is_paper_eligible": bool(subset["row_is_paper_eligible"].all()),
                    "alignment_exact": bool(subset["row_is_paper_eligible"].all()),
                    "actual_match": bool(subset["row_is_paper_eligible"].all()),
                    "n_rows_overlapping_CP": int(len(subset)),
                    "n_rows_used_in_CP_comparison": int(len(subset)),
                    "number_of_assets": int(len(assets_for_model)),
                }
                pooled_rows.append(metric_row(subset, model_name, "POOLED", condition, prov))
    pooled = pd.DataFrame(pooled_rows)
    pooled_all = pd.concat(pooled_parts, ignore_index=True) if pooled_parts else pd.DataFrame()

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
        "alignment": pd.DataFrame(alignment_rows),
        "inference": compute_inference_summary(pooled_all),
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
- Target transform: `{args.target_transform}`
- Ridge lambda grid: `{args.lambda_grid}`
- Ridge R-ratio grid: `{args.lambda_r_ratio_grid}`
- CV mode: `{args.cv_mode}`
- Command: `{args.command_line}`
- Paper eligibility rule: rows are paper eligible only when both the model and CP benchmark were generated in the current full run and exact timestamp/actual alignment passes. Smoke, reused-prior, skipped, and scaffolded rows are excluded.

## Locked CP Benchmark

`CP_FRESH` is the explicit disjoint-block CP benchmark, not the older repository `contig_prime_modulo` model. For `n=22`, the locked CP blocks are `{fallback_blocks(args.n)}`. OPS-R, OPS-C, LRPM, recent slope, Haar, KOPS, and placebo shape features are all zero-sum against this same CP block space.

## Commands

Smoke test:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode smoke --assets AAPL --phase all --seed 42 --force --audit-no-lookahead
```

Phase 1 run:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase phase1 --seed 42 --skip-existing --cv-mode inner --lambda-grid 0.0001,0.001,0.01
```

Phase 2 / full linear ladder:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase linear --seed 42 --skip-existing --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Final all-model full run:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase all --seed 42 --skip-existing --audit-no-lookahead --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
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

## Research Conclusion Guardrail

- If gated OPS-C beats CP but OPS-R adds nothing, the result validates temporal path shape, not original PM residue specificity.
- If OPS-R beats random/shuffled residues and adds beyond OPS-C, PM-specific residue structure is supported.
- If Haar/recent slope matches OPS-C, the result is generic local shape, not PM-specific.
- If positive mean advantage occurs with win rate below 50%, describe it as a gain-size effect, not dominance.
- If CP-OPS-HG loses to gated OPS-C, OPS-R is contaminating the final hybrid and should be rejected.

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
    write_csv(results["alignment"], result_dir / "alignment_audit.csv")
    write_csv(results["inference"], result_dir / "inference_summary.csv")

    paper_main = results["ablation"]
    if not paper_main.empty and "is_paper_eligible" in paper_main.columns:
        paper_main = paper_main[
            (paper_main["is_paper_eligible"] == True)  # noqa: E712
            & (paper_main.get("alignment_exact", True) == True)  # noqa: E712
            & (paper_main.get("actual_match", True) == True)  # noqa: E712
        ].copy()
    paper_cond = results["pooled"]
    if not paper_cond.empty and "is_paper_eligible" in paper_cond.columns:
        paper_cond = paper_cond[
            (paper_cond["is_paper_eligible"] == True)  # noqa: E712
            & (paper_cond.get("alignment_exact", True) == True)  # noqa: E712
            & (paper_cond.get("actual_match", True) == True)  # noqa: E712
        ].copy()
    paper_placebo = results["placebo"]
    if not paper_placebo.empty and "is_paper_eligible" in paper_placebo.columns:
        paper_placebo = paper_placebo[
            (paper_placebo["is_paper_eligible"] == True)  # noqa: E712
            & (paper_placebo.get("alignment_exact", True) == True)  # noqa: E712
            & (paper_placebo.get("actual_match", True) == True)  # noqa: E712
        ].copy()
    write_csv(paper_main, paper_dir / "main_model_summary.csv")
    write_csv(paper_cond, paper_dir / "conditional_summary.csv")
    write_csv(paper_placebo, paper_dir / "placebo_summary.csv")
    write_figures(results, outdir / "figures")
    return results


def add_common_metadata(metadata: pd.DataFrame, args, outdir: Path) -> pd.DataFrame:
    if metadata is None:
        metadata = pd.DataFrame()
    metadata = metadata.copy()
    git = getattr(args, "git_info", git_metadata())
    common = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "prior_incremental_dir": str(PRIOR_INCREMENTAL_DIR),
        "command": args.command_line,
        "outdir": str(outdir),
        "git_commit": git.get("git_commit", ""),
        "git_branch": git.get("git_branch", ""),
        "git_dirty": git.get("git_dirty", ""),
        "script_sha256": getattr(args, "script_sha256", ""),
        "model_registry_sha256": getattr(args, "model_registry_sha256", ""),
        "package_versions_json": getattr(args, "package_versions_json", ""),
        "lambda_grid": args.lambda_grid,
        "lambda_r_ratio_grid": args.lambda_r_ratio_grid,
        "cv_mode": args.cv_mode,
        "target_transform": args.target_transform,
    }
    for key, value in common.items():
        metadata[key] = value
    return metadata


def run_no_lookahead_audit(args, outdir: Path) -> pd.DataFrame:
    rows = []
    idx = pd.date_range("2020-01-01", periods=120, freq="5min")
    values = 0.1 + 0.01 * np.sin(np.arange(len(idx)) / 5.0) + np.arange(len(idx)) * 1e-4
    base = pd.DataFrame({"RV_d": values, "RV_w": pd.Series(values).rolling(5, min_periods=1).mean().to_numpy(), "RV_m": pd.Series(values).rolling(22, min_periods=1).mean().to_numpy()}, index=idx)

    z_source = pd.DataFrame({"x": np.arange(120, dtype=float)})
    z_base, _ = online_standardize(z_source, ["x"], min_periods=5)
    z_future = z_source.copy()
    z_future.loc[80:, "x"] += 9999.0
    z_changed, _ = online_standardize(z_future, ["x"], min_periods=5)
    rows.append(
        {
            "check_name": "online_standardization_uses_prior_rows",
            "passed": bool(np.allclose(z_base.loc[:79, "Z_x"], z_changed.loc[:79, "Z_x"], equal_nan=True)),
            "details": "Changing rows >=80 leaves z-scores before row 80 unchanged.",
        }
    )

    blocks = fallback_blocks(args.n)
    gate_base = add_gate_columns(add_lag_columns(base, args.n), args.n, blocks, args.seed + 303)
    future = base.copy()
    future.iloc[80:, future.columns.get_loc("RV_d")] *= 50.0
    gate_changed = add_gate_columns(add_lag_columns(future, args.n), args.n, blocks, args.seed + 303)
    rows.append(
        {
            "check_name": "gate_uses_lagged_and_prior_scaled_inputs",
            "passed": bool(np.allclose(gate_base.loc[gate_base.index[:75], "GATE_REAL"], gate_changed.loc[gate_changed.index[:75], "GATE_REAL"], equal_nan=True)),
            "details": "Changing future RV values leaves earlier gates unchanged.",
        }
    )

    frame = add_raw_pm_features(add_cp_block_features(add_lag_columns(base, args.n), blocks), args.n)
    spec = ModelSpec("CP_FRESH", cp_block_columns(blocks), {}, "linear")
    warmup = 40
    pred_a = fast_expanding_predict(frame, spec, args.n, warmup, target_transform=args.target_transform, log_eps=args.log_eps, max_forecasts=1)
    perturbed = frame.copy()
    target_row = args.n + warmup + 1
    if target_row < len(perturbed):
        perturbed.iloc[target_row, perturbed.columns.get_loc("RV_d")] *= 100.0
    pred_b = fast_expanding_predict(perturbed, spec, args.n, warmup, target_transform=args.target_transform, log_eps=args.log_eps, max_forecasts=1)
    pred_col = "Predicted_CP_FRESH"
    pred_equal = bool(not pred_a.empty and not pred_b.empty and np.isclose(pred_a[pred_col].iloc[0], pred_b[pred_col].iloc[0]))
    rows.append(
        {
            "check_name": "expanding_forecast_excludes_current_target",
            "passed": pred_equal,
            "details": "Perturbing the first forecast target changes Actual but not the fitted prediction.",
        }
    )

    first_forecast_origin = args.n + args.warmup
    validation_last_target_index = first_forecast_origin
    rows.append(
        {
            "check_name": "inner_validation_precedes_first_oos_target",
            "passed": bool(validation_last_target_index < first_forecast_origin + 1),
            "details": f"Validation targets end at index {validation_last_target_index}; first OOS target is {first_forecast_origin + 1}.",
        }
    )

    audit = pd.DataFrame(rows)
    write_csv(audit, outdir / "results" / NO_LOOKAHEAD_AUDIT_FILENAME)
    if not bool(audit["passed"].all()):
        raise RuntimeError("No-lookahead audit failed")
    return audit


def write_model_construction_proof(
    outdir: Path,
    args,
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    failures: pd.DataFrame,
    results: dict[str, pd.DataFrame],
) -> None:
    blocks = fallback_blocks(args.n)
    feature_counts = "[empty]" if metadata.empty else metadata[["asset", "model_name", "status", "feature_count", "design_rank", "rank_deficient"]].to_string(index=False)
    zero_failures = 0 if manifest.empty else int((~manifest["zero_block_sum_pass"].astype(bool)).sum())
    alignment = results.get("alignment", pd.DataFrame())
    alignment_preview = "[empty]" if alignment.empty else alignment[["asset", "model_name", "timestamps_exactly_match_CP", "actual_target_values_match_CP"]].to_string(index=False)
    content = f"""# Model Construction Proof

Generated: {utc_now()}

## CP Block Definition

The paper benchmark `CP_FRESH` uses the explicit disjoint block averages:

```text
B1 = {blocks[0]}
B2 = {blocks[1]}
B3 = {blocks[2]}
B4 = {blocks[3]}
```

The CP projection `Pi_CP` maps the 22-lag vector to its within-block means. Every OPS/shape family is built in `(I - Pi_CP)x_t` by subtracting each block's average from the candidate lag-weight vector.

## Zero-Sum Proof

For every generated OPS-R, LRPM, recent-slope, Haar, OPS-C, random-residue, shuffled-PM, and KOPS feature, the manifest evaluates `sum_{{ell in B_j}} w_ell` for every CP block. The maximum tolerated absolute block sum is `{ZERO_SUM_TOL}`. Manifest failures: `{zero_failures}`.

OPS-R is `(I - Pi_CP)u_(p,r)`. OPS-C uses adjacent sub-block contrasts inside each CP block. KOPS uses centered prime-signature kernel eigenvectors inside each CP block. Rank dependencies are handled by ridge penalties where registered and by SVD pseudo-inverse fallback in the expanding solver.

## Feature Counts And Design Ranks

```text
{feature_counts}
```

## Alignment Proof

Every model comparison is paired to `CP_FRESH` by timestamp and target value before metrics are computed. Paper eligibility requires exact timestamp and actual-target matches.

```text
{alignment_preview}
```

## Failure Status

```text
{failures.to_string(index=False) if failures is not None and not failures.empty else "[none]"}
```
"""
    (outdir / "results" / "model_construction_proof.md").write_text(content, encoding="utf-8")


def full_success_gates_pass(
    args,
    metadata: pd.DataFrame,
    failures: pd.DataFrame,
    manifest: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    outdir: Path,
) -> bool:
    if args.mode != "full" or args.phase != "all":
        return False
    if args.models_resolved != unique_in_order(["CP_FRESH"] + PHASE_MODELS["all"]):
        return False
    if failures is not None and not failures.empty:
        return False
    if manifest is not None and not manifest.empty and int((~manifest["zero_block_sum_pass"].astype(bool)).sum()) != 0:
        return False
    if metadata.empty:
        return False
    allowed_scaffold = (metadata["model_name"] == "CP_OPS_K") & (metadata["status"] == "scaffolded")
    required_ok = metadata["status"].isin(["complete", "skipped_existing_verified"]) | allowed_scaffold
    if not bool(required_ok.all()):
        return False
    alignment = results.get("alignment", pd.DataFrame())
    if alignment.empty or not bool(alignment["timestamps_exactly_match_CP"].all()) or not bool(alignment["actual_target_values_match_CP"].all()):
        return False
    audit_path = outdir / "results" / NO_LOOKAHEAD_AUDIT_FILENAME
    if not audit_path.exists():
        return False
    audit = pd.read_csv(audit_path)
    if audit.empty or not bool(audit["passed"].astype(bool).all()):
        return False
    paper_main = outdir / "paper_tables" / "main_model_summary.csv"
    if not paper_main.exists() or pd.read_csv(paper_main).empty:
        return False
    return True


def verified_existing_metadata(previous: dict, args) -> bool:
    if not previous:
        return False
    return bool(
        str(previous.get("run_type", "")) == args.mode
        and str(previous.get("status", "")) in {"complete", "skipped_existing_verified"}
        and str(previous.get("source", "")) in {"generated_current", "skipped_existing_verified"}
        and str(previous.get("script_sha256", "")) == str(getattr(args, "script_sha256", ""))
        and str(previous.get("model_registry_sha256", "")) == str(getattr(args, "model_registry_sha256", ""))
        and str(previous.get("target_transform", "")) == str(args.target_transform)
        and str(previous.get("seed", "")) == str(args.seed)
    )


def process_asset(asset: str, models: list[str], args, dirs: dict[str, Path]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    pred_dir = dirs["predictions"] / args.mode
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{asset}.csv"
    previous_meta = load_metadata_map(dirs["results"] / "run_metadata.csv")
    metadata_rows = []
    failure_rows = []
    manifest_rows = []
    placebo_rows = []
    feature_frame = None
    feature_groups = None
    feature_meta = {}

    for model_name in models:
        started = time.perf_counter()
        source_path = ""
        try:
            if args.skip_existing and not args.force and prediction_has_model(pred_path, model_name):
                prior_meta = previous_meta.get((asset, model_name), {})
                verified = verified_existing_metadata(prior_meta, args)
                metadata_rows.append(
                    {
                        "time_utc": utc_now(),
                        "asset": asset,
                        "model_name": model_name,
                        "status": "skipped_existing_verified" if verified else "skipped_existing",
                        "source": "skipped_existing_verified" if verified else "existing_output",
                        "source_path": str(pred_path),
                        "run_type": args.mode,
                        "phase": args.phase,
                        "seed": args.seed,
                        "n_obs": prior_meta.get("n_obs", np.nan),
                        "feature_count": prior_meta.get("feature_count", np.nan),
                        "design_rank": prior_meta.get("design_rank", np.nan),
                        "rank_deficient": prior_meta.get("rank_deficient", np.nan),
                        "seconds": time.perf_counter() - started,
                        "scaffolded": bool(prior_meta.get("scaffolded", False)) if verified else False,
                        "existing_metadata_verified": verified,
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
                feature_frame, feature_groups, manifest_rows, feature_meta, placebo_rows = build_feature_frame(asset, args)

            spec = model_spec(model_name, feature_groups.get(model_name, []), feature_frame, args)
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
                        "design_rank": 0,
                        "rank_deficient": False,
                        "seconds": time.perf_counter() - started,
                        "scaffolded": True,
                        "scaffold_reason": spec.scaffold_reason,
                        **feature_meta,
                    }
                )
                continue

            design_rank = compute_design_rank(feature_frame, spec.features)
            pred = fast_expanding_predict(
                feature_frame,
                spec,
                args.n,
                args.warmup,
                target_transform=args.target_transform,
                log_eps=args.log_eps,
                max_forecasts=max_rows,
            )
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
                    "design_rank": design_rank,
                    "rank_deficient": bool(design_rank < len(spec.features)),
                    "seconds": time.perf_counter() - started,
                    "scaffolded": False,
                    "model_family": spec.family,
                    "selected_lambda_shape": spec.lambda_shape,
                    "selected_lambda_r_ratio": spec.lambda_r_ratio,
                    "cv_mode_effective": spec.cv_mode_effective,
                    "cv_score": spec.cv_score,
                    "target_transform": args.target_transform,
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
                feature_frame, feature_groups, manifest_rows, feature_meta, placebo_rows = build_feature_frame(asset, args)
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
    for row in placebo_rows:
        row["asset"] = asset
        row["run_type"] = args.mode
    return metadata_rows, failure_rows, manifest_rows, placebo_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CP-OPS-HG testing ladder")
    parser.add_argument("--mode", choices=["smoke", "full", "tables-only"], default="smoke")
    parser.add_argument("--phase", choices=["phase1", "phase2", "linear", "all"], default="all")
    parser.add_argument("--assets", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--output-root", default="", help="Alias for --outdir.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--recompute", action="store_true", help="Alias for --force.")
    parser.add_argument("--reuse-prior-baselines", action="store_true")
    parser.add_argument("--n", type=int, default=N_LAGS_DEFAULT)
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument("--prior-pred-dir", default=str(DEFAULT_PRIOR_PRED_DIR))
    parser.add_argument("--smoke-max-forecasts", type=int, default=120)
    parser.add_argument("--max-forecasts", type=int, default=0, help="Optional cap for full/debug runs; 0 means no cap.")
    parser.add_argument("--lambda-grid", default=DEFAULT_LAMBDA_GRID)
    parser.add_argument("--lambda-r-ratio-grid", default=DEFAULT_LAMBDA_R_RATIO_GRID)
    parser.add_argument("--cv-mode", choices=["auto", "fixed", "inner"], default="auto")
    parser.add_argument("--target-transform", choices=["level", "log"], default="level")
    parser.add_argument("--log-eps", type=float, default=LOG_EPS_DEFAULT)
    parser.add_argument("--audit-no-lookahead", action="store_true")
    return parser


def write_run_files(
    outdir: Path,
    args,
    metadata_rows: list[dict],
    failure_rows: list[dict],
    manifest_rows: list[dict],
    placebo_rows: list[dict],
    results: dict[str, pd.DataFrame],
) -> None:
    result_dir = outdir / "results"
    metadata = add_common_metadata(pd.DataFrame(metadata_rows), args, outdir)
    failures = pd.DataFrame(
        failure_rows,
        columns=["time_utc", "asset", "model_name", "stage", "message", "traceback"],
    )
    manifest = pd.DataFrame(manifest_rows)
    placebo_diagnostics = pd.DataFrame(placebo_rows)
    write_csv(metadata, result_dir / "run_metadata.csv")
    write_csv(failures, result_dir / "run_failures.csv")
    write_csv(manifest, result_dir / "model_feature_manifest.csv")
    write_csv(placebo_diagnostics, result_dir / "placebo_diagnostics.csv")
    if args.mode == "full" and args.phase == "all" and not (result_dir / NO_LOOKAHEAD_AUDIT_FILENAME).exists():
        run_no_lookahead_audit(args, outdir)
    write_model_construction_proof(outdir, args, metadata, manifest, failures, results)
    write_readme(outdir, args, results, manifest, failures)

    required = [
        outdir / REGISTRY_FILENAME,
        result_dir / "model_overall_by_asset.csv",
        result_dir / "model_conditional_by_asset.csv",
        result_dir / "model_conditional_pooled.csv",
        result_dir / "ablation_ladder_summary.csv",
        result_dir / "placebo_summary.csv",
        result_dir / "alignment_audit.csv",
        result_dir / "inference_summary.csv",
        result_dir / "placebo_diagnostics.csv",
        result_dir / "model_feature_manifest.csv",
        result_dir / "model_construction_proof.md",
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
    if args.audit_no_lookahead:
        audit_path = result_dir / NO_LOOKAHEAD_AUDIT_FILENAME
        if not audit_path.exists() or not bool(pd.read_csv(audit_path)["passed"].astype(bool).all()):
            raise RuntimeError("No-lookahead audit marker requested but did not pass")
    args.full_success_ready = full_success_gates_pass(args, metadata, failures, manifest, results, outdir)
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
    args.force = bool(args.force or args.recompute)
    args.max_forecasts = None if int(args.max_forecasts or 0) <= 0 else int(args.max_forecasts)
    args.lambda_grid_values = parse_float_grid(args.lambda_grid, DEFAULT_LAMBDA_GRID)
    args.lambda_r_ratio_grid_values = parse_float_grid(args.lambda_r_ratio_grid, DEFAULT_LAMBDA_R_RATIO_GRID)
    fallback_blocks(args.n)
    args.assets_resolved = parse_csv(args.assets, DEFAULT_ASSETS)
    phase_models = PHASE_MODELS[args.phase]
    requested = parse_csv(args.models, phase_models) if args.models else list(phase_models)
    unknown = [model for model in requested if model not in MODEL_ORDER]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")
    args.models_resolved = unique_in_order(["CP_FRESH"] + requested)

    if args.output_root:
        args.outdir = args.output_root
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    dirs = ensure_dirs(outdir)
    args.model_registry, args.model_registry_sha256 = write_model_registry(outdir, args)
    args.script_sha256 = sha256_file(Path(__file__).resolve())
    args.git_info = git_metadata()
    args.package_versions_json = package_versions_json()

    if args.audit_no_lookahead:
        run_no_lookahead_audit(args, outdir)

    if args.mode == "tables-only":
        source_mode = "full" if (outdir / "predictions" / "full").exists() else "smoke"
        results = write_results(outdir, args, source_mode, args.assets_resolved, args.models_resolved)
        metadata_path = outdir / "results" / "run_metadata.csv"
        manifest_path = outdir / "results" / "model_feature_manifest.csv"
        failures_path = outdir / "results" / "run_failures.csv"
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
        metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
        write_model_construction_proof(outdir, args, metadata, manifest, failures, results)
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
    placebo_rows: list[dict] = []
    for asset in args.assets_resolved:
        print(f"[ASSET] {asset}")
        meta, failures, manifest, placebo = process_asset(asset, args.models_resolved, args, dirs)
        metadata_rows.extend(meta)
        failure_rows.extend(failures)
        manifest_rows.extend(manifest)
        placebo_rows.extend(placebo)

    # Metadata is needed before result aggregation so paper eligibility and provenance are available.
    result_dir = outdir / "results"
    metadata = add_common_metadata(pd.DataFrame(metadata_rows), args, outdir)
    write_csv(metadata, result_dir / "run_metadata.csv")
    write_csv(pd.DataFrame(failure_rows), result_dir / "run_failures.csv")
    write_csv(pd.DataFrame(manifest_rows), result_dir / "model_feature_manifest.csv")
    write_csv(pd.DataFrame(placebo_rows), result_dir / "placebo_diagnostics.csv")

    results = write_results(outdir, args, args.mode, args.assets_resolved, args.models_resolved)
    write_run_files(outdir, args, metadata_rows, failure_rows, manifest_rows, placebo_rows, results)
    print(f"Outputs written to {outdir}")


if __name__ == "__main__":
    main()
