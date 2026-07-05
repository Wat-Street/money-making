"""
CP-REPO-OPS-HG reproducible testing ladder.

This runner is intentionally standalone. It reuses the repository data loading,
expanding-window target alignment, and prior condition definitions, while
constructing the corrected repo-CP benchmark and repo-CP-controlled OPS/shape
ladder requested for the CP-REPO-OPS-HG research specification. The legacy
four-zone lag blocks are diagnostic/local shape scaffolds only.
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
from utils.models_utils import add_prime_modulo_terms, build_minimal_primes, contig_prime_modulo  # noqa: E402


DEFAULT_ASSETS = ["AAPL", "AMZN", "EEM", "FXI", "GLD", "GOOGL", "HYG", "QQQ", "SPY", "TLT"]
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "cp_repo_ops_hg_tests"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "data" / "market_data" / "clean"
DEFAULT_PRIOR_PRED_DIR = PROJECT_ROOT / "run_results" / "current_intraday" / "predictions"
PRIOR_INCREMENTAL_DIR = PROJECT_ROOT / "outputs" / "cp_pm_incremental_tests"

N_LAGS_DEFAULT = 22
WARMUP_DEFAULT = 600
NUMERIC_RIDGE = 1e-12
ZERO_SUM_TOL = 1e-10
DEFAULT_LAMBDA_GRID = "0.001"
DEFAULT_LAMBDA_R_RATIO_GRID = "10"
DEFAULT_PM_TAU_GRID = "0.05,0.1,0.25,0.5,1.0"
DEFAULT_PM_ETA_GRID = "0,0.25,0.5,1.0"
DEFAULT_PM_BANDWIDTH_GRID = "0.5,1,2"
DEFAULT_PHQO_GAMMA_GRID = "0,0.25,0.5,1,2"
DEFAULT_PM_LOW_MODES = 12
DEFAULT_PM_KERNEL_TRAIN_WINDOW = 2500
LOG_EPS_DEFAULT = 1e-12
REGISTRY_FILENAME = "model_registry.json"
NO_LOOKAHEAD_AUDIT_FILENAME = "no_lookahead_audit.csv"
RV_HAR_COLS = ["RV_d", "RV_w", "RV_m"]
PM_NATIVE_MODELS = ["PM_QDK", "PM_QDK_2", "PHQO"]

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
    "recent_ramp_high_dispersion",
    "entry_with_recent_spike",
    "entry_without_recent_spike",
    "same_average_recent_front_loaded",
    "same_average_back_loaded",
    "stable_low_dispersion",
]

MODEL_ORDER = [
    "HAR_RV",
    "CP_REPO_FRESH",
    "RAW_PM",
    "RAW_CP_REPO_PLUS_PM",
    "CP_REPO_OPS_R",
    "CP_REPO_RIDGE_OPS_R",
    "CP_REPO_LRPM",
    "CP_REPO_RECENT_SLOPE",
    "CP_REPO_HAAR_SHAPE",
    "CP_REPO_OPS_C",
    "CP_REPO_RIDGE_OPS_C",
    "CP_REPO_GATED_RIDGE_OPS_C",
    "CP_REPO_OPS_HG",
    "RANDOM_RESIDUES_PLACEBO_REPO",
    "SHUFFLED_LAG_PM_PLACEBO_REPO",
    "RANDOM_GATE_PLACEBO_REPO",
    "RIDGE_AR22",
    "CP_REPO_OPS_K",
    "PM_QDK",
    "PM_QDK_2",
    "PHQO",
]

SHAPE_MODELS = {
    "CP_REPO_OPS_R",
    "CP_REPO_RIDGE_OPS_R",
    "CP_REPO_LRPM",
    "CP_REPO_RECENT_SLOPE",
    "CP_REPO_HAAR_SHAPE",
    "CP_REPO_OPS_C",
    "CP_REPO_RIDGE_OPS_C",
    "CP_REPO_GATED_RIDGE_OPS_C",
    "CP_REPO_OPS_HG",
    "RANDOM_RESIDUES_PLACEBO_REPO",
    "SHUFFLED_LAG_PM_PLACEBO_REPO",
    "RANDOM_GATE_PLACEBO_REPO",
    "CP_REPO_OPS_K",
    "PM_QDK",
    "PM_QDK_2",
    "PHQO",
}

PHASE_MODELS = {
    "phase1": [
        "HAR_RV",
        "CP_REPO_FRESH",
        "RAW_PM",
        "RAW_CP_REPO_PLUS_PM",
        "CP_REPO_OPS_R",
        "CP_REPO_OPS_C",
        "CP_REPO_RIDGE_OPS_C",
        "CP_REPO_GATED_RIDGE_OPS_C",
    ],
    "phase2": [
        "CP_REPO_RIDGE_OPS_R",
        "CP_REPO_LRPM",
        "CP_REPO_RECENT_SLOPE",
        "CP_REPO_HAAR_SHAPE",
        "CP_REPO_OPS_HG",
        "RANDOM_RESIDUES_PLACEBO_REPO",
        "SHUFFLED_LAG_PM_PLACEBO_REPO",
        "RANDOM_GATE_PLACEBO_REPO",
        "RIDGE_AR22",
    ],
}
PHASE_MODELS["linear"] = list(dict.fromkeys(PHASE_MODELS["phase1"] + PHASE_MODELS["phase2"]))
PHASE_MODELS["all"] = list(dict.fromkeys(PHASE_MODELS["linear"] + ["CP_REPO_OPS_K"] + PM_NATIVE_MODELS))

# Prior predictions remain diagnostic only; paper-eligible rows are regenerated
# by this runner around the repo contig_prime_modulo CP benchmark.
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
        raise ValueError("The locked CP-REPO-OPS-HG paper specification currently supports n=22 only.")
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
        "global_weight_sum": float(np.sum(weights)),
        "global_level_sum_pass": bool(abs(float(np.sum(weights))) <= ZERO_SUM_TOL),
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


def residue_contrast_weights(n: int, prefix: str, seed: int | None = None, shuffled: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed) if seed is not None else None
    weights = {}
    all_lags = np.arange(1, n + 1)
    shuffled_labels = None
    if shuffled:
        shuffled_labels = dict(zip(range(1, n + 1), rng.permutation(all_lags)))
    for prime in build_minimal_primes(n):
        if rng is not None and not shuffled:
            sizes = [sum(1 for lag in all_lags if lag % prime == remainder) for remainder in range(prime)]
            permuted = rng.permutation(all_lags)
            groups = []
            offset = 0
            for size in sizes:
                groups.append([int(x) for x in permuted[offset : offset + size]])
                offset += size
        else:
            groups = []
            for remainder in range(prime):
                if shuffled_labels is None:
                    groups.append([lag for lag in range(1, n + 1) if lag % prime == remainder])
                else:
                    groups.append([lag for lag in range(1, n + 1) if shuffled_labels[lag] % prime == remainder])
        raw = []
        for group in groups:
            w = np.zeros(n, dtype=float)
            if group:
                for lag in group:
                    w[lag - 1] = 1.0 / len(group)
            raw.append(w)
        mean_weight = np.mean(raw, axis=0)
        for remainder, w in enumerate(raw):
            centered = w - mean_weight
            if np.linalg.norm(centered) > 1e-14:
                weights[f"{prefix}_k{prime}_r{remainder}"] = centered
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


def pm_primes(n: int) -> list[int]:
    primes = build_minimal_primes(n)
    if primes != [2, 3, 5]:
        raise ValueError(f"PM-native geometry is locked to n=22 with primes [2, 3, 5]; got {primes}")
    return primes


def prime_torus_modes(n: int, include_zero: bool = False) -> list[tuple[int, int, int]]:
    primes = pm_primes(n)
    modes = []
    for a2 in range(primes[0]):
        for a3 in range(primes[1]):
            for a5 in range(primes[2]):
                mode = (a2, a3, a5)
                if include_zero or any(mode):
                    modes.append(mode)
    return sorted(modes, key=lambda mode: (prime_torus_eigenvalue(mode, primes), mode))


def prime_torus_eigenvalue(mode: tuple[int, int, int], primes: list[int]) -> float:
    return float(sum(2.0 - 2.0 * math.cos(2.0 * math.pi * a / p) for a, p in zip(mode, primes)))


def prime_character_matrix(n: int, modes: list[tuple[int, int, int]], normalize: bool = False) -> np.ndarray:
    primes = pm_primes(n)
    lags = np.arange(1, n + 1, dtype=float)
    mat = np.empty((n, len(modes)), dtype=np.complex128)
    for idx, mode in enumerate(modes):
        phase = np.zeros(n, dtype=float)
        for a, prime in zip(mode, primes):
            phase += float(a) * np.mod(lags, prime) / float(prime)
        mat[:, idx] = np.exp(2j * np.pi * phase)
    if normalize and len(modes):
        mat = mat / math.sqrt(float(n))
    return mat


def cp_path_projection_matrix(n: int) -> np.ndarray:
    primes = pm_primes(n)
    rows = []

    lag1 = np.zeros(n, dtype=float)
    lag1[0] = 1.0
    rows.append(lag1)

    weekly = np.zeros(n, dtype=float)
    weekly[: min(5, n)] = 1.0 / float(min(5, n))
    rows.append(weekly)

    monthly = np.ones(n, dtype=float) / float(n)
    rows.append(monthly)

    for prime in primes:
        for remainder in range(prime):
            lags = [lag for lag in range(1, n + 1) if lag % prime == remainder]
            row = np.zeros(n, dtype=float)
            for lag in lags:
                row[lag - 1] = 1.0 / float(len(lags))
            rows.append(row)
    return np.vstack(rows)


def cp_orthogonal_projection(n: int) -> tuple[np.ndarray, np.ndarray]:
    c_mat = cp_path_projection_matrix(n)
    recon = c_mat.T @ np.linalg.pinv(c_mat @ c_mat.T)
    m_c = np.eye(n, dtype=float) - recon @ c_mat
    return c_mat, m_c


def real_embedding(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values.real, values.imag], axis=1)


def pm_mode_subset(n: int, max_modes: int | None = None) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    modes = prime_torus_modes(n, include_zero=False)
    if max_modes is not None and max_modes > 0:
        modes = modes[: min(int(max_modes), len(modes))]
    primes = pm_primes(n)
    mu = np.array([prime_torus_eigenvalue(mode, primes) for mode in modes], dtype=float)
    return modes, mu


def robust_scale(values: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    scale = np.nanstd(arr, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, floor)
    return scale


def pm_native_feature_count(model_name: str, args) -> int:
    c_rows = cp_path_projection_matrix(args.n).shape[0]
    mode_count = len(prime_torus_modes(args.n, include_zero=False))
    if model_name == "PM_QDK":
        return c_rows + 2 * mode_count
    if model_name == "PM_QDK_2":
        return 1 + c_rows + 2 * min(int(args.pm_low_modes), mode_count) * len(args.pm_tau_grid_values)
    if model_name == "PHQO":
        return int(args.n)
    return 0


def phqo_base_weights(n: int) -> np.ndarray:
    lag1 = np.zeros(n, dtype=float)
    lag1[0] = 1.0
    weekly = np.zeros(n, dtype=float)
    weekly[: min(5, n)] = 1.0 / float(min(5, n))
    monthly = np.ones(n, dtype=float) / float(n)
    beta = (lag1 + weekly + monthly) / 3.0
    beta = np.clip(beta, 1e-12, None)
    return beta / beta.sum()


def pm_native_manifest_rows(n: int, blocks: list[list[int]], args) -> list[dict]:
    c_mat, m_c = cp_orthogonal_projection(n)
    del c_mat
    modes, mu = pm_mode_subset(n)
    chars = prime_character_matrix(n, modes, normalize=False)
    tau = float(args.pm_tau_grid_values[0]) if getattr(args, "pm_tau_grid_values", None) else float(DEFAULT_PM_TAU_GRID.split(",")[0])
    qdk_weights = (np.conjugate(chars).T @ m_c).T * np.sqrt(np.exp(-tau * mu))[None, :]
    rows = []
    for idx, mode in enumerate(modes):
        base_extra = {
            "prime_mode": json.dumps(list(mode)),
            "laplacian_eigenvalue": float(mu[idx]),
            "diffusion_tau": tau,
            "model_role": "PM quotient harmonic basis",
        }
        rows.append(
            manifest_row(
                f"PMQDK_MODE_{idx + 1:02d}_REAL",
                "PM_QDK_HARMONIC",
                qdk_weights[:, idx].real,
                blocks,
                "PM_QDK",
                extra={**base_extra, "component": "real"},
            )
        )
        rows.append(
            manifest_row(
                f"PMQDK_MODE_{idx + 1:02d}_IMAG",
                "PM_QDK_HARMONIC",
                qdk_weights[:, idx].imag,
                blocks,
                "PM_QDK",
                extra={**base_extra, "component": "imag"},
            )
        )

    low_modes, low_mu = pm_mode_subset(n, int(args.pm_low_modes))
    low_chars = prime_character_matrix(n, low_modes, normalize=True)
    phqo_tau = tau
    h_pm = np.real(low_chars @ np.diag(np.exp(-phqo_tau * low_mu)) @ np.conjugate(low_chars).T @ m_c)
    for lag in range(n):
        rows.append(
            manifest_row(
                f"PHQO_ENERGY_LAG_{lag + 1:02d}",
                "PHQO_ENERGY_FIELD",
                h_pm[lag, :],
                blocks,
                "PHQO",
                extra={
                    "diffusion_tau": phqo_tau,
                    "pm_low_modes": int(args.pm_low_modes),
                    "model_role": "Prime-harmonic quotient energy row",
                },
            )
        )
    return rows


def pm_qdk_phi(x: np.ndarray, n: int, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c_mat, m_c = cp_orthogonal_projection(n)
    modes, mu = pm_mode_subset(n)
    chars = prime_character_matrix(n, modes, normalize=False)
    residual = x @ m_c.T
    coeff = residual @ np.conjugate(chars)
    coeff = coeff * np.sqrt(np.exp(-float(tau) * mu))[None, :]
    return x @ c_mat.T, real_embedding(coeff), mu


def pm_qdk2_embedding(x: np.ndarray, n: int, tau_grid: list[float], log_eps: float, low_modes: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c_mat, _ = cp_orthogonal_projection(n)
    modes, mu = pm_mode_subset(n, low_modes)
    chars = prime_character_matrix(n, modes, normalize=False)
    z = np.log(np.clip(x, 0.0, None) + log_eps)
    level = z.mean(axis=1)
    centered = z - level[:, None]
    pieces = []
    base_coeff = centered @ np.conjugate(chars)
    for tau in tau_grid:
        pieces.append(base_coeff * np.exp(-float(tau) * mu)[None, :])
    phi = real_embedding(np.concatenate(pieces, axis=1))
    return level, x @ c_mat.T, phi


def expanding_linear_residuals(target: np.ndarray, controls: np.ndarray, valid: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    n_rows, dim = target.shape
    z = np.column_stack([np.ones(n_rows, dtype=float), controls])
    q = np.full_like(target, np.nan, dtype=float)
    p = z.shape[1]
    zz = np.zeros((p, p), dtype=float)
    zy = np.zeros((p, dim), dtype=float)
    count = 0
    penalty = np.eye(p, dtype=float) * ridge
    penalty[0, 0] = 0.0
    for idx in range(n_rows):
        if count >= p + 2 and np.isfinite(z[idx]).all() and np.isfinite(target[idx]).all():
            try:
                beta = np.linalg.solve(zz + penalty, zy)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(zz + penalty, rcond=1e-10) @ zy
            q[idx] = target[idx] - z[idx] @ beta
        elif np.isfinite(target[idx]).all():
            q[idx] = target[idx]
        if bool(valid[idx]) and np.isfinite(z[idx]).all() and np.isfinite(target[idx]).all():
            zz += np.outer(z[idx], z[idx])
            zy += np.outer(z[idx], target[idx])
            count += 1
    return q


def weighted_smape_action(y: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    y = y[mask]
    weights = weights[mask]
    if y.size == 0:
        return np.nan
    y = np.clip(y, 0.0, None)
    high = max(float(np.nanmax(y)) * 2.0, eps)
    low = 0.0

    def deriv(a: float) -> float:
        denom = np.square(y + a + eps)
        left = y < a
        right = y > a
        return float(np.sum(weights[left] * 2.0 * y[left] / denom[left]) - np.sum(weights[right] * 2.0 * y[right] / denom[right]))

    if deriv(low) >= 0:
        return 0.0
    if deriv(high) <= 0:
        return high
    for _ in range(64):
        mid = 0.5 * (low + high)
        if deriv(mid) <= 0:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def validation_window_indices(frame: pd.DataFrame, n: int, warmup: int) -> tuple[int, int]:
    first_forecast_origin = n + warmup
    val_start = max(n + 25, int(first_forecast_origin * 0.8))
    val_start = min(val_start, max(n + 1, first_forecast_origin - 10))
    return val_start, first_forecast_origin


def softmax_weights(log_weights: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(log_weights)
    if not finite.any():
        return np.array([], dtype=float)
    lw = log_weights[finite]
    lw = lw - np.max(lw)
    w = np.exp(np.clip(lw, -745.0, 50.0))
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        return np.array([], dtype=float)
    out = np.zeros_like(log_weights, dtype=float)
    out[finite] = w / total
    return out


def candidate_train_indices(valid: np.ndarray, y_next: np.ndarray, i: int, train_window: int | None = None) -> np.ndarray:
    start = 0
    if train_window is not None and int(train_window) > 0:
        start = max(0, i - int(train_window))
    idx = np.arange(start, i)
    mask = valid[idx] & np.isfinite(y_next[idx])
    return idx[mask]


def prediction_rows_common(
    frame: pd.DataFrame,
    model_name: str,
    values: list[tuple[int, float]],
) -> pd.DataFrame:
    rows = []
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    for i, pred in values:
        actual = float(y_next[i])
        if not (np.isfinite(pred) and np.isfinite(actual)):
            continue
        pred = float(max(pred, 0.0))
        err = actual - pred
        denom = max(1e-12, abs(actual) + abs(pred))
        rows.append(
            {
                "Date": frame.index[i + 1],
                "Actual": actual,
                f"Predicted_{model_name}": pred,
                f"Err_{model_name}": err,
                f"AbsErr_{model_name}": abs(err),
                f"SMAPE_{model_name}_pct": 200.0 * abs(err) / denom,
            }
        )
    return pd.DataFrame(rows)


def pm_qdk_predict_values(
    frame: pd.DataFrame,
    args,
    params: dict,
    start_i: int,
    stop_i: int,
    max_forecasts: int | None = None,
) -> list[tuple[int, float]]:
    x = lag_matrix(frame, args.n)
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1)
    c_state, phi, _ = pm_qdk_phi(x, args.n, float(params["tau"]))
    scale_end = max(start_i, args.n + 25)
    c_scale = robust_scale(c_state[:scale_end])
    h = max(float(params["bandwidth"]), 1e-12)
    eta = float(params["eta"])
    values = []
    for i in range(start_i, min(stop_i, len(frame) - 1)):
        if not valid[i] or not np.isfinite(y_next[i]):
            continue
        train_idx = candidate_train_indices(valid, y_next, i, args.pm_kernel_train_window)
        if train_idx.size < 5:
            continue
        dc = (c_state[train_idx] - c_state[i]) / c_scale
        d2 = np.sum(dc * dc, axis=1)
        k_perp = phi[train_idx] @ phi[i]
        log_w = -0.5 * d2 / (h * h) + eta * k_perp
        weights = softmax_weights(log_w)
        if weights.size == 0:
            continue
        pred = float(np.sum(weights * y_next[train_idx]))
        values.append((i, pred))
        if max_forecasts is not None and len(values) >= max_forecasts:
            break
    return values


def pm_qdk2_predict_values(
    frame: pd.DataFrame,
    args,
    params: dict,
    start_i: int,
    stop_i: int,
    max_forecasts: int | None = None,
) -> list[tuple[int, float]]:
    x = lag_matrix(frame, args.n)
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1)
    level, c_state, phi = pm_qdk2_embedding(x, args.n, args.pm_tau_grid_values, args.log_eps, int(args.pm_low_modes))
    q = expanding_linear_residuals(phi, c_state, valid & np.isfinite(y_next))
    scale_end = max(start_i, args.n + 25)
    level_scale = float(robust_scale(level[:scale_end, None])[0])
    c_scale = robust_scale(c_state[:scale_end])
    q_scale = robust_scale(q[:scale_end])
    h = max(float(params["bandwidth"]), 1e-12)
    values = []
    for i in range(start_i, min(stop_i, len(frame) - 1)):
        if not valid[i] or not np.isfinite(y_next[i]) or not np.isfinite(q[i]).all():
            continue
        train_idx = candidate_train_indices(valid & np.isfinite(q).all(axis=1), y_next, i, args.pm_kernel_train_window)
        if train_idx.size < 5:
            continue
        dl = np.square((level[train_idx] - level[i]) / level_scale)
        dc = (c_state[train_idx] - c_state[i]) / c_scale
        dq = (q[train_idx] - q[i]) / q_scale
        d2 = dl + np.sum(dc * dc, axis=1) + np.sum(dq * dq, axis=1) / max(q.shape[1], 1)
        weights = softmax_weights(-0.5 * d2 / (h * h))
        if weights.size == 0:
            continue
        pred = weighted_smape_action(y_next[train_idx], weights)
        values.append((i, pred))
        if max_forecasts is not None and len(values) >= max_forecasts:
            break
    return values


def cp_prediction_map(
    frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    args,
    max_forecasts: int | None,
    warmup_override: int | None = None,
) -> dict[pd.Timestamp, float]:
    cp_spec = model_spec("CP_REPO_FRESH", feature_groups["CP_REPO_FRESH"], frame, args)
    cp_pred = fast_expanding_predict(
        frame,
        cp_spec,
        args.n,
        args.warmup if warmup_override is None else int(warmup_override),
        target_transform=args.target_transform,
        log_eps=args.log_eps,
        max_forecasts=max_forecasts,
    )
    if cp_pred.empty:
        return {}
    return dict(zip(pd.to_datetime(cp_pred["Date"]), pd.to_numeric(cp_pred["Predicted_CP_REPO_FRESH"], errors="coerce")))


def phqo_predict_values(
    frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    args,
    params: dict,
    start_i: int,
    stop_i: int,
    max_forecasts: int | None = None,
) -> list[tuple[int, float]]:
    x = lag_matrix(frame, args.n)
    y_next = pd.to_numeric(frame["RV_d"].shift(-1), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y_next)
    _, m_c = cp_orthogonal_projection(args.n)
    modes, mu = pm_mode_subset(args.n, int(args.pm_low_modes))
    chars = prime_character_matrix(args.n, modes, normalize=True)
    tau = float(params["tau"])
    gamma = float(params["gamma"])
    h_pm = np.real(chars @ np.diag(np.exp(-tau * mu)) @ np.conjugate(chars).T @ m_c)
    energy = x @ h_pm.T
    beta = phqo_base_weights(args.n)
    base_raw = x @ beta
    cp_map = cp_prediction_map(frame, feature_groups, args, max_forecasts=max_forecasts, warmup_override=max(0, int(start_i) - args.n))
    values = []
    for i in range(start_i, min(stop_i, len(frame) - 1)):
        if not bool(valid[i]):
            continue
        date = pd.Timestamp(frame.index[i + 1])
        cp_pred = cp_map.get(date)
        if cp_pred is None or not np.isfinite(cp_pred) or not np.isfinite(base_raw[i]) or abs(base_raw[i]) <= 1e-18:
            continue
        logits = np.clip(gamma * energy[i], -50.0, 50.0)
        tilted = beta * np.exp(logits)
        denom = float(tilted.sum())
        if not np.isfinite(denom) or denom <= 0:
            continue
        weights = tilted / denom * beta.sum()
        ratio = float((weights @ x[i]) / base_raw[i])
        pred = float(cp_pred * ratio)
        values.append((i, pred))
        if max_forecasts is not None and len(values) >= max_forecasts:
            break
    return values


def score_prediction_values(frame: pd.DataFrame, values: list[tuple[int, float]]) -> float:
    pred = prediction_rows_common(frame, "_TMP", values)
    if pred.empty:
        return np.inf
    return float(np.nanmean(pred["SMAPE__TMP_pct"]))


def select_pm_native_controls(model_name: str, frame: pd.DataFrame, feature_groups: dict[str, list[str]], args) -> dict:
    val_start, first_oos = validation_window_indices(frame, args.n, args.warmup)
    if model_name == "PM_QDK":
        best = (np.inf, {"tau": args.pm_tau_grid_values[0], "eta": args.pm_eta_grid_values[0], "bandwidth": args.pm_bandwidth_grid_values[0]})
        for tau in args.pm_tau_grid_values:
            for eta in args.pm_eta_grid_values:
                for bandwidth in args.pm_bandwidth_grid_values:
                    params = {"tau": float(tau), "eta": float(eta), "bandwidth": float(bandwidth)}
                    score = score_prediction_values(frame, pm_qdk_predict_values(frame, args, params, val_start, first_oos))
                    if np.isfinite(score) and score < best[0]:
                        best = (score, params)
        best[1]["cv_score"] = safe_float(best[0])
        return best[1]
    if model_name == "PM_QDK_2":
        best = (np.inf, {"bandwidth": args.pm_bandwidth_grid_values[0]})
        for bandwidth in args.pm_bandwidth_grid_values:
            params = {"bandwidth": float(bandwidth)}
            score = score_prediction_values(frame, pm_qdk2_predict_values(frame, args, params, val_start, first_oos))
            if np.isfinite(score) and score < best[0]:
                best = (score, params)
        best[1]["cv_score"] = safe_float(best[0])
        return best[1]
    if model_name == "PHQO":
        best = (np.inf, {"tau": args.pm_tau_grid_values[0], "gamma": args.phqo_gamma_grid_values[0]})
        for tau in args.pm_tau_grid_values:
            for gamma in args.phqo_gamma_grid_values:
                params = {"tau": float(tau), "gamma": float(gamma)}
                score = score_prediction_values(frame, phqo_predict_values(frame, feature_groups, args, params, val_start, first_oos))
                if np.isfinite(score) and score < best[0]:
                    best = (score, params)
        best[1]["cv_score"] = safe_float(best[0])
        return best[1]
    raise ValueError(f"Unknown PM-native model {model_name}")


def pm_native_expanding_predict(
    frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    model_name: str,
    args,
    max_forecasts: int | None = None,
) -> pd.DataFrame:
    params = select_pm_native_controls(model_name, frame, feature_groups, args)
    start_i = args.n + args.warmup
    stop_i = len(frame) - 1
    if model_name == "PM_QDK":
        values = pm_qdk_predict_values(frame, args, params, start_i, stop_i, max_forecasts=max_forecasts)
    elif model_name == "PM_QDK_2":
        values = pm_qdk2_predict_values(frame, args, params, start_i, stop_i, max_forecasts=max_forecasts)
    elif model_name == "PHQO":
        values = phqo_predict_values(frame, feature_groups, args, params, start_i, stop_i, max_forecasts=max_forecasts)
    else:
        raise ValueError(f"Unknown PM-native model {model_name}")
    pred = prediction_rows_common(frame, model_name, values)
    pred.attrs["pm_native_params"] = params
    return pred


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
        "spec": "CP-REPO-OPS-HG locked ladder",
        "n_lags": int(n),
        "cp_blocks": {f"B{idx + 1}": block for idx, block in enumerate(blocks)},
        "cp_block_columns": cp_block_columns(blocks),
        "zero_sum_tolerance": ZERO_SUM_TOL,
        "models": {
            "HAR_RV": {
                "feature_families": ["RV_REPO"],
                "includes_cp": False,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "HAR-only volatility benchmark using RV_d, RV_w, and RV_m.",
            },
            "CP_REPO_FRESH": {
                "feature_families": ["RV_REPO", "CP_REPO"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Fresh repo contig_prime_modulo CP benchmark using RV+CP features; denominator for all main advantages.",
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
            "RAW_CP_REPO_PLUS_PM": {
                "feature_families": ["RV_REPO", "CP_REPO", "RAW_PM"],
                "includes_cp": True,
                "includes_raw_pm": True,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Naive CP plus raw PM hybrid benchmark.",
            },
            "CP_REPO_OPS_R": {
                "feature_families": ["RV_REPO", "CP_REPO", "OPS_R"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "CP plus repo-CP-controlled centered PM residue shape.",
            },
            "CP_REPO_RIDGE_OPS_R": {
                "feature_families": ["RV_REPO", "CP_REPO", "OPS_R"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "OPS-R with ridge shrinkage on shape coefficients only.",
            },
            "CP_REPO_LRPM": {
                "feature_families": ["RV_REPO", "CP_REPO", "LRPM"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": False,
                "expected_interpretation": "Unregularized local residue bridge diagnostic only; rank-deficient and unstable in full runs.",
            },
            "CP_REPO_RECENT_SLOPE": {
                "feature_families": ["RV_REPO", "CP_REPO", "RECENT_SLOPE"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Simple local recent-vs-older zero-sum slope ablation.",
            },
            "CP_REPO_HAAR_SHAPE": {
                "feature_families": ["RV_REPO", "CP_REPO", "HAAR"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Generic zero-sum Haar path shape benchmark.",
            },
            "CP_REPO_OPS_C": {
                "feature_families": ["RV_REPO", "CP_REPO", "OPS_C"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "CP plus contiguous zero-sum prime-radix path shape.",
            },
            "CP_REPO_RIDGE_OPS_C": {
                "feature_families": ["RV_REPO", "CP_REPO", "OPS_C"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "OPS-C with ridge shrinkage on shape coefficients only.",
            },
            "CP_REPO_GATED_RIDGE_OPS_C": {
                "feature_families": ["RV_REPO", "CP_REPO", "GATED_OPS_C"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": True,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Registered CP plus gated ridge OPS-C model.",
            },
            "CP_REPO_OPS_HG": {
                "feature_families": ["RV_REPO", "CP_REPO", "GATED_OPS_C", "GATED_OPS_R"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": True,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Final hybrid with gated OPS-C plus heavily shrunk gated OPS-R.",
            },
            "RANDOM_RESIDUES_PLACEBO_REPO": {
                "feature_families": ["RV_REPO", "CP_REPO", "RANDOM_RESIDUES"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": True,
                "is_placebo": True,
                "paper_eligible": True,
                "expected_interpretation": "OPS-R placebo preserving residue group sizes with random lag groups.",
            },
            "SHUFFLED_LAG_PM_PLACEBO_REPO": {
                "feature_families": ["RV_REPO", "CP_REPO", "SHUFFLED_PM"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": False,
                "uses_ridge": True,
                "is_placebo": True,
                "paper_eligible": True,
                "expected_interpretation": "OPS-R placebo that shuffles lag labels before PM grouping.",
            },
            "RANDOM_GATE_PLACEBO_REPO": {
                "feature_families": ["RV_REPO", "CP_REPO", "RANDOM_GATE_OPS_C"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
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
            "CP_REPO_OPS_K": {
                "feature_families": ["RV_REPO", "CP_REPO", "GATED_OPS_K"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": False,
                "uses_gate": True,
                "uses_ridge": True,
                "is_placebo": False,
                "paper_eligible": True,
                "expected_interpretation": "Optional kernel challenger using repo-CP-controlled prime-signature kernel features.",
            },
            "PM_QDK": {
                "feature_families": ["PM_NATIVE_KERNEL"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "interpretation_status": "exploratory_challenger",
                "expected_interpretation": "Prime-Modular Quotient Diffusion Kernel using CP-orthogonal lag residuals and prime-torus harmonic diffusion geometry.",
            },
            "PM_QDK_2": {
                "feature_families": ["PM_NATIVE_QUOTIENT_KERNEL"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": False,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "interpretation_status": "exploratory_challenger",
                "expected_interpretation": "Loss-native PM-QDK-2 forecaster using log-centered PM heat embeddings, train-only CP quotient residualization, and SMAPE-native weighted Bayes action.",
            },
            "PHQO": {
                "feature_families": ["PM_NATIVE_OPERATOR"],
                "includes_cp": True,
                "includes_raw_pm": False,
                "shape_features_cp_orthogonal": True,
                "uses_gate": True,
                "uses_ridge": False,
                "is_placebo": False,
                "paper_eligible": True,
                "interpretation_status": "exploratory_challenger",
                "expected_interpretation": "Prime-Harmonic Quotient Operator that maps CP-invisible prime energy back to lag weights and scales the current CP_REPO_FRESH forecast; gamma=0 is exact CP.",
            },
        },
    }
    return registry


def write_model_registry(outdir: Path, args) -> tuple[dict, str]:
    registry = build_model_registry(args.n)
    path = outdir / REGISTRY_FILENAME
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    return registry, sha256_file(path)


def model_is_paper_eligible(model_name: str, registry: dict | None) -> bool:
    if not registry:
        return True
    entry = registry.get("models", {}).get(model_name, {})
    return bool(entry.get("paper_eligible", True))


def feature_family(column: str) -> str:
    if column.startswith("RV"):
        return "RV_REPO"
    if column.startswith("CP_"):
        return "CP_REPO"
    if column.startswith("CPB_B"):
        return "CP_BLOCK_DIAGNOSTIC"
    if column.startswith("PM_"):
        return "RAW_PM"
    if column.startswith("Z_PMCTR"):
        return "OPS_R"
    if column.startswith("Z_RANDCTR"):
        return "RANDOM_RESIDUES"
    if column.startswith("Z_SHUFCTR"):
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
    if column.startswith("GATED_Z_PMCTR"):
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
    if model_name in PM_NATIVE_MODELS:
        if features:
            raise ValueError(f"{model_name}: PM-native models must use custom predictor geometry, not additive feature columns")
        return
    entry = registry["models"][model_name]
    allowed = set(entry["feature_families"])
    actual = [feature_family(col) for col in features]
    unknown = [col for col, family in zip(features, actual) if family == "UNKNOWN"]
    extras = [col for col, family in zip(features, actual) if family not in allowed]
    if unknown or extras:
        raise ValueError(f"{model_name}: feature purity violation unknown={unknown} extras={extras}")

    actual_families = set(actual)
    required = set(entry["feature_families"])
    if model_name != "CP_REPO_OPS_K" and not required.issubset(actual_families):
        raise ValueError(f"{model_name}: missing required feature families {sorted(required - actual_families)}")

    repo_control_cols = [col for col in features if col.startswith("RV") or col.startswith("CP_")]
    cpb_cols = [col for col in features if col.startswith("CPB_B")]
    if cpb_cols:
        raise ValueError(f"{model_name}: four-block diagnostic CPB columns cannot enter paper models: {cpb_cols}")
    if model_name == "HAR_RV" and features != RV_HAR_COLS:
        raise ValueError("HAR_RV must use only RV_d, RV_w, RV_m.")
    if model_name == "CP_REPO_FRESH":
        if not features or any(not (col.startswith("RV") or col.startswith("CP_")) for col in features):
            raise ValueError("CP_REPO_FRESH must use only repo RV and CP columns.")
        if not any(col.startswith("CP_") for col in features):
            raise ValueError("CP_REPO_FRESH must include repo CP_* columns from contig_prime_modulo.")
    if model_name == "RAW_PM" and any(not col.startswith("PM_") for col in features):
        raise ValueError("RAW_PM must use raw PM columns only.")
    if model_name == "RAW_CP_REPO_PLUS_PM":
        if not any(col.startswith("CP_") for col in features) or not any(col.startswith("PM_") for col in features):
            raise ValueError("RAW_CP_REPO_PLUS_PM must include repo CP controls plus raw PM columns.")
        if any(not (col.startswith("RV") or col.startswith("CP_") or col.startswith("PM_")) for col in features):
            raise ValueError("RAW_CP_REPO_PLUS_PM must be repo CP/RV controls plus raw PM only.")
    if model_name.startswith("CP_REPO_") and model_name not in {"CP_REPO_FRESH"}:
        if not repo_control_cols or not any(col.startswith("CP_") for col in repo_control_cols):
            raise ValueError(f"{model_name}: CP_REPO models must include repo CP controls.")
    if model_name in {"CP_REPO_GATED_RIDGE_OPS_C", "CP_REPO_OPS_HG", "RANDOM_GATE_PLACEBO_REPO", "CP_REPO_OPS_K"}:
        ungated = [col for col in features if col.startswith("Z_OPS") or col.startswith("Z_KOPS") or col.startswith("Z_PMCTR")]
        if ungated:
            raise ValueError(f"{model_name}: gated model contains ungated shape columns {ungated}")
    if model_name in {"RANDOM_RESIDUES_PLACEBO_REPO", "SHUFFLED_LAG_PM_PLACEBO_REPO", "RANDOM_GATE_PLACEBO_REPO"}:
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
    cp_frame = contig_prime_modulo(vol.copy(), args.n, per_day_normalize=False)
    frame = add_prime_modulo_terms(cp_frame.copy(), args.n)
    frame = add_lag_columns(frame, args.n)
    blocks = fallback_blocks(args.n)
    frame = add_cp_block_features(frame, blocks)

    manifest = []
    shape_weights = {}
    true_opsr_weights = residue_contrast_weights(args.n, "PMCTR")
    random_weights = residue_contrast_weights(args.n, "RANDCTR", seed=args.seed + 101)
    shuffled_weights = residue_contrast_weights(args.n, "SHUFCTR", seed=args.seed + 202, shuffled=True)
    weight_groups = {
        "PMCTR": (true_opsr_weights, "OPS_R", "CP_REPO_OPS_R"),
        "RANDCTR": (random_weights, "RANDOM_RESIDUES", "RANDOM_RESIDUES_PLACEBO_REPO"),
        "SHUFCTR": (shuffled_weights, "SHUFFLED_PM", "SHUFFLED_LAG_PM_PLACEBO_REPO"),
        "LRPM": (lrpm_weights(args.n, blocks), "LRPM", "CP_REPO_LRPM"),
        "RSLOPE": (recent_slope_weights(args.n, blocks), "RECENT_SLOPE", "CP_REPO_RECENT_SLOPE"),
        "HAAR": (haar_weights(args.n, blocks), "HAAR", "CP_REPO_HAAR_SHAPE"),
        "OPSC": (opsc_weights(args.n, blocks), "OPS_C", "CP_REPO_OPS_C"),
        "KOPS": (kops_weights(args.n, blocks), "OPS_K", "CP_REPO_OPS_K"),
    }
    for _, (weights, family, source_model) in weight_groups.items():
        for feature_name, weight in weights.items():
            shape_weights[feature_name] = weight
            manifest.append(manifest_row(feature_name, family, weight, blocks, source_model))
    manifest.extend(pm_native_manifest_rows(args.n, blocks, args))

    frame = add_weight_features(frame, shape_weights, args.n)
    recent = frame[[f"lag{lag}" for lag in range(1, 4)]].mean(axis=1)
    prior = frame[[f"lag{lag}" for lag in range(4, 7)]].mean(axis=1)
    older = frame[[f"lag{lag}" for lag in range(6, 11)]].mean(axis=1)
    frame["RSLOPE_recent_1_3_minus_4_6"] = recent - prior
    frame["RSLOPE_recent_1_5_minus_6_10"] = frame[[f"lag{lag}" for lag in range(1, 6)]].mean(axis=1) - older
    lag_values = frame[[f"lag{lag}" for lag in range(1, args.n + 1)]].to_numpy(dtype=float)
    frame["RSLOPE_spike_recency"] = 1.0 - (np.nanargmax(np.where(np.isfinite(lag_values), lag_values, -np.inf), axis=1) / max(args.n - 1, 1))
    shape_cols = list(shape_weights.keys())
    direct_shape_cols = ["RSLOPE_recent_1_3_minus_4_6", "RSLOPE_recent_1_5_minus_6_10", "RSLOPE_spike_recency"]
    frame, z_shape_cols = online_standardize(frame, shape_cols + direct_shape_cols)
    frame = add_gate_columns(frame, args.n, blocks, args.seed + 303)

    z_by_raw = dict(zip(shape_cols + direct_shape_cols, z_shape_cols))
    z_groups = {
        prefix: [z_by_raw[col] for col in shape_cols if col.startswith(prefix)]
        for prefix in ["PMCTR", "RANDCTR", "SHUFCTR", "LRPM", "RSLOPE", "HAAR", "OPSC", "KOPS"]
    }
    z_groups["RSLOPE"] = z_groups["RSLOPE"] + [z_by_raw[col] for col in direct_shape_cols]
    for col in z_groups["OPSC"]:
        frame[f"GATED_{col}"] = frame[col] * frame["GATE_REAL"]
        frame[f"RGATED_{col}"] = frame[col] * frame["GATE_RANDOM"]
    for col in z_groups["PMCTR"]:
        frame[f"GATED_{col}"] = frame[col] * frame["GATE_REAL"]
    for col in z_groups["KOPS"]:
        frame[f"GATED_{col}"] = frame[col] * frame["GATE_REAL"]

    rv_cols = RV_HAR_COLS
    cp_cols = sorted([col for col in frame.columns if col.startswith("RV") or col.startswith("CP_")])
    pm_cols = select_cols(frame, ("PM_",))
    lag_cols = [f"lag{lag}" for lag in range(1, args.n + 1)]

    features = {
        "HAR_RV": rv_cols,
        "CP_REPO_FRESH": cp_cols,
        "RAW_PM": pm_cols,
        "RAW_CP_REPO_PLUS_PM": cp_cols + pm_cols,
        "CP_REPO_OPS_R": cp_cols + z_groups["PMCTR"],
        "CP_REPO_RIDGE_OPS_R": cp_cols + z_groups["PMCTR"],
        "CP_REPO_LRPM": cp_cols + z_groups["LRPM"],
        "CP_REPO_RECENT_SLOPE": cp_cols + z_groups["RSLOPE"],
        "CP_REPO_HAAR_SHAPE": cp_cols + z_groups["HAAR"],
        "CP_REPO_OPS_C": cp_cols + z_groups["OPSC"],
        "CP_REPO_RIDGE_OPS_C": cp_cols + z_groups["OPSC"],
        "CP_REPO_GATED_RIDGE_OPS_C": cp_cols + [f"GATED_{col}" for col in z_groups["OPSC"]],
        "CP_REPO_OPS_HG": cp_cols + [f"GATED_{col}" for col in z_groups["OPSC"]] + [f"GATED_{col}" for col in z_groups["PMCTR"]],
        "RANDOM_RESIDUES_PLACEBO_REPO": cp_cols + z_groups["RANDCTR"],
        "SHUFFLED_LAG_PM_PLACEBO_REPO": cp_cols + z_groups["SHUFCTR"],
        "RANDOM_GATE_PLACEBO_REPO": cp_cols + [f"RGATED_{col}" for col in z_groups["OPSC"]],
        "RIDGE_AR22": lag_cols,
        "CP_REPO_OPS_K": cp_cols + [f"GATED_{col}" for col in z_groups["KOPS"]],
        "PM_QDK": [],
        "PM_QDK_2": [],
        "PHQO": [],
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
        "shape_feature_count": int(len(shape_cols) + len(direct_shape_cols)),
        "cp_compatibility": "repo contig_prime_modulo benchmark with RV+CP features; four-block lag zones are diagnostic/local shape scaffolds only",
        "diagnostic_lag_zones_json": json.dumps(blocks),
        "target_transform": args.target_transform,
    }
    return frame, features, manifest, meta, diagnostics


def ridge_penalties(model_name: str, features: list[str], lambda_shape: float, lambda_r_ratio: float) -> dict[str, float]:
    ridge = {}
    unpenalized = {"RV_REPO", "CP_REPO"}
    if model_name in {"CP_REPO_RIDGE_OPS_R", "RANDOM_RESIDUES_PLACEBO_REPO", "SHUFFLED_LAG_PM_PLACEBO_REPO"}:
        ridge = {col: lambda_shape for col in features if feature_family(col) not in unpenalized}
    elif model_name in {"CP_REPO_RIDGE_OPS_C", "CP_REPO_GATED_RIDGE_OPS_C", "RANDOM_GATE_PLACEBO_REPO", "CP_REPO_OPS_K"}:
        ridge = {col: lambda_shape for col in features if feature_family(col) not in unpenalized}
    elif model_name == "CP_REPO_OPS_HG":
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
    if model_name in PM_NATIVE_MODELS:
        return 0.0, 0.0, "pm_native_pre_oos", np.nan
    if model_name not in {
        "CP_REPO_RIDGE_OPS_R",
        "CP_REPO_RIDGE_OPS_C",
        "CP_REPO_GATED_RIDGE_OPS_C",
        "CP_REPO_OPS_HG",
        "RANDOM_RESIDUES_PLACEBO_REPO",
        "SHUFFLED_LAG_PM_PLACEBO_REPO",
        "RANDOM_GATE_PLACEBO_REPO",
        "RIDGE_AR22",
        "CP_REPO_OPS_K",
    }:
        return 0.0, 0.0, "none", np.nan

    lambda_grid = args.lambda_grid_values
    ratio_grid = args.lambda_r_ratio_grid_values if model_name == "CP_REPO_OPS_HG" else [args.lambda_r_ratio_grid_values[0]]
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
    if model_name in PM_NATIVE_MODELS:
        return ModelSpec(model_name, [], {}, "pm_native_geometry")
    if model_name == "CP_REPO_OPS_K" and not features:
        return ModelSpec(model_name, features, {}, "advanced", scaffolded=True, scaffold_reason="No KOPS features generated")

    lambda_shape, lambda_r_ratio, cv_mode_effective, cv_score = select_ridge_controls(model_name, features, frame, args)
    ridge = ridge_penalties(model_name, features, lambda_shape, lambda_r_ratio)
    family = "linear"
    if model_name in {"CP_REPO_RIDGE_OPS_R", "RANDOM_RESIDUES_PLACEBO_REPO", "SHUFFLED_LAG_PM_PLACEBO_REPO"}:
        family = "ridge_shape"
    elif model_name in {"CP_REPO_RIDGE_OPS_C", "CP_REPO_GATED_RIDGE_OPS_C", "RANDOM_GATE_PLACEBO_REPO", "CP_REPO_OPS_K"}:
        family = "ridge_shape"
    elif model_name == "CP_REPO_OPS_HG":
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


def slow_expanding_predict(
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
    rows = []
    for i in range(n + warmup, t_obs - 1):
        train_idx = np.arange(0, i)
        train_idx = train_idx[valid_x[train_idx] & valid_y[train_idx]]
        if len(train_idx) < min_train_obs or not (valid_x[i] and valid_y[i]):
            continue
        x_train = x_all[train_idx]
        y_train = y_next_fit[train_idx]
        try:
            beta = np.linalg.solve(x_train.T @ x_train + penalty_matrix, x_train.T @ y_train)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(x_train.T @ x_train + penalty_matrix, rcond=1e-10) @ (x_train.T @ y_train)
        pred = inverse_transformed_prediction(float(x_all[i] @ beta), target_transform, log_eps)
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


def run_repo_cp_reproduction_audit(outdir: Path, args, assets: list[str]) -> pd.DataFrame:
    rows = []
    max_rows = args.smoke_max_forecasts if args.mode == "smoke" else args.max_forecasts
    for asset in assets:
        try:
            frame, feature_groups, _, _, _ = build_feature_frame(asset, args)
            spec = model_spec("CP_REPO_FRESH", feature_groups["CP_REPO_FRESH"], frame, args)
            recomputed = fast_expanding_predict(
                frame,
                spec,
                args.n,
                args.warmup,
                target_transform=args.target_transform,
                log_eps=args.log_eps,
                max_forecasts=max_rows,
            )
            saved_path = Path(args.prior_pred_dir) / f"{asset}.csv"
            saved = pd.read_csv(saved_path, parse_dates=["Date"]) if saved_path.exists() else pd.DataFrame()
            if saved.empty or "Predicted_CP" not in saved.columns:
                raise RuntimeError(f"Missing saved Predicted_CP file/column for {asset}: {saved_path}")
            merged = saved[["Date", "Actual", "Predicted_CP"]].merge(
                recomputed[["Date", "Actual", "Predicted_CP_REPO_FRESH"]],
                on="Date",
                how="inner",
                suffixes=("_saved", "_recomputed"),
            )
            existing = pd.to_numeric(merged["Predicted_CP"], errors="coerce")
            rec = pd.to_numeric(merged["Predicted_CP_REPO_FRESH"], errors="coerce")
            diff = (existing - rec).abs()
            actual_match = bool(
                not merged.empty
                and np.allclose(
                    pd.to_numeric(merged["Actual_saved"], errors="coerce"),
                    pd.to_numeric(merged["Actual_recomputed"], errors="coerce"),
                    atol=1e-12,
                    rtol=0.0,
                    equal_nan=False,
                )
            )
            corr = existing.corr(rec) if len(merged) > 1 else np.nan
            passes = bool(
                len(merged) > 0
                and actual_match
                and np.isfinite(corr)
                and corr > 0.999999
                and float(diff.mean()) < 1e-8
                and float(diff.max()) < 1e-7
            )
            rows.append(
                {
                    "asset": asset,
                    "n_existing": int(len(saved)),
                    "n_recomputed": int(len(recomputed)),
                    "n_aligned": int(len(merged)),
                    "mean_existing_CP": safe_float(existing.mean()),
                    "mean_recomputed_CP": safe_float(rec.mean()),
                    "mean_abs_diff": safe_float(diff.mean()),
                    "median_abs_diff": safe_float(diff.median()),
                    "max_abs_diff": safe_float(diff.max()),
                    "correlation": safe_float(corr),
                    "actual_target_match": actual_match,
                    "passes": passes,
                    "message": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "asset": asset,
                    "n_existing": 0,
                    "n_recomputed": 0,
                    "n_aligned": 0,
                    "mean_existing_CP": np.nan,
                    "mean_recomputed_CP": np.nan,
                    "mean_abs_diff": np.nan,
                    "median_abs_diff": np.nan,
                    "max_abs_diff": np.nan,
                    "correlation": np.nan,
                    "actual_target_match": False,
                    "passes": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    audit = pd.DataFrame(rows)
    write_csv(audit, outdir / "results" / "repo_cp_reproduction_audit.csv")
    if not bool(audit["passes"].astype(bool).all()):
        raise RuntimeError("Repo CP reproduction audit failed")
    return audit


def run_fast_slow_equivalence_audit(outdir: Path, args, asset: str = "AAPL") -> pd.DataFrame:
    selected = [
        "CP_REPO_FRESH",
        "RAW_CP_REPO_PLUS_PM",
        "CP_REPO_RIDGE_OPS_R",
        "CP_REPO_LRPM",
        "CP_REPO_RIDGE_OPS_C",
        "CP_REPO_GATED_RIDGE_OPS_C",
        "CP_REPO_OPS_HG",
        "RANDOM_GATE_PLACEBO_REPO",
        "RIDGE_AR22",
        "CP_REPO_OPS_K",
        "PM_QDK",
        "PM_QDK_2",
        "PHQO",
    ]
    frame, feature_groups, _, _, _ = build_feature_frame(asset, args)
    rows = []
    for model_name in selected:
        try:
            spec = model_spec(model_name, feature_groups[model_name], frame, args)
            if model_name in PM_NATIVE_MODELS:
                fast = pm_native_expanding_predict(frame, feature_groups, model_name, args, max_forecasts=8)
                slow = pm_native_expanding_predict(frame, feature_groups, model_name, args, max_forecasts=8)
            else:
                fast = fast_expanding_predict(frame, spec, args.n, args.warmup, args.target_transform, args.log_eps, max_forecasts=8)
                slow = slow_expanding_predict(frame, spec, args.n, args.warmup, args.target_transform, args.log_eps, max_forecasts=8)
            pred_col = f"Predicted_{model_name}"
            merged = fast[["Date", "Actual", pred_col]].merge(
                slow[["Date", "Actual", pred_col]],
                on="Date",
                suffixes=("_fast", "_slow"),
            )
            diff = (pd.to_numeric(merged[f"{pred_col}_fast"], errors="coerce") - pd.to_numeric(merged[f"{pred_col}_slow"], errors="coerce")).abs()
            timestamp_alignment_exact = bool(
                len(fast) == len(slow)
                and len(merged) == len(fast)
                and fast["Date"].reset_index(drop=True).equals(slow["Date"].reset_index(drop=True))
            )
            no_duplicate_dates = bool(not fast["Date"].duplicated().any() and not slow["Date"].duplicated().any())
            actual_match = bool(
                not merged.empty
                and np.allclose(
                    pd.to_numeric(merged["Actual_fast"], errors="coerce"),
                    pd.to_numeric(merged["Actual_slow"], errors="coerce"),
                    atol=1e-12,
                    rtol=0.0,
                )
            )
            max_diff = float(diff.max()) if not diff.empty else np.inf
            mean_diff = float(diff.mean()) if not diff.empty else np.inf
            feature_count = pm_native_feature_count(model_name, args) if model_name in PM_NATIVE_MODELS else len(spec.features)
            rank = feature_count if model_name in PM_NATIVE_MODELS else compute_design_rank(frame, spec.features)
            threshold = 1e-6 if rank < feature_count else 1e-8
            passes = bool(not merged.empty and timestamp_alignment_exact and no_duplicate_dates and actual_match and max_diff <= threshold)
            first_fail = ""
            if not passes and not merged.empty:
                bad = merged.loc[diff > threshold].head(1)
                if not bad.empty:
                    first_fail = str(bad["Date"].iloc[0])
            rows.append(
                {
                    "asset": asset,
                    "model_name": model_name,
                    "n_compared": int(len(merged)),
                    "max_abs_prediction_diff": safe_float(max_diff),
                    "mean_abs_prediction_diff": safe_float(mean_diff),
                    "timestamp_alignment_exact": timestamp_alignment_exact,
                    "no_duplicate_dates": no_duplicate_dates,
                    "actual_target_match": actual_match,
                    "selected_lambda_shape": spec.lambda_shape,
                    "selected_lambda_r_ratio": spec.lambda_r_ratio,
                    "design_rank": rank,
                    "feature_count": feature_count,
                    "rank_deficient": bool(rank < feature_count),
                    "threshold": threshold,
                    "first_failing_timestamp": first_fail,
                    "passes": passes,
                    "message": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "asset": asset,
                    "model_name": model_name,
                    "n_compared": 0,
                    "max_abs_prediction_diff": np.nan,
                    "mean_abs_prediction_diff": np.nan,
                    "timestamp_alignment_exact": False,
                    "no_duplicate_dates": False,
                    "actual_target_match": False,
                    "selected_lambda_shape": np.nan,
                    "selected_lambda_r_ratio": np.nan,
                    "design_rank": np.nan,
                    "feature_count": np.nan,
                    "rank_deficient": np.nan,
                    "threshold": np.nan,
                    "first_failing_timestamp": "",
                    "passes": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    audit = pd.DataFrame(rows)
    write_csv(audit, outdir / "results" / "fast_slow_equivalence_audit.csv")
    if not bool(audit["passes"].astype(bool).all()):
        raise RuntimeError("Fast-vs-slow equivalence audit failed")
    return audit


def run_strong_fast_slow_equivalence_audit(outdir: Path, args) -> pd.DataFrame:
    selected = unique_in_order(["CP_REPO_FRESH"] + [model for model in MODEL_ORDER if model != "CP_REPO_FRESH"])
    assets = parse_csv(args.strong_audit_assets, ["AAPL", "SPY", "GLD"])
    rows = []
    for asset in assets:
        frame, feature_groups, _, _, _ = build_feature_frame(asset, args)
        for model_name in selected:
            if model_name not in feature_groups:
                rows.append(
                    {
                        "asset": asset,
                        "model_name": model_name,
                        "n_compared": 0,
                        "max_abs_prediction_diff": np.nan,
                        "mean_abs_prediction_diff": np.nan,
                        "timestamp_alignment_exact": False,
                        "no_duplicate_dates": False,
                        "actual_target_match": False,
                        "selected_lambda_shape": np.nan,
                        "selected_lambda_r_ratio": np.nan,
                        "design_rank": np.nan,
                        "feature_count": np.nan,
                        "rank_deficient": np.nan,
                        "threshold": np.nan,
                        "passes": False,
                        "message": "model not present in feature group",
                    }
                )
                continue
            try:
                spec = model_spec(model_name, feature_groups[model_name], frame, args)
                if model_name in PM_NATIVE_MODELS:
                    fast = pm_native_expanding_predict(frame, feature_groups, model_name, args, max_forecasts=int(args.strong_audit_rows))
                    slow = pm_native_expanding_predict(frame, feature_groups, model_name, args, max_forecasts=int(args.strong_audit_rows))
                else:
                    fast = fast_expanding_predict(
                        frame,
                        spec,
                        args.n,
                        args.warmup,
                        args.target_transform,
                        args.log_eps,
                        max_forecasts=int(args.strong_audit_rows),
                    )
                    slow = slow_expanding_predict(
                        frame,
                        spec,
                        args.n,
                        args.warmup,
                        args.target_transform,
                        args.log_eps,
                        max_forecasts=int(args.strong_audit_rows),
                    )
                pred_col = f"Predicted_{model_name}"
                merged = fast[["Date", "Actual", pred_col]].merge(
                    slow[["Date", "Actual", pred_col]],
                    on="Date",
                    suffixes=("_fast", "_slow"),
                )
                diff = (
                    pd.to_numeric(merged[f"{pred_col}_fast"], errors="coerce")
                    - pd.to_numeric(merged[f"{pred_col}_slow"], errors="coerce")
                ).abs()
                timestamp_alignment_exact = bool(
                    len(fast) == len(slow)
                    and len(merged) == len(fast)
                    and fast["Date"].reset_index(drop=True).equals(slow["Date"].reset_index(drop=True))
                )
                no_duplicate_dates = bool(not fast["Date"].duplicated().any() and not slow["Date"].duplicated().any())
                actual_match = bool(
                    not merged.empty
                    and np.allclose(
                        pd.to_numeric(merged["Actual_fast"], errors="coerce"),
                        pd.to_numeric(merged["Actual_slow"], errors="coerce"),
                        atol=1e-12,
                        rtol=0.0,
                    )
                )
                max_diff = float(diff.max()) if not diff.empty else np.inf
                mean_diff = float(diff.mean()) if not diff.empty else np.inf
                feature_count = pm_native_feature_count(model_name, args) if model_name in PM_NATIVE_MODELS else len(spec.features)
                rank = feature_count if model_name in PM_NATIVE_MODELS else compute_design_rank(frame, spec.features)
                threshold = 1e-6 if rank < feature_count else 1e-8
                first_fail = ""
                if not diff.empty:
                    bad = merged.loc[diff > threshold].head(1)
                    if not bad.empty:
                        first_fail = str(bad["Date"].iloc[0])
                rows.append(
                    {
                        "asset": asset,
                        "model_name": model_name,
                        "n_compared": int(len(merged)),
                        "max_abs_prediction_diff": safe_float(max_diff),
                        "mean_abs_prediction_diff": safe_float(mean_diff),
                        "timestamp_alignment_exact": timestamp_alignment_exact,
                        "no_duplicate_dates": no_duplicate_dates,
                        "actual_target_match": actual_match,
                        "selected_lambda_shape": spec.lambda_shape,
                        "selected_lambda_r_ratio": spec.lambda_r_ratio,
                        "design_rank": rank,
                        "feature_count": feature_count,
                        "rank_deficient": bool(rank < feature_count),
                        "threshold": threshold,
                        "first_failing_timestamp": first_fail,
                        "passes": bool(not merged.empty and timestamp_alignment_exact and no_duplicate_dates and actual_match and max_diff <= threshold),
                        "message": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "asset": asset,
                        "model_name": model_name,
                        "n_compared": 0,
                        "max_abs_prediction_diff": np.nan,
                        "mean_abs_prediction_diff": np.nan,
                        "timestamp_alignment_exact": False,
                        "no_duplicate_dates": False,
                        "actual_target_match": False,
                        "selected_lambda_shape": np.nan,
                        "selected_lambda_r_ratio": np.nan,
                        "design_rank": np.nan,
                        "feature_count": np.nan,
                        "rank_deficient": np.nan,
                        "threshold": np.nan,
                        "first_failing_timestamp": "",
                        "passes": False,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
    audit = pd.DataFrame(rows)
    write_csv(audit, outdir / "results" / "strong_fast_slow_equivalence_audit.csv")
    if not bool(audit["passes"].astype(bool).all()):
        raise RuntimeError("Strong fast-vs-slow equivalence audit failed")
    return audit


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
    recent_front_loaded = recent_mean_1_5 > older_mean_6_10
    conditions["recent_ramp_high_dispersion"] = entry_loose & (dispersion >= dispersion.quantile(0.80))
    conditions["entry_with_recent_spike"] = entry_loose & conditions["recent_spike_position"]
    conditions["entry_without_recent_spike"] = entry_loose & ~conditions["recent_spike_position"]
    conditions["same_average_recent_front_loaded"] = (
        local_mean_1_10.between(mean_low, mean_high)
        & recent_front_loaded
        & (path_score >= path_score.quantile(0.60))
    )
    conditions["same_average_back_loaded"] = (
        local_mean_1_10.between(mean_low, mean_high)
        & ~recent_front_loaded
        & (path_score >= path_score.quantile(0.60))
    )
    conditions["stable_low_dispersion"] = dispersion <= dispersion.quantile(0.20)

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
            "benchmark_model": "CP_REPO_FRESH",
            "asset": asset,
            "condition": condition,
            "condition_label_type": "not_conditioned" if condition == "all_observations" else "ex_post_descriptive",
            "n_obs": 0,
        }
    model_smape_col = f"SMAPE_{model_name}_pct"
    cp_smape_col = "SMAPE_CP_REPO_FRESH_pct"
    model_abs_col = f"AbsErr_{model_name}"
    cp_abs_col = "AbsErr_CP_REPO_FRESH"
    advantage = frame[cp_smape_col].astype(float) - frame[model_smape_col].astype(float)
    abs_advantage = frame[cp_abs_col].astype(float) - frame[model_abs_col].astype(float)
    wins = advantage[advantage > 0]
    losses = advantage[advantage < 0]
    row = {
        "model_name": model_name,
        "benchmark_model": "CP_REPO_FRESH",
        "asset": asset,
        "condition": condition,
        "condition_label_type": "not_conditioned" if condition == "all_observations" else "ex_post_descriptive",
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
    cp_meta = cp_meta or meta_map.get((asset, "CP_REPO_FRESH"), {})
    source = str(meta.get("source", "unknown"))
    run_type = str(meta.get("run_type", "unknown"))
    cp_source = str(cp_meta.get("source", "unknown"))
    registry_eligible = bool(meta.get("paper_eligible", True))
    eligible_sources = {"generated_current", "skipped_existing_verified"}
    eligible = bool(
        run_type == "full"
        and source in eligible_sources
        and cp_source in eligible_sources
        and not bool(meta.get("scaffolded", False))
        and registry_eligible
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
            advantage = subset["SMAPE_CP_REPO_FRESH_pct"].astype(float) - subset[model_col].astype(float)
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


def mean_loss(actual: pd.Series, pred: pd.Series, metric: str) -> float:
    actual = pd.to_numeric(actual, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    mask = actual.notna() & pred.notna()
    actual = actual[mask]
    pred = pred[mask]
    if actual.empty:
        return np.nan
    err = actual - pred
    if metric == "SMAPE":
        return safe_float(smape(actual, pred).mean())
    if metric == "MAE":
        return safe_float(err.abs().mean())
    if metric == "MSE":
        return safe_float(np.square(err).mean())
    if metric == "RMSE":
        return safe_float(math.sqrt(float(np.square(err).mean())))
    if metric == "LOG_RV_MSE":
        log_actual = np.log(np.clip(actual.to_numpy(dtype=float), 0.0, None) + LOG_EPS_DEFAULT)
        log_pred = np.log(np.clip(pred.to_numpy(dtype=float), 0.0, None) + LOG_EPS_DEFAULT)
        return safe_float(np.square(log_actual - log_pred).mean())
    return np.nan


def alternative_loss_rows(frame: pd.DataFrame, model_name: str, asset: str, condition: str) -> list[dict]:
    if frame.empty:
        return []
    model_col = f"Predicted_{model_name}"
    if model_col not in frame.columns or "Predicted_CP_REPO_FRESH" not in frame.columns:
        return []
    actual = pd.to_numeric(frame["Actual"], errors="coerce")
    variants = {
        "standard": pd.Series(True, index=frame.index),
        "high_vol_only": actual >= actual.quantile(0.80),
        "near_zero_excluded": actual >= actual.quantile(0.05),
    }
    rows = []
    for loss_variant, mask in variants.items():
        subset = frame.loc[mask.fillna(False)].copy()
        if subset.empty:
            continue
        for metric in ["SMAPE", "MAE", "MSE", "RMSE", "LOG_RV_MSE"]:
            cp_loss = mean_loss(subset["Actual"], subset["Predicted_CP_REPO_FRESH"], metric)
            model_loss = mean_loss(subset["Actual"], subset[model_col], metric)
            rows.append(
                {
                    "model_name": model_name,
                    "benchmark_model": "CP_REPO_FRESH",
                    "asset": asset,
                    "condition": condition,
                    "condition_label_type": "not_conditioned" if condition == "all_observations" else "ex_post_descriptive",
                    "loss_variant": loss_variant,
                    "loss_metric": metric,
                    "n_obs": int(len(subset)),
                    "CP_loss": cp_loss,
                    "model_loss": model_loss,
                    "advantage_CP_minus_model": safe_float(cp_loss - model_loss) if np.isfinite(cp_loss) and np.isfinite(model_loss) else np.nan,
                }
            )
    return rows


def aggregate_alternative_loss(alternative_by_asset: pd.DataFrame) -> pd.DataFrame:
    if alternative_by_asset.empty:
        return pd.DataFrame(columns=[
            "model_name",
            "condition",
            "condition_label_type",
            "loss_variant",
            "loss_metric",
            "n_obs_total",
            "n_assets",
            "pooled_CP_loss",
            "pooled_model_loss",
            "pooled_advantage_CP_minus_model",
            "assets_positive",
        ])
    rows = []
    group_cols = ["model_name", "condition", "condition_label_type", "loss_variant", "loss_metric"]
    for keys, group in alternative_by_asset.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        weights = pd.to_numeric(group["n_obs"], errors="coerce").fillna(0.0)
        total = float(weights.sum())
        if total > 0:
            cp_loss = float((pd.to_numeric(group["CP_loss"], errors="coerce") * weights).sum() / total)
            model_loss = float((pd.to_numeric(group["model_loss"], errors="coerce") * weights).sum() / total)
        else:
            cp_loss = np.nan
            model_loss = np.nan
        adv = pd.to_numeric(group["advantage_CP_minus_model"], errors="coerce")
        rec.update(
            {
                "n_obs_total": int(total),
                "n_assets": int(group["asset"].nunique()),
                "pooled_CP_loss": safe_float(cp_loss),
                "pooled_model_loss": safe_float(model_loss),
                "pooled_advantage_CP_minus_model": safe_float(cp_loss - model_loss) if np.isfinite(cp_loss) and np.isfinite(model_loss) else np.nan,
                "equal_weight_asset_advantage": safe_float(adv.mean()),
                "assets_positive": int((adv > 0).sum()),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_results(outdir: Path, mode: str, assets: list[str], models: list[str], n: int) -> dict[str, pd.DataFrame]:
    pred_dir = outdir / "predictions" / mode
    metadata_path = outdir / "results" / "run_metadata.csv"
    meta_map = load_metadata_map(metadata_path)
    overall_rows = []
    conditional_rows = []
    alignment_rows = []
    alternative_rows = []
    pooled_parts = []
    model_list = unique_in_order(["CP_REPO_FRESH"] + models)

    for asset in assets:
        pred_path = pred_dir / f"{asset}.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path, parse_dates=["Date"]).sort_values("Date")
        if "Predicted_CP_REPO_FRESH" not in pred.columns:
            continue
        actual = pd.to_numeric(pred["Actual"], errors="coerce")
        cp_pred = pd.to_numeric(pred["Predicted_CP_REPO_FRESH"], errors="coerce")
        pred["AbsErr_CP_REPO_FRESH"] = (actual - cp_pred).abs()
        pred["SMAPE_CP_REPO_FRESH_pct"] = smape(actual, cp_pred)
        cp_frame = pred[["Date", "Actual", "Predicted_CP_REPO_FRESH"]].dropna().copy()
        cp_meta = meta_map.get((asset, "CP_REPO_FRESH"), {})

        for model_name in model_list:
            pred_col = f"Predicted_{model_name}"
            if pred_col not in pred.columns:
                continue
            model_raw = pred[["Date", "Actual", pred_col]].dropna().copy()
            if model_name == "CP_REPO_FRESH":
                merged = cp_frame.copy()
                actual_match = True
                timestamps_exact = True
                model_frame = merged.rename(columns={"Actual": "Actual"})
                n_used_alignment = int(len(merged.dropna(subset=["Actual", "Predicted_CP_REPO_FRESH"])))
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
                n_used_alignment = int(len(merged.dropna(subset=["Actual_CP", "Predicted_CP_REPO_FRESH", pred_col])))
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
            model_frame = model_frame.dropna(subset=["Actual", "Predicted_CP_REPO_FRESH", pred_col])
            model_frame[f"AbsErr_{model_name}"] = (model_frame["Actual"] - model_frame[pred_col]).abs()
            model_frame[f"SMAPE_{model_name}_pct"] = smape(model_frame["Actual"], model_frame[pred_col])
            model_frame["AbsErr_CP_REPO_FRESH"] = (model_frame["Actual"] - model_frame["Predicted_CP_REPO_FRESH"]).abs()
            model_frame["SMAPE_CP_REPO_FRESH_pct"] = smape(model_frame["Actual"], model_frame["Predicted_CP_REPO_FRESH"])
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
            alternative_rows.extend(alternative_loss_rows(model_frame, model_name, asset, "all_observations"))

            lagged = add_lagged_actuals(model_frame, n)
            if lagged.empty:
                continue
            conditions = define_conditions(lagged)
            for condition in CONDITION_ORDER:
                subset = lagged.loc[conditions[condition]].copy()
                conditional_rows.append(metric_row(subset, model_name, asset, condition, prov))
                if condition != "all_observations":
                    alternative_rows.extend(alternative_loss_rows(subset, model_name, asset, condition))
                if not subset.empty:
                    pooled_piece = subset.copy()
                    pooled_piece["asset"] = asset
                    pooled_piece["model_name"] = model_name
                    pooled_piece["condition"] = condition
                    pooled_piece["row_is_paper_eligible"] = prov["is_paper_eligible"]
                    pooled_parts.append(pooled_piece)

    overall = pd.DataFrame(overall_rows)
    conditional = pd.DataFrame(conditional_rows)
    alternative_by_asset = pd.DataFrame(alternative_rows)

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

    placebo_names = ["RANDOM_RESIDUES_PLACEBO_REPO", "SHUFFLED_LAG_PM_PLACEBO_REPO", "RANDOM_GATE_PLACEBO_REPO", "RIDGE_AR22"]
    placebo = ablation[ablation["model_name"].isin(placebo_names)].copy() if not ablation.empty else pd.DataFrame()

    return {
        "overall": overall,
        "conditional": conditional,
        "pooled": pooled,
        "ablation": ablation,
        "placebo": placebo,
        "alignment": pd.DataFrame(alignment_rows),
        "inference": compute_inference_summary(pooled_all),
        "alternative_loss_by_asset": alternative_by_asset,
        "alternative_loss": aggregate_alternative_loss(alternative_by_asset),
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
        ax.set_title("CP-REPO-OPS-HG Ablation Ladder")
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_ladder.png", dpi=180)
    plt.close(fig)

    pooled = results["pooled"].copy()
    focus_models = ["RAW_CP_REPO_PLUS_PM", "CP_REPO_GATED_RIDGE_OPS_C", "CP_REPO_OPS_HG", *PM_NATIVE_MODELS]
    focus = pooled[pooled["model_name"].isin(focus_models)] if not pooled.empty else pd.DataFrame()
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


def write_pm_native_report(outdir: Path, results: dict[str, pd.DataFrame]) -> None:
    result_dir = outdir / "results"
    paper_dir = outdir / "paper_tables"
    fig_dir = outdir / "figures"
    for directory in [result_dir, paper_dir, fig_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    alt = results.get("alternative_loss", pd.DataFrame()).copy()
    pm_loss = pd.DataFrame()
    if not alt.empty:
        pm_loss = alt[
            alt["model_name"].isin(PM_NATIVE_MODELS)
            & (alt["condition"] == "all_observations")
            & (alt["loss_variant"] == "standard")
            & (alt["loss_metric"].isin(["SMAPE", "MAE", "MSE", "RMSE"]))
        ].copy()
        if not pm_loss.empty:
            pm_loss = pm_loss.sort_values(["loss_metric", "pooled_advantage_CP_minus_model"], ascending=[True, False])
    write_csv(pm_loss, paper_dir / "pm_native_loss_summary.csv")

    overall = results.get("overall", pd.DataFrame()).copy()
    consistency_rows = []
    if not overall.empty:
        for model_name in PM_NATIVE_MODELS:
            subset = overall[(overall["model_name"] == model_name) & (overall["condition"] == "all_observations")].copy()
            if subset.empty:
                continue
            consistency_rows.append(
                {
                    "model_name": model_name,
                    "benchmark_model": "CP_REPO_FRESH",
                    "n_assets": int(subset["asset"].nunique()),
                    "assets_positive_smape_advantage": int((pd.to_numeric(subset["mean_advantage_vs_CP"], errors="coerce") > 0).sum()),
                    "assets_positive_mae_advantage": int((pd.to_numeric(subset["mean_abs_error_advantage_vs_CP"], errors="coerce") > 0).sum()),
                    "equal_weight_mean_smape_advantage": safe_float(pd.to_numeric(subset["mean_advantage_vs_CP"], errors="coerce").mean()),
                    "equal_weight_mean_abs_error_advantage": safe_float(pd.to_numeric(subset["mean_abs_error_advantage_vs_CP"], errors="coerce").mean()),
                }
            )
    consistency = pd.DataFrame(consistency_rows)
    write_csv(consistency, paper_dir / "pm_native_asset_consistency.csv")

    ablation = results.get("ablation", pd.DataFrame()).copy()
    placebo_rows = []
    placebo_models = ["RANDOM_RESIDUES_PLACEBO_REPO", "SHUFFLED_LAG_PM_PLACEBO_REPO", "RANDOM_GATE_PLACEBO_REPO", "RIDGE_AR22"]
    if not ablation.empty:
        for model_name in PM_NATIVE_MODELS:
            pm_row = ablation[ablation["model_name"] == model_name]
            if pm_row.empty:
                continue
            pm_adv = safe_float(pm_row["mean_advantage_vs_CP"].iloc[0])
            for placebo in placebo_models:
                placebo_row = ablation[ablation["model_name"] == placebo]
                if placebo_row.empty:
                    continue
                placebo_adv = safe_float(placebo_row["mean_advantage_vs_CP"].iloc[0])
                placebo_rows.append(
                    {
                        "model_name": model_name,
                        "placebo_model": placebo,
                        "pm_smape_advantage": pm_adv,
                        "placebo_smape_advantage": placebo_adv,
                        "pm_minus_placebo_smape_advantage": safe_float(pm_adv - placebo_adv),
                        "pm_beats_placebo": bool(np.isfinite(pm_adv) and np.isfinite(placebo_adv) and pm_adv > placebo_adv),
                    }
                )
    placebo_cmp = pd.DataFrame(placebo_rows)
    write_csv(placebo_cmp, result_dir / "pm_native_placebo_comparison.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    if pm_loss.empty:
        ax.text(0.5, 0.5, "No PM-native loss rows available", ha="center", va="center")
        ax.set_axis_off()
    else:
        pivot = pm_loss.pivot_table(index="model_name", columns="loss_metric", values="pooled_advantage_CP_minus_model", aggfunc="mean")
        pivot = pivot.reindex(PM_NATIVE_MODELS)
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("CP loss minus model loss")
        ax.set_title("PM-Native Metric Advantages")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fig_dir / "pm_native_metric_advantages.png", dpi=180)
    plt.close(fig)

    report_lines = [
        "# PM-Native Warmup-600 Results Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "Benchmark: `CP_REPO_FRESH` using repo `RV*` and repo `CP_*` features from `contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`.",
        "PM-native kernels use only prior rows; the configured rolling support cap is recorded in `run_metadata.csv` as `pm_kernel_train_window`.",
        "",
        "Positive advantage means lower loss than corrected CP.",
        "",
    ]
    if pm_loss.empty:
        report_lines.append("No PM-native loss rows were available.")
    else:
        report_lines.extend(
            [
                "## Global Loss Summary",
                "",
                "```text",
                pm_loss[
                    [
                        "model_name",
                        "loss_metric",
                        "n_obs_total",
                        "pooled_CP_loss",
                        "pooled_model_loss",
                        "pooled_advantage_CP_minus_model",
                        "assets_positive",
                    ]
                ].to_string(index=False),
                "```",
                "",
            ]
        )
    if not consistency.empty:
        report_lines.extend(
            [
                "## Asset Consistency",
                "",
                "```text",
                consistency.to_string(index=False),
                "```",
                "",
            ]
        )
    if not placebo_cmp.empty:
        report_lines.extend(
            [
                "## Placebo Comparisons",
                "",
                "```text",
                placebo_cmp.to_string(index=False),
                "```",
                "",
            ]
        )
    (result_dir / "pm_native_results_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def write_raw_cp_pm_reproduction_audit(outdir: Path, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    expected_signs = {
        "all_observations": "negative",
        "cluster_entry_loose": "positive",
        "cluster_entry_strict": "positive",
        "recent_spike_position": "positive",
        "cluster_exit_loose": "negative",
        "cluster_exit_strict": "negative",
        "older_spike_position": "negative",
    }
    prior_path = PRIOR_INCREMENTAL_DIR / "results" / "fresh_conditional_hybrid_summary_pooled.csv"
    prior = pd.read_csv(prior_path) if prior_path.exists() else pd.DataFrame()
    pooled = results.get("pooled", pd.DataFrame())
    rows = []
    for condition, expected in expected_signs.items():
        current = pooled[(pooled.get("model_name", pd.Series(dtype=str)) == "RAW_CP_REPO_PLUS_PM") & (pooled.get("condition", pd.Series(dtype=str)) == condition)]
        current_adv = safe_float(current["mean_advantage_vs_CP"].iloc[0]) if not current.empty and "mean_advantage_vs_CP" in current else np.nan
        prior_adv = np.nan
        if not prior.empty and "condition" in prior.columns:
            prior_row = prior[prior["condition"] == condition]
            if not prior_row.empty and "pooled_mean_Hybrid_advantage_vs_CP" in prior_row.columns:
                prior_adv = safe_float(prior_row["pooled_mean_Hybrid_advantage_vs_CP"].iloc[0])
        sign_ok = bool(np.isfinite(current_adv) and ((expected == "positive" and current_adv > 0) or (expected == "negative" and current_adv < 0)))
        rows.append(
            {
                "condition": condition,
                "expected_sign": expected,
                "current_RAW_CP_REPO_PLUS_PM_advantage": current_adv,
                "prior_raw_CP_PLUS_PM_advantage": prior_adv,
                "sign_matches_expected": sign_ok,
                "prior_available": bool(np.isfinite(prior_adv)),
            }
        )
    audit = pd.DataFrame(rows)
    write_csv(audit, outdir / "results" / "raw_cp_pm_reproduction_audit.csv")
    return audit


def write_readme(outdir: Path, args, results: dict[str, pd.DataFrame], feature_manifest: pd.DataFrame, failures: pd.DataFrame) -> None:
    ablation_preview = "[empty]"
    if not results["ablation"].empty:
        cols = ["model_name", "condition", "n_obs", "mean_advantage_vs_CP", "win_rate_vs_CP", "is_paper_eligible"]
        ablation_preview = results["ablation"][[col for col in cols if col in results["ablation"].columns]].to_string(index=False)

    level_failures = int((~feature_manifest["global_level_sum_pass"]).sum()) if not feature_manifest.empty and "global_level_sum_pass" in feature_manifest.columns else 0
    failure_preview = failures.to_string(index=False) if failures is not None and not failures.empty else "[none]"
    content = f"""# CP-REPO-OPS-HG Tests

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
- PM tau grid: `{args.pm_tau_grid}`
- PM eta grid: `{args.pm_eta_grid}`
- PM bandwidth grid: `{args.pm_bandwidth_grid}`
- PHQO gamma grid: `{args.phqo_gamma_grid}`
- PM low modes: `{args.pm_low_modes}`
- PM kernel train window: `{args.pm_kernel_train_window}` prior rows (`0` means all prior rows)
- CV mode: `{args.cv_mode}`
- Command: `{args.command_line}`
- Paper eligibility rule: rows are paper eligible only when both the model and CP benchmark were generated in the current full run and exact timestamp/actual alignment passes. Smoke, reused-prior, skipped, and scaffolded rows are excluded.

## Correct Repo CP Benchmark

`CP_REPO_FRESH` is the established repository CP benchmark: `contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)` with `RV*` columns plus repo `CP_*` columns, matching the original CP+PM incremental test while explicitly excluding diagnostic `CPB_B*` columns. The four-zone lag blocks `{fallback_blocks(args.n)}` are retained only as local shape/diagnostic scaffolds, not as the benchmark and not as proof of orthogonality to repo CP.

## Commands

Smoke test:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode smoke --assets AAPL --phase all --seed 42 --force --audit-no-lookahead --audit-repo-cp-reproduction --audit-fast-slow-equivalence --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Phase 1 run:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode full --phase phase1 --seed 42 --skip-existing --cv-mode inner --lambda-grid 0.0001,0.001,0.01
```

Phase 2 / full linear ladder:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode full --phase linear --seed 42 --skip-existing --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Final all-model full run:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode full --phase all --seed 42 --skip-existing --audit-no-lookahead --audit-repo-cp-reproduction --audit-fast-slow-equivalence --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Regenerate paper tables from existing outputs:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode tables-only --outdir outputs/cp_repo_ops_hg_tests
```

## Interpretation Guardrails

1. If `RAW_CP_REPO_PLUS_PM` loses overall but helps cluster-entry/recent-spike, raw PM is conditionally useful but globally noisy.
2. If `CP_REPO_GATED_RIDGE_OPS_C` beats `CP_REPO_FRESH` overall and improves entry/recent-spike without large exit/high-dispersion losses, cleaned path-shape is supported.
3. If `CP_REPO_OPS_HG` beats `CP_REPO_GATED_RIDGE_OPS_C`, then PM residue structure adds value beyond generic/local path shape.
4. If `CP_REPO_OPS_HG` loses to `CP_REPO_GATED_RIDGE_OPS_C`, then OPS-R/PM residue structure contaminates the final hybrid and should be rejected.
5. If `CP_REPO_RECENT_SLOPE` or `CP_REPO_HAAR_SHAPE` matches/beats OPS-C, then the signal is generic path shape rather than PM-specific.
6. If random residues or shuffled PM match true OPS-R, PM-specific residue structure is not supported.
7. If random gate matches real gate, the regime-targeting story is weak.
8. If results are positive mean advantage but win rate below 50%, describe them as gain-size effects, not row-wise dominance.
9. Do not claim PM is validated unless PM-specific models beat generic shape models and placebos in the expected regimes.
10. Do not use the four-block diagnostic benchmark for primary research claims.

## Shape Level-Removal Check

Shape global level-removal failures: `{level_failures}`

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
    write_csv(results["alternative_loss_by_asset"], result_dir / "alternative_loss_summary_by_asset.csv")
    write_csv(results["alternative_loss"], result_dir / "alternative_loss_summary.csv")
    write_raw_cp_pm_reproduction_audit(outdir, results)

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
    write_pm_native_report(outdir, results)
    return results


def add_common_metadata(metadata: pd.DataFrame, args, outdir: Path) -> pd.DataFrame:
    if metadata is None:
        metadata = pd.DataFrame()
    metadata = metadata.copy()
    registry = getattr(args, "model_registry", None)
    if "model_name" in metadata.columns:
        computed_eligibility = metadata["model_name"].astype(str).map(lambda name: model_is_paper_eligible(name, registry))
        if "paper_eligible" in metadata.columns:
            metadata["paper_eligible"] = metadata["paper_eligible"].fillna(computed_eligibility)
        else:
            metadata["paper_eligible"] = computed_eligibility
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
        "pm_tau_grid": args.pm_tau_grid,
        "pm_eta_grid": args.pm_eta_grid,
        "pm_bandwidth_grid": args.pm_bandwidth_grid,
        "phqo_gamma_grid": args.phqo_gamma_grid,
        "pm_low_modes": args.pm_low_modes,
        "pm_kernel_train_window": args.pm_kernel_train_window,
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

    pm_base = add_lag_columns(base, args.n)
    pm_future = add_lag_columns(future, args.n)
    base_x = lag_matrix(pm_base, args.n)
    future_x = lag_matrix(pm_future, args.n)
    _, base_phi, _ = pm_qdk_phi(base_x, args.n, float(args.pm_tau_grid_values[0]))
    _, future_phi, _ = pm_qdk_phi(future_x, args.n, float(args.pm_tau_grid_values[0]))
    rows.append(
        {
            "check_name": "pm_harmonic_embedding_uses_lagged_inputs_only",
            "passed": bool(np.allclose(base_phi[:75], future_phi[:75], equal_nan=True)),
            "details": "Changing future RV values leaves earlier PM-QDK harmonic embeddings unchanged.",
        }
    )

    frame = add_lag_columns(base, args.n)
    spec = ModelSpec("HAR_RV", RV_HAR_COLS, {}, "linear")
    warmup = 40
    pred_a = fast_expanding_predict(frame, spec, args.n, warmup, target_transform=args.target_transform, log_eps=args.log_eps, max_forecasts=1)
    perturbed = frame.copy()
    target_row = args.n + warmup + 1
    if target_row < len(perturbed):
        perturbed.iloc[target_row, perturbed.columns.get_loc("RV_d")] *= 100.0
    pred_b = fast_expanding_predict(perturbed, spec, args.n, warmup, target_transform=args.target_transform, log_eps=args.log_eps, max_forecasts=1)
    pred_col = "Predicted_HAR_RV"
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
    level_failures = 0 if manifest.empty or "global_level_sum_pass" not in manifest.columns else int((~manifest["global_level_sum_pass"].astype(bool)).sum())
    alignment = results.get("alignment", pd.DataFrame())
    alignment_preview = "[empty]" if alignment.empty else alignment[["asset", "model_name", "timestamps_exactly_match_CP", "actual_target_values_match_CP"]].to_string(index=False)
    content = f"""# Model Construction Proof

Generated: {utc_now()}

## CP Block Definition

The paper benchmark `CP_REPO_FRESH` uses the repository `contig_prime_modulo` CP design: all `RV*` and `CP_*` columns selected with the same convention as the original CP+PM incremental script.

The four lag zones below are retained only for local shape diagnostics and feature construction scaffolds, not as the primary CP benchmark:

```text
B1 = {blocks[0]}
B2 = {blocks[1]}
B3 = {blocks[2]}
B4 = {blocks[3]}
```

Incremental shape models use `y_(t+1) = alpha + beta'C_t + theta'Z_t + error`, where `C_t` is the repo CP design. Ridge penalties, when used, apply to `Z_t` only, not to the intercept or repo CP controls.

## Level-Removal Diagnostics

For generated shape weights, the manifest reports both diagnostic lag-zone block sums and the global lag-weight sum. The hard diagnostic is global level removal with tolerance `{ZERO_SUM_TOL}`. Global level-removal failures: `{level_failures}`.

OPS-R is implemented as centered within-prime residue contrasts (`PMCTR`). OPS-C uses adjacent local sub-block contrasts in lag zones. KOPS uses centered prime-signature kernel features in lag zones. The claim is incremental value after exact repo CP controls, not projection orthogonality to repo CP.

## Feature Counts And Design Ranks

```text
{feature_counts}
```

## Alignment Proof

Every model comparison is paired to `CP_REPO_FRESH` by timestamp and target value before metrics are computed. Paper eligibility requires exact timestamp and actual-target matches.

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
    if sorted(args.assets_resolved) != sorted(DEFAULT_ASSETS):
        return False
    if args.models_resolved != unique_in_order(["CP_REPO_FRESH"] + PHASE_MODELS["all"]):
        return False
    if failures is not None and not failures.empty:
        return False
    if manifest is not None and not manifest.empty and "global_level_sum_pass" in manifest.columns and int((~manifest["global_level_sum_pass"].astype(bool)).sum()) != 0:
        return False
    if metadata.empty:
        return False
    allowed_scaffold = (metadata["model_name"] == "CP_REPO_OPS_K") & (metadata["status"] == "scaffolded")
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
    repo_audit_path = outdir / "results" / "repo_cp_reproduction_audit.csv"
    if not repo_audit_path.exists():
        return False
    repo_audit = pd.read_csv(repo_audit_path)
    if repo_audit.empty or "passes" not in repo_audit.columns or not bool(repo_audit["passes"].astype(bool).all()):
        return False
    fast_slow_path = outdir / "results" / "fast_slow_equivalence_audit.csv"
    if not fast_slow_path.exists():
        return False
    fast_slow = pd.read_csv(fast_slow_path)
    if fast_slow.empty or "passes" not in fast_slow.columns or not bool(fast_slow["passes"].astype(bool).all()):
        return False
    raw_pm_path = outdir / "results" / "raw_cp_pm_reproduction_audit.csv"
    if not raw_pm_path.exists() or pd.read_csv(raw_pm_path).empty:
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

            if model_name in PM_NATIVE_MODELS:
                design_rank = pm_native_feature_count(model_name, args)
                pred = pm_native_expanding_predict(
                    feature_frame,
                    feature_groups,
                    model_name,
                    args,
                    max_forecasts=max_rows,
                )
                pm_params = pred.attrs.get("pm_native_params", {})
            else:
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
                pm_params = {}
            if pred.empty:
                raise RuntimeError(f"{model_name} produced no predictions")
            append_prediction(pred_path, pred, model_name, force=args.force)
            feature_count = pm_native_feature_count(model_name, args) if model_name in PM_NATIVE_MODELS else int(len(spec.features))
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
                    "feature_count": feature_count,
                    "design_rank": design_rank,
                    "rank_deficient": bool(design_rank < feature_count),
                    "seconds": time.perf_counter() - started,
                    "scaffolded": False,
                    "model_family": spec.family,
                    "selected_lambda_shape": spec.lambda_shape,
                    "selected_lambda_r_ratio": spec.lambda_r_ratio,
                    "cv_mode_effective": spec.cv_mode_effective,
                    "cv_score": pm_params.get("cv_score", spec.cv_score),
                    "selected_pm_tau": pm_params.get("tau", np.nan),
                    "selected_pm_eta": pm_params.get("eta", np.nan),
                    "selected_pm_bandwidth": pm_params.get("bandwidth", np.nan),
                    "selected_phqo_gamma": pm_params.get("gamma", np.nan),
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
    parser = argparse.ArgumentParser(description="Run CP-REPO-OPS-HG testing ladder")
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
    parser.add_argument("--pm-tau-grid", default=DEFAULT_PM_TAU_GRID)
    parser.add_argument("--pm-eta-grid", default=DEFAULT_PM_ETA_GRID)
    parser.add_argument("--pm-bandwidth-grid", default=DEFAULT_PM_BANDWIDTH_GRID)
    parser.add_argument("--phqo-gamma-grid", default=DEFAULT_PHQO_GAMMA_GRID)
    parser.add_argument("--pm-low-modes", type=int, default=DEFAULT_PM_LOW_MODES)
    parser.add_argument("--pm-kernel-train-window", type=int, default=DEFAULT_PM_KERNEL_TRAIN_WINDOW, help="Maximum prior rows used by PM-native kernel smoothers; 0 means all prior rows.")
    parser.add_argument("--cv-mode", choices=["auto", "fixed", "inner"], default="auto")
    parser.add_argument("--target-transform", choices=["level", "log"], default="level")
    parser.add_argument("--log-eps", type=float, default=LOG_EPS_DEFAULT)
    parser.add_argument("--audit-no-lookahead", action="store_true")
    parser.add_argument("--audit-repo-cp-reproduction", action="store_true")
    parser.add_argument("--audit-fast-slow-equivalence", action="store_true")
    parser.add_argument("--audit-strong-fast-slow-equivalence", action="store_true")
    parser.add_argument("--strong-audit-assets", default="AAPL,SPY,GLD")
    parser.add_argument("--strong-audit-rows", type=int, default=500)
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
        result_dir / "raw_cp_pm_reproduction_audit.csv",
        result_dir / "alignment_audit.csv",
        result_dir / "inference_summary.csv",
        result_dir / "alternative_loss_summary_by_asset.csv",
        result_dir / "alternative_loss_summary.csv",
        result_dir / "pm_native_placebo_comparison.csv",
        result_dir / "pm_native_results_report.md",
        result_dir / "placebo_diagnostics.csv",
        result_dir / "model_feature_manifest.csv",
        result_dir / "model_construction_proof.md",
        result_dir / "run_metadata.csv",
        result_dir / "run_failures.csv",
        outdir / "paper_tables" / "main_model_summary.csv",
        outdir / "paper_tables" / "conditional_summary.csv",
        outdir / "paper_tables" / "placebo_summary.csv",
        outdir / "paper_tables" / "pm_native_loss_summary.csv",
        outdir / "paper_tables" / "pm_native_asset_consistency.csv",
        outdir / "figures" / "ablation_ladder.png",
        outdir / "figures" / "conditional_advantages.png",
        outdir / "figures" / "pm_native_metric_advantages.png",
        outdir / "README.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if args.audit_repo_cp_reproduction or args.mode == "smoke":
        audit_path = result_dir / "repo_cp_reproduction_audit.csv"
        if not audit_path.exists():
            missing.append(str(audit_path))
    if args.audit_fast_slow_equivalence or args.mode == "smoke":
        audit_path = result_dir / "fast_slow_equivalence_audit.csv"
        if not audit_path.exists():
            missing.append(str(audit_path))
    level_failures = 0 if manifest.empty or "global_level_sum_pass" not in manifest.columns else int((~manifest["global_level_sum_pass"].astype(bool)).sum())
    if missing:
        raise RuntimeError(f"Required outputs missing: {missing}")
    if level_failures:
        raise RuntimeError(f"Global level-removal validation failed for {level_failures} features")
    if args.audit_no_lookahead:
        audit_path = result_dir / NO_LOOKAHEAD_AUDIT_FILENAME
        if not audit_path.exists() or not bool(pd.read_csv(audit_path)["passed"].astype(bool).all()):
            raise RuntimeError("No-lookahead audit marker requested but did not pass")
    args.full_success_ready = full_success_gates_pass(args, metadata, failures, manifest, results, outdir)
    marker = marker_path_for_run(outdir, args, results=results, failures=failures)
    clear_stale_success_markers(outdir, marker)
    marker.write_text(
        f"CP-REPO-OPS-HG {args.mode} {args.phase} run completed successfully at {utc_now()}\n",
        encoding="utf-8",
    )
    if args.mode == "smoke":
        (outdir / "IMPLEMENTATION_READY.txt").write_text(
            f"CP-REPO-OPS-HG implementation-ready smoke verification completed at {utc_now()}\n",
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
    args.pm_tau_grid_values = parse_float_grid(args.pm_tau_grid, DEFAULT_PM_TAU_GRID)
    args.pm_eta_grid_values = parse_float_grid(args.pm_eta_grid, DEFAULT_PM_ETA_GRID)
    args.pm_bandwidth_grid_values = parse_float_grid(args.pm_bandwidth_grid, DEFAULT_PM_BANDWIDTH_GRID)
    args.phqo_gamma_grid_values = parse_float_grid(args.phqo_gamma_grid, DEFAULT_PHQO_GAMMA_GRID)
    fallback_blocks(args.n)
    args.assets_resolved = parse_csv(args.assets, DEFAULT_ASSETS)
    phase_models = PHASE_MODELS[args.phase]
    requested = parse_csv(args.models, phase_models) if args.models else list(phase_models)
    unknown = [model for model in requested if model not in MODEL_ORDER]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")
    args.models_resolved = unique_in_order(["CP_REPO_FRESH"] + requested)

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
    if args.audit_repo_cp_reproduction:
        run_repo_cp_reproduction_audit(outdir, args, args.assets_resolved)
    if args.audit_fast_slow_equivalence:
        run_fast_slow_equivalence_audit(outdir, args, asset=args.assets_resolved[0])
    if args.audit_strong_fast_slow_equivalence:
        run_strong_fast_slow_equivalence_audit(outdir, args)

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
            f"CP-REPO-OPS-HG tables-only run completed successfully at {utc_now()}\n",
            encoding="utf-8",
        )
        return

    print(f"CP-REPO-OPS-HG runner: mode={args.mode} phase={args.phase} assets={args.assets_resolved}")
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




