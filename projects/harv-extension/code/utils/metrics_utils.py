
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Iterable
from math import log
from scipy.stats import norm, chi2

EPS = 1e-12

def smape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(eps, np.abs(y_true) + np.abs(y_pred))
    smape_t = 200.0 * np.abs(y_true - y_pred) / denom
    return float(np.mean(smape_t))

def per_timestamp_smape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(eps, np.abs(y_true) + np.abs(y_pred))
    return 200.0 * np.abs(y_true - y_pred) / denom

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = y_true - y_pred

    smape_pct = smape_percent(y_true, y_pred)
    rmse = float(np.sqrt(np.mean(e**2)))
    abs_e = np.abs(e)
    mae = float(np.mean(abs_e))
    medae = float(np.median(abs_e)) if abs_e.size else float('nan')

    # directional accuracy: sign of next-step change vs forecasted change
    if len(y_true) >= 2:
        dy_true = np.diff(y_true)
        dy_pred = y_pred[1:] - y_true[:-1]
        dir_hits = (np.sign(dy_true) == np.sign(dy_pred)).astype(float)
        dir_acc_pct = float(100.0 * np.mean(dir_hits))
    else:
        dir_acc_pct = float('nan')

    return {
        "smape_pct": smape_pct,
        "rmse": rmse,
        "mae": mae,
        "medae": medae,
        "diracc_pct": dir_acc_pct,
        "n_test": float(len(y_true))
    }

def newey_west_var(d: np.ndarray, lags: int) -> float:
    """NW HAC variance estimator of the sample mean of d."""
    d = np.asarray(d, dtype=float)
    T = d.shape[0]
    if T <= 1:
        return np.nan
    d_centered = d - np.mean(d)
    # gamma_0
    gamma0 = np.dot(d_centered, d_centered) / T
    var_hat = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = np.dot(d_centered[k:], d_centered[:-k]) / T
        var_hat += 2.0 * w * cov
    # variance of the mean
    return var_hat / T

def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = None) -> Tuple[float, float]:
    """
    Diebold–Mariano test for equal predictive accuracy.
    Returns (statistic, pvalue) using asymptotic normal approximation.
    Default loss: pass in per-timestamp losses, e.g., |e| for MAE or (e**2) for MSE.
    """
    la = np.asarray(loss_a, dtype=float)
    lb = np.asarray(loss_b, dtype=float)
    n = min(len(la), len(lb))
    if n < 5:
        return np.nan, np.nan
    la = la[:n]
    lb = lb[:n]
    d = la - lb
    if lags is None:
        lags = max(1, int(round(n ** (1.0 / 3.0))))
    var_mean = newey_west_var(d, lags)
    if not np.isfinite(var_mean) or var_mean <= 0:
        return np.nan, np.nan
    dm_stat = np.mean(d) / np.sqrt(var_mean)
    pval = 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(pval)

def fisher_p(pvals: Iterable[float]) -> float:
    """Fisher method for combining independent p-values."""
    pvals = [p for p in pvals if (p is not None and np.isfinite(p) and p > 0 and p <= 1)]
    if len(pvals) == 0:
        return np.nan
    stat = -2.0 * np.sum(np.log(pvals))
    df = 2 * len(pvals)
    return float(1.0 - chi2.cdf(stat, df))

def rolling_smape_series(y_true: pd.Series, y_pred: pd.Series, window: int) -> pd.Series:
    arr = per_timestamp_smape_percent(y_true.values, y_pred.values)
    s = pd.Series(arr, index=y_true.index)
    return s.rolling(window=window, min_periods=1).mean()
