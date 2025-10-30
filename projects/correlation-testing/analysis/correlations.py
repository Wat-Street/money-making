# correlations.py
from __future__ import annotations
import pandas as pd

def _align_with_lag(a: pd.Series, b: pd.Series, lag: int) -> tuple[pd.Series, pd.Series]:
    """
    Positive `lag` means B is shifted forward (B leads A).
    Negative `lag` means B is shifted backward (B lags A).
    """
    if lag > 0:
        b_shifted = b.shift(lag)
        a_aligned, b_aligned = a.align(b_shifted, join="inner")
    elif lag < 0:
        a_shifted = a.shift(-lag)
        a_aligned, b_aligned = a_shifted.align(b, join="inner")
    else:
        a_aligned, b_aligned = a.align(b, join="inner")

    mask = a_aligned.notna() & b_aligned.notna()
    return a_aligned[mask], b_aligned[mask]

def compute_lagged_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    max_lag: int = 10,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute correlation for lags in [-max_lag, ..., +max_lag].
    method: "pearson" or "spearman".
    Returns DataFrame with columns: ["Lag", "Correlation"].
    """
    method = method.lower()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    rows = []
    a = pd.to_numeric(series_a, errors="coerce")
    b = pd.to_numeric(series_b, errors="coerce")

    for lag in range(-max_lag, max_lag + 1):
        a_lag, b_lag = _align_with_lag(a, b, lag)
        if len(a_lag) < 2:
            corr = float("nan")
        else:
            if method == "pearson":
                corr = a_lag.corr(b_lag, method="pearson")
            else:
                corr = a_lag.rank(method="average").corr(
                    b_lag.rank(method="average"), method="pearson"
                )
        rows.append((lag, corr))

    return pd.DataFrame(rows, columns=["Lag", "Correlation"])

def summarize_best_lag(corr_df: pd.DataFrame) -> dict:
    """Return the lag with max absolute correlation and its value."""
    s = corr_df["Correlation"].dropna()
    if s.empty:
        return {"best_lag": None, "best_corr": None}
    idx = s.abs().idxmax()
    row = corr_df.loc[idx]
    return {"best_lag": int(row["Lag"]), "best_corr": float(row["Correlation"])}
