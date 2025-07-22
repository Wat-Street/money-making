import pandas as pd

def compute_lagged_correlation(series_a: pd.Series, series_b: pd.Series, max_lag: int) -> pd.DataFrame:
    """
    Compute lagged correlation between two series.
    Args:
        series_a (pd.Series): First time series (e.g., df['Close_<ticker_a>']).
        series_b (pd.Series): Second time series (e.g., df['Close_<ticker_b>']).
        max_lag (int): Maximum lag (both positive and negative) to compute.
    Returns:
        pd.DataFrame: DataFrame with columns 'Lag' and 'Correlation'.
    """
    # Drop the first row if it contains the ticker name (non-numeric)
    def clean_series(s: pd.Series) -> pd.Series:
        s_clean = s.copy()
        if not pd.api.types.is_numeric_dtype(s_clean.iloc[0]):
            s_clean = s_clean.iloc[1:]
        s_numeric = pd.to_numeric(s_clean, errors='coerce')
        if not isinstance(s_numeric, pd.Series):
            s_numeric = pd.Series(s_numeric, index=s_clean.index)
        return s_numeric

    series_a = clean_series(series_a)
    series_b = clean_series(series_b)

    lags = range(-max_lag, max_lag + 1)
    correlations = []
    for lag in lags:
        if lag > 0:
            shifted_a = series_a.shift(lag)
            aligned_a, aligned_b = shifted_a.align(series_b, join='inner')
        elif lag < 0:
            shifted_b = series_b.shift(-lag)
            aligned_a, aligned_b = series_a.align(shifted_b, join='inner')
        else:
            aligned_a, aligned_b = series_a.align(series_b, join='inner')

        corr = aligned_a.corr(aligned_b) # uses Pearson formula by default for calculating the coefficient
        correlations.append(corr)
    return pd.DataFrame({"Lag": list(lags), "Correlation": correlations})
