import math
from typing import Optional
import pandas as pd

class FeatureTransform:
    """
    Feature transformation and engineering functions. Assume that all inputs have been cleaned, validated,
    and aligned. For example, if a function requires two Series, they must be date and shape aligned.

    All dataframes must have format as follows ("long format"):
    
    date index | ticker | price | market_cap | earnings_per_share | ...
    ----------------------------------------------------------------------
    2023-01-01 |  AAPL  |  230  |    1.54    |          20
    2023-01-01 |  META  |  150  |    1.54    |
    2022-01-01 |  AAPL  |  230  |    1.54    |          20
    2022-01-01 |  META  |  150  |    1.54    |

    Sample Data:
    ```
    import pandas as pd
    df = pd.read_parquet("")
    ```
    """

    ############################################
    # Data Series Transformations
    ############################################
    
    @staticmethod
    def rolling_beta(
        security_series: pd.Series, benchmark_series: pd.Series, window: int, min_window_pct: float=0.8
    ) -> pd.Series:
        """Compute beta with a rolling lookback window"""

        assert window > 0
        assert 0 <= min_window_pct <= 1
        min_periods = FeatureTransform._get_min_periods_length(window, min_window_pct)

        return_series = security_series.pct_change(1).rolling(window, min_periods=min_periods)
        benchmark_return_series = benchmark_series.pct_change(1).rolling(window, min_periods=min_periods)
        
        # Beta = Cov(R_p, R_m) / Var(R_m) where R_p is portfolio return, R_m is market (benchmark) return
        betas = return_series.cov(benchmark_return_series) / benchmark_return_series.var()
        return betas

    @staticmethod
    def percent_from_trailing_max(
        series: pd.Series, window: int, min_window_pct: float=0.8
    ) -> pd.Series:
        assert window > 0
        assert 0 <= min_window_pct <= 1
        min_periods = FeatureTransform._get_min_periods_length(window, min_window_pct)

        return series / series.rolling(window, min_periods=min_periods).max() - 1

    @staticmethod
    def percent_from_trailing_min(
        series: pd.Series, window: int, min_window_pct: float=0.8
    ) -> pd.Series:
        assert window > 0
        assert 0 <= min_window_pct <= 1
        min_periods = FeatureTransform._get_min_periods_length(window, min_window_pct)

        return series / series.rolling(window, min_periods=min_periods).min() - 1

    @staticmethod
    def sma(
        series: pd.Series, window: int, min_window_pct: float=0.8
    ) -> pd.Series:
        """Calculate a simple moving average"""
        assert window > 0
        assert 0 <= min_window_pct <= 1
        min_periods = FeatureTransform._get_min_periods_length(window, min_window_pct)

        return series.rolling(window=window, min_periods=min_periods).mean()
    
    @staticmethod
    def ema(
        series: pd.Series, window: int, min_window_pct: float=0.8, adjust: bool = False, halflife: Optional[float] = None, span: Optional[float] = None, com: Optional[float] = None
        # TODO: add other params for weights
    ) -> pd.Series:
        """Calculate an exponentially-weighted moving average"""
        assert window > 0
        assert 0 <= min_window_pct <= 1
        min_periods = FeatureTransform._get_min_periods_length(window, min_window_pct)

        # Use provided halflife, span, or com if given, otherwise default to window for span
        if halflife is not None:
            return series.ewm(halflife=halflife, min_periods=min_periods, adjust=adjust).mean()
        elif span is not None:
            return series.ewm(span=span, min_periods=min_periods, adjust=adjust).mean()
        elif com is not None:
            return series.ewm(com=com, min_periods=min_periods, adjust=adjust).mean()
        else:
            return series.ewm(span=window, min_periods=min_periods, adjust=adjust).mean()

        
    ############################################
    # DataFrame Normalization Methods
    ############################################

    @staticmethod
    def rolling_zscore(
        data: pd.DataFrame, window: int, min_window_pct=0.8, group_column='ticker', inplace=True
    ) -> pd.DataFrame:
        """Compute a rolling window Z score grouped by some column.

        Args:
            data (pd.DataFrame): _description_
            window (int): _description_
            min_window_pct (float, optional): _description_. Defaults to 0.8.
            group_column (str, optional): _description_. Defaults to 'ticker'.
            inplace (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
        assert window > 0
        assert 0 <= min_window_pct <= 1
        min_periods = FeatureTransform._get_min_periods_length(window, min_window_pct)
        if not inplace:
            data = data.copy()

        # Function to calculate Z-score
        def rolling_zscore(x):
            r = x.rolling(window=window, min_periods=min_periods)
            m = r.mean()
            s = r.std(ddof=0)
            z = (x - m) / s
            return z

        # Apply Z-score function to each group
        numeric_cols = data.select_dtypes(include='number').columns
        for col in numeric_cols:
            data[f'{col}_r_zscore'] = data.groupby(group_column)[col].transform(rolling_zscore)

        return data

    @staticmethod
    def cross_sectional_zscore(
        data: pd.DataFrame, min_window_pct=0.8, inplace=True
    ) -> pd.DataFrame:
        """Compute the cross-sectional Z score grouped by the dataframe index.

        Args:
            data (pd.DataFrame): _description_
            min_window_pct (float, optional): _description_. Defaults to 0.8.
            inplace (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
        assert 0 <= min_window_pct <= 1
        if not inplace:
            data = data.copy()

        # group by date, and Z score normalize the features on each date.
        def zscore(df):
            return (df - df.mean()) / df.std(ddof=0)

        numeric_cols = data.select_dtypes(include='number').columns
        for col in numeric_cols:
            # Assume dataframe is the index 
            data[f'{col}_zscore'] = data.groupby(level=0)[col].transform(zscore)
            
        return data
    
    ############################################
    # Private Methods
    ############################################

    @staticmethod
    def _get_min_periods_length(window: int, min_window_pct: float) -> int:
        """Gets the minimum period required for a rolling window."""
        return int(math.ceil(window * min_window_pct))

