import os
import pandas as pd
import yfinance as yf

def get_data_path(ticker, start_date, end_date, folder="data/raw"):
    """
    Generate the file path for storing or loading a stock's raw data CSV.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date of the data (YYYY-MM-DD).
        end_date (str): End date of the data (YYYY-MM-DD).
        folder (str): Folder where the file should be stored or searched.

    Returns:
        str: Full file path for the cached CSV file.
    """
    filename = f"{ticker}_{start_date}_to_{end_date}_raw.csv"
    return os.path.join(folder, filename)


def get_stock_data(ticker, start_date, end_date, cache=True):
    """
    Fetch historical stock price data using yfinance, with optional caching.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date of the data (YYYY-MM-DD).
        end_date (str): End date of the data (YYYY-MM-DD).
        cache (bool): If True, load from or save to local CSV cache.

    Returns:
        pd.DataFrame: DataFrame with columns ['Date', 'Close', 'Ticker'].
    """
    path = get_data_path(ticker, start_date, end_date)
    if cache and os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])

    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df = df[["Date", "Close"]]
    df["Ticker"] = ticker
    if cache:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(path, index=False)
    
    return df


def load_pair_data(ticker_a, ticker_b, start_date, end_date, cache=True):
    """
    Load and align historical price data for a pair of stocks.

    Args:
        ticker_a (str): First stock ticker (e.g., 'AAPL').
        ticker_b (str): Second stock ticker (e.g., 'MSFT').
        start_date (str): Start date for both stocks.
        end_date (str): End date for both stocks.
        cache (bool): Whether to cache individual stock data locally.

    Returns:
        pd.DataFrame: Merged DataFrame with aligned 'Date', 'Close_<ticker>' columns.
    """
    print(f"Loading data for {ticker_a}")
    df_a = get_stock_data(ticker_a, start_date, end_date, cache)

    print(f"Loading data for {ticker_b}")
    df_b = get_stock_data(ticker_b, start_date, end_date, cache)

    merged = pd.merge(df_a, df_b, on="Date", suffixes=(f"_{ticker_a}", f"_{ticker_b}"))
    return merged


def load_multiple_pairs(pairs, start_date, end_date, cache=True):
    """
    Load and align historical data for multiple stock pairs.

    Args:
        pairs (list of tuples): List of (ticker_a, ticker_b) pairs.
        start_date (str): Start date for all data fetches.
        end_date (str): End date for all data fetches.
        cache (bool): Whether to cache and reuse raw data.

    Returns:
        dict: Dictionary mapping (ticker_a, ticker_b) → merged DataFrame.
    """
    pair_data = {}
    for ticker_a, ticker_b in pairs:
        df = load_pair_data(ticker_a, ticker_b, start_date, end_date, cache)
        pair_data[(ticker_a, ticker_b)] = df
    return pair_data