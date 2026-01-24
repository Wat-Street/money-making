import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
import time
from yfinance.exceptions import YFPricesMissingError

def handleDaily(df):
    result = pd.DataFrame(index=df.index)
    result['Close'] = df['Close']
    result['Volume'] = df['Volume']
    result['Log_Return'] = np.log(result['Close'] / result['Close'].shift(1))
    result['Squared_Return'] = result['Log_Return'] ** 2
    return result.dropna()

def daterange(start_date, end_date, delta):
    current = start_date
    while current < end_date:
        yield current, min(current + delta, end_date)
        current += delta

def fetch_data_in_chunks(ticker, start_date, end_date, chunk_size_days=30, delay_time=0.88):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    dfs = []
    for chunk_start, chunk_end in daterange(start, end, timedelta(days=chunk_size_days)):
        print(f"Fetching data from {chunk_start.date()} to {chunk_end.date()}")
        df_chunk = yf.download(ticker, start=chunk_start, end=chunk_end, progress=False)
        dfs.append(df_chunk)
        time.sleep(delay_time)

    full_df = pd.concat(dfs).drop_duplicates().sort_index()
    return full_df

def load_local_5m_csv(ticker: str, base_dir: str = "Datasets/clean", target_tz: str = "America/New_York"):
    """
    Load 5-minute OHLCV CSV with Datetime index.
    Expected columns: Date/Datetime, Open, High, Low, Close, Volume.
    """
    import os
    import pandas as pd
    candidates = [
        os.path.join(base_dir, f"{ticker}_5m.csv"),
        os.path.join(base_dir, f"{ticker}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            for c in ["Datetime", "Date", "timestamp", "date"]:
                if c in df.columns:
                    dt = pd.to_datetime(df[c], utc=True, errors="coerce")
                    if hasattr(dt, 'dt'):
                        dt = dt.dt.tz_convert(target_tz).dt.tz_localize(None)
                    else:
                        dt = dt.tz_convert(target_tz).tz_localize(None)
                    df[c] = dt
                    df = df.set_index(c)
                    break
            df = df.sort_index()
            rename_map = {c: c.title() for c in ["open","high","low","close","volume"] if c in df.columns}
            df = df.rename(columns=rename_map)
            needed = ["Close","Volume"]
            if not all(col in df.columns for col in needed):
                raise ValueError(f"Missing columns in {path}. Need at least Close and Volume.")
            return df
    raise FileNotFoundError(f"No local CSV found for {ticker} in {base_dir}")

def fetch_intraday_data_in_chunks(ticker, start_date, end_date, chunk_hours=1, delay_time=0.88, interval='5m'):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dfs = []

    for day_start, day_end in daterange(start, end, timedelta(days=1)):
        for chunk_start, chunk_end in daterange(day_start, day_end + timedelta(days=1), timedelta(hours=chunk_hours)):
            print(f"Fetching {ticker} data from {chunk_start} to {chunk_end} (interval={interval})")
            time.sleep(delay_time)
            try:
                df_chunk = yf.download(
                    ticker,
                    start=chunk_start,
                    end=chunk_end,
                    interval=interval,
                    progress=False
                )
                if df_chunk.empty:
                    print(f"No data for {chunk_start} to {chunk_end}, skipping...")
                    continue
                dfs.append(df_chunk)
            except Exception as e:
                print(f"Error during fetch: {e}. Skipping {chunk_start} to {chunk_end}.")
                continue

    if dfs:
        full_df = pd.concat(dfs).drop_duplicates().sort_index()
        return full_df
    else:
        print("No data fetched at all.")
        return pd.DataFrame()

def fetch_data(ticker, start_date, end_date):
    df = fetch_data_in_chunks(ticker, start_date=start_date, end_date=end_date, chunk_size_days=30, delay_time=0.88)
    return handleDaily(df)

def handleIntraday(df):
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert('America/New_York').tz_localize(None)
    result = pd.DataFrame(index=idx)
    result['Close'] = df['Close'].values
    result['Volume'] = df['Volume'].values
    result['Log_Return'] = np.log(result['Close'] / result['Close'].shift(1))
    result.loc[result.index.time == pd.Timestamp('09:30').time(), 'Log_Return'] = np.nan
    result['Squared_Return'] = result['Log_Return'] ** 2
    return result.dropna()

def fetch_intraday_data(ticker: str = "AAPL", start_date: str = None, end_date: str = None, use_local: bool = False, local_dir: str = "Datasets/clean"):
    if use_local:
        df = load_local_5m_csv(ticker, base_dir=local_dir)
        return handleIntraday(df)
    if end_date is None:
        end_date = datetime.today()
    else:
        end_date = pd.to_datetime(end_date)
    if start_date is None:
        start_date = end_date - timedelta(days=60)
    else:
        start_date = pd.to_datetime(start_date)
    df = fetch_intraday_data_in_chunks(ticker, start_date, end_date, chunk_hours=1, delay_time=0.88, interval='5m')
    return handleIntraday(df)

# Prediction Model
def fit_and_predict_extended(data, features, n, warmup=30, model_name: str = "Model"):
    predictions = []
    for i in range(n + warmup, len(data) - 1):
        try:
            train_data = data.iloc[:i+1].copy()
            X = train_data[features]
            y = train_data['RV_d'].shift(-1)
            X, y = X.iloc[:-1], y.iloc[:-1]
            X = add_constant(X)
            model = OLS(y, X).fit()
            test_row = data.iloc[[i]][features].copy()
            test_row = add_constant(test_row, has_constant='add')
            test_row = test_row.reindex(columns=X.columns, fill_value=0)
            pred = float(model.predict(test_row).squeeze())
            y_true_next = float(data.iloc[i + 1]['RV_d'])
            err = y_true_next - pred
            abs_err = abs(err)
            denom = max(1e-12, abs(y_true_next) + abs(pred))
            smape_pct = 200.0 * abs_err / denom
            predictions.append({
                'Date': data.index[i + 1],
                'Actual': y_true_next,
                f'Predicted_{model_name}': pred,
                f'Err_{model_name}': err,
                f'AbsErr_{model_name}': abs_err,
                f'SMAPE_{model_name}_pct': smape_pct
            })
        except Exception as e:
            print(f"Warning at index {i}: {str(e)}")
            continue
    if predictions:
        results = pd.DataFrame(predictions)
        results.set_index('Date', inplace=True)
        pred_cols = [c for c in results.columns if c.startswith('Predicted_')]
        if pred_cols:
            results['Predicted'] = results[pred_cols[0]]
        return results
    else:
        return pd.DataFrame()

# Calculate realized volatility (DAILY)
def calculate_realized_volatility(df, n):
    result = pd.DataFrame(index=df.index)
    result['SR'] = df['Squared_Return']                       # variance contribution
    result['RV_d'] = np.sqrt(df['Squared_Return'])            # realized volatility (per day)
    result['RV_w'] = np.sqrt(df['Squared_Return'].rolling(window=5).sum())
    result['RV_m'] = np.sqrt(df['Squared_Return'].rolling(window=n).sum())
    result['Volume'] = df['Volume']
    return result.dropna()

# Calculate realized volatility (INTRADAY 5m)
def calculate_intraday_realized_volatility(df):
    result = pd.DataFrame(index=df.index)
    result['SR'] = df['Squared_Return']                       # variance contribution per 5m bar
    result['RV_d'] = np.sqrt(df['Squared_Return'])            # per-bar realized vol (5m)

    # Weekly = 78 periods (1 trading day = 78 5-min periods)
    result['RV_w'] = np.sqrt(df['Squared_Return'].rolling(window=78).sum())
    # Monthly = 78 * 21 periods (21 trading days)
    result['RV_m'] = np.sqrt(df['Squared_Return'].rolling(window=78 * 21).sum())

    result['Volume'] = df['Volume']
    return result.dropna()
