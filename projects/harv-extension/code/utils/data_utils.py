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

def fetch_intraday_data_in_chunks(ticker, start_date, end_date, chunk_hours=1, delay_time=0.88, interval='5m'):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dfs = []

    for day_start, day_end in daterange(start, end, timedelta(days=1)):
        current_day = day_start.date()
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
    result = pd.DataFrame(index=df.index)
    result['Close'] = df['Close']
    result['Volume'] = df['Volume']
    result['Date'] = df.index.date
    result['Log_Return'] = np.log(result['Close'] / result['Close'].shift(1))
    result.loc[result.index.time == pd.Timestamp('09:30').time(), 'Log_Return'] = np.nan
    result['Squared_Return'] = result['Log_Return'] ** 2
    result = result.drop('Date', axis=1)
    return result.dropna()

def fetch_intraday_data():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)
    ticker = "AAPL"
    df = fetch_intraday_data_in_chunks(ticker, start_date, end_date, chunk_hours=1, delay_time=0.88, interval='5m')
    print(df)
    return handleIntraday(df)

# Prediction Model
def fit_and_predict_extended(data, features, n, warmup=30):
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
            pred = model.predict(test_row).squeeze()
            predictions.append({
                'Date': data.index[i + 1],
                'Actual': data.iloc[i + 1]['RV_d'],
                'Predicted': pred
            })
        except Exception as e:
            print(f"Warning at index {i}: {str(e)}")
            continue
    if predictions:
        results = pd.DataFrame(predictions)
        results.set_index('Date', inplace=True)
        return results
    else:
        return pd.DataFrame()

# Calculate realized volatility
def calculate_realized_volatility(df, n):
    result = pd.DataFrame(index=df.index)
    result['RV_d'] = df['Squared_Return']
    result['RV_w'] = df['Squared_Return'].rolling(window=5).mean()
    result['RV_m'] = df['Squared_Return'].rolling(window=n).mean()
    result['Volume'] = df['Volume']
    return result.dropna()

# Calculate realized volatility
def calculate_intraday_realized_volatility(df):
    result = pd.DataFrame(index=df.index)
    result['RV_d'] = df['Squared_Return']
    result['RV_w'] = df['Squared_Return'].rolling(window=24 * 60 // 5).mean()
    result['RV_m'] = df['Squared_Return'].rolling(window=24 * 60).mean()
    result['Volume'] = df['Volume']
    return result.dropna()
