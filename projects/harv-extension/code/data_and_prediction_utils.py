import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant

# Fetch data
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    result = pd.DataFrame(index=df.index)
    result['Close'] = df['Close']
    result['Log_Return'] = np.log(result['Close'] / result['Close'].shift(1))
    result['Squared_Return'] = result['Log_Return'] ** 2
    return result.dropna()

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
