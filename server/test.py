from meanrev.Mean_Reversion_Test import test
from meanrev.Mean_Reversion_Train import train
import yfinance as yf

historic_data_train = yf.download("AAPL", start="2015-01-01", end="2022-01-01",  interval = "1d")
prices_train = historic_data_train['Adj Close']
historic_data_test = yf.download("AAPL", start="2022-01-01", end="2024-05-06",  interval = "1d")
prices_test = historic_data_test['Adj Close']
model = train(prices_train, 100)
suggestions = test(model, prices_test)
print(suggestions)