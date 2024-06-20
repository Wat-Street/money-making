import yfinance as yf
import requests
import json
from datetime import datetime

def retrieve_data(ticker, start,end):
    data = yf.download(ticker, start="2022-05-01", end="2022-05-03",  interval = "1m")
    prices = data['Adj Close']

