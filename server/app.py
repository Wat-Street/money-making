from flask import Flask, request, jsonify
import requests
import math

from meanrev.Mean_Reversion_Test import test
from meanrev.Mean_Reversion_Train import train
import pandas as pd
import requests
import datetime
from dotenv import load_dotenv
import os

from pyngrok import ngrok

import yfinance as yf

# Load environment variables from .env.local file
load_dotenv('./.env.local')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

POLYGON_KEY_ID = os.getenv('POLYGON_KEY_ID')

app = Flask(__name__)

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    if request.method == 'GET':
        return jsonify({'message': 'This is a GET request.'})
    elif request.method == 'POST':
        try:
            data = request.json
            return jsonify({'message': 'This is a POST request.', 'data': data})
        except Exception as e:
            return jsonify({'error': 'Invalid JSON data.'}), 400


@app.route('/api/meanreversion', methods=['POST'])
def meanreversion_endpoint():
    try:
        print(request.json)
        
        request_data = request.json
        ticker = request_data['ticker']
        train_start = request_data['trainstart']
        train_end = request_data['trainend']
        test_start = request_data['teststart']
        test_end = request_data['testend']
        max_holding = int(request_data['maxholding'])

        # Download historical data for training
        historic_data_train = yf.download(ticker, start=train_start, end=train_end, interval="1d")
        prices_train = historic_data_train['Adj Close']
        
        # Download historical data for testing
        historic_data_test = yf.download(ticker, start=test_start, end=test_end, interval="1d")
        prices_test = historic_data_test['Adj Close']
        
        # Train the model and get suggestions
        model = train(prices_train, max_holding)
        suggestions = test(model, prices_test)

        model_return = trade_index_with_confidence_as_duration(max_holding, 1000, ticker, prices_test, suggestions)
        normal_return = (prices_test[-1] - prices_test[0]) / prices_test[0]
        
        # Create the response in the desired format
        results = {
            "holding": {
                "difference": normal_return,
                "positive": normal_return > 0,
            },
            "model": {
                "difference": model_return,
                "positive": model_return > 0,
            }
        };
        return jsonify({"results": results, "points": suggestions})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

class Account():
    def __init__(self):
        self.balance = 0
        self.holdings = []
        self.min_balance = 0


    def buy(self, stock, quantity):
        if stock not in self.holdings:
            self.holdings.append(stock)
        self.balance -= stock.price * quantity
        self.min_balance = min(self.balance, self.min_balance)
        stock.add_holding(quantity)

    def sell(self, stock, quantity):
        self.balance += stock.price * quantity
        stock.remove_holding(quantity)

    def net_worth(self):
        return self.balance + sum([(stock.price * stock.holding) for stock in self.holdings])

class Stock():
    def __init__(self, name, price):
        self.name = name
        self.price = price
        self.holding = 0

    def update_price(self, new_price):
        self.price = new_price

    def add_holding(self, quantity):
        self.holding += quantity

    def remove_holding(self, quantity):
        self.holding -= quantity


inverse_time_effect3 = lambda L, x: L * (x ** 2)
def trade_index_with_confidence_as_duration(MAX_HOLDING, MAX_TRANSACTION, TICKER, testing_prices, predictions):
    account = Account()
    stock = Stock(TICKER, testing_prices.iloc[0])

    sorted_preds = sorted(predictions, key=lambda x: datetime.strptime(x['datetime'], '%Y-%m-%d %H:%M'))

    for action in sorted_preds:
        stock.update_price(testing_prices.iloc[action])
        if action['suggestion'] == 'Buy':
            account.buy(stock, round(abs((MAX_TRANSACTION / stock.price) * account['confidence']), 3))
        else:
            account.sell(stock, round(abs((MAX_TRANSACTION / stock.price) * account['confidence']), 3))
    return (account.net_worth - abs(account.min_balance)) / abs(account.min_balance)
    

@app.route('/api/list_tickers', methods=['GET'])
def list_tickers():
    try:
        # url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&sort=ticker&order=asc&limit=10&apiKey={POLYGON_API_KEY}"
        # response = requests.get(url)
        # data = response.json()

        # tickers = [ticker['ticker'] for ticker in data['results']]
        tickers = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]
        return jsonify(tickers)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/stock_data', methods=['GET'])
def stock_data():
    try:
        ticker = request.args.get('ticker')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not start_date:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=101)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        data = response.json()
        if 'results' not in data:
            return jsonify({'error': 'No data found for the given ticker and date range.'}), 404

        stock_data = []
        for item in data['results']:
            stock_data.append({
                'datetime': datetime.datetime.fromtimestamp(item['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'open': item['o'],
                'high': item['h'],
                'low': item['l'],
                'close': item['c'],
                'volume': item['v']
            })

        return jsonify(stock_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

if __name__ == '__main__':

    # uncomment the following 2 lines to run without ngrok (this is prod level code!)
    public_url = ngrok.connect(name='flask').public_url
    print(" * ngrok URL: " + public_url + " *")
    app.run()
