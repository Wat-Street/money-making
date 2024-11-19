from flask import Flask, request, jsonify
import requests
import math

from meanrev.Mean_Reversion_Test import test
from meanrev.Mean_Reversion_Train import train
import pandas as pd
import requests
from datetime import datetime, timedelta
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
        # Validate request data exists
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        request_data = request.json
        
        # Validate required fields
        required_fields = ['ticker', 'trainstart', 'trainend', 'teststart', 'testend', 'maxholding']
        missing_fields = [field for field in required_fields if field not in request_data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Extract and validate data
        ticker = request_data['ticker']
        train_start = request_data['trainstart']
        train_end = request_data['trainend']
        test_start = request_data['teststart']
        test_end = request_data['testend']
        
        try:
            max_holding = int(request_data['maxholding'])
            if max_holding <= 0:
                return jsonify({'error': 'maxholding must be a positive integer'}), 400
        except ValueError:
            return jsonify({'error': 'maxholding must be a valid integer'}), 400

        # Download historical data with error handling
        try:
            historic_data_train = yf.download(ticker, start=train_start, end=train_end, interval="1d")
            if historic_data_train.empty:
                return jsonify({'error': f'No training data found for {ticker} in specified date range'}), 404
            prices_train = historic_data_train['Adj Close']
            
            historic_data_test = yf.download(ticker, start=test_start, end=test_end, interval="1d")
            if historic_data_test.empty:
                return jsonify({'error': f'No test data found for {ticker} in specified date range'}), 404
            prices_test = historic_data_test['Adj Close']
        except Exception as e:
            return jsonify({'error': f'Failed to download stock data: {str(e)}'}), 500

        # Train model and get suggestions with error handling
        try:
            model = train(prices_train, max_holding)
            suggestions = test(model, prices_test)
        except Exception as e:
            return jsonify({'error': f'Model training/testing failed: {str(e)}'}), 500

        try:
            model_return = trade_index_with_confidence_as_duration(max_holding, 1000, ticker, prices_test, suggestions)
            normal_return = (prices_test[-1] - prices_test[0]) / prices_test[0]
        except Exception as e:
            return jsonify({'error': f'Return calculation failed: {str(e)}'}), 500

        # Create the response
        results = {
            "holding": {
                "difference": float(normal_return),  # Convert numpy types to native Python
                "positive": bool(normal_return > 0),
            },
            "model": {
                "difference": float(model_return),
                "positive": bool(model_return > 0),
            }
        }
        }
        return jsonify({"results": results, "points": suggestions})

    except Exception as e:
        # Log the full error for debugging
        print(f"Unexpected error in meanreversion_endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500
    

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

    try:
        sorted_preds = sorted(predictions, key=lambda x: datetime.strptime(x['datetime'], '%Y-%m-%d %H:%M:%S'))

        for action in sorted_preds:
            try:
                stock.update_price(testing_prices.loc[action['datetime']])  # Ensure 'action' is a valid index
            except Exception as e:
                print(f"Error updating stock price for action {action}: {str(e)}")
                continue  # Skip this action if there's an error

            try:
                if action['suggestion'] == 'Buy':
                    account.buy(stock, round(abs((MAX_TRANSACTION / stock.price) * action["confidence"]), 3))
                else:
                    account.sell(stock, round(abs((MAX_TRANSACTION / stock.price) * action["confidence"]), 3))
            except Exception as e:
                print(f"Error during buy/sell operation for action {action}: {str(e)}")
                continue

    except Exception as e:
        print(f"Unexpected error in trade_index_with_confidence_as_duration: {str(e)}")
        return None

    return (account.net_worth() - abs(account.min_balance)) / abs(account.min_balance)
    

@app.route('/api/list_tickers', methods=['GET'])
def list_tickers():
    try:
        tickers = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]
        return jsonify(tickers)
    except Exception as e:
        print(f"Unexpected error in list_tickers: {str(e)}")
        return jsonify({'error': 'Failed to retrieve ticker list'}), 500

@app.route('/api/stock_data', methods=['GET'])
def stock_data():
    try:
        # Validate required parameters
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400

        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=101)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # Validate API key
        if not POLYGON_API_KEY:
            return jsonify({'error': 'API key not configured'}), 500

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'Failed to fetch data from Polygon API: {str(e)}'}), 503

        if 'results' not in data:
            return jsonify({'error': 'No data found for the given ticker and date range'}), 404

        stock_data = []
        for item in data['results']:
            stock_data.append({
                'datetime': datetime.fromtimestamp(item['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'open': item['o'],
                'high': item['h'],
                'low': item['l'],
                'close': item['c'],
                'volume': item['v']
            })

        return jsonify(stock_data)
    except Exception as e:
        print(f"Unexpected error in stock_data: {str(e)}")
        return jsonify({'error': 'Failed to retrieve stock data'}), 500
    

if __name__ == '__main__':

    # uncomment the following 2 lines to run without ngrok (this is prod level code!)
    public_url = ngrok.connect(name='flask').public_url
    print(" * ngrok URL: " + public_url + " *")
    app.run()
