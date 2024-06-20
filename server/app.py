from flask import Flask, request, jsonify
import requests

from meanrev.Mean_Reversion_Test import test
from meanrev.Mean_Reversion_Train import train
import pandas as pd
import requests
import datetime
from dotenv import load_dotenv
import os

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
        
        # Create the response in the desired format
        
        return jsonify(suggestions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

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
    app.run(port=5002)
    # listener = ngrok.forward(5000, authtoken_from_env=True)
    # print(f"Ingress established at {listener.url()}")


