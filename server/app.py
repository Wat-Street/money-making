from flask import Flask, request, jsonify
import requests
from arb.arb import find_arbitrage_opportunity
from arb.arb import get_data
from meanrev.Mean_Reversion_Test import test
from meanrev.Mean_Reversion_Train import train

import yfinance as yf

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

if __name__ == '__main__':
    app.run(port=5002)
    # listener = ngrok.forward(5000, authtoken_from_env=True)
    # print(f"Ingress established at {listener.url()}")


