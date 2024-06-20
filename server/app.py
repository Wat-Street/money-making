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
        
        # historic_data_train = yf.download(ticker=request.json.ticker, start=request.json.trainstart, end=request.json.trainend,  interval = "1d")
        # prices_train = historic_data_train['Adj Close']
        # historic_data_test = yf.download(ticker=request.json.ticker, start=request.json.teststart, end=request.json.testend,  interval = "1d")
        # prices_test = historic_data_test['Adj Close']
        # model = train(prices_train, request.json.maxholding)
        # suggestions = test(model, prices_test)
        return jsonify({'message': 'This is a POST request.', 'data': jsonify(request.json)})
    except Exception as e:
        return jsonify({'error': 'Invalid JSON data.'}), 400

# @app.route('/api/meanreversion/graph', methods=['GET'])
# def meanreversion_graph():
#     try:
#         historic_data = yf.download(ticker=request.json.ticker, start=request.json.start, end=request.json.end,  interval = "1d")
#         prices = historic_data['Adj Close']
#         # https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?apiKey=sAIqmjrJ4z9za1KjU1Ejk3hCi2A5VlbJ
#         # return jsonify({'message': 'This is a POST request.', 'data': data})
#     except Exception as e:
#         return jsonify({'error': 'Invalid JSON data.'}), 400

# @app.route('/api/crypto')
# def crypto_endpoint():
#     return "Crypto endpoint methods: /book, /exchanges"


# @app.route('/api/crypto/book', methods=['GET'])
# def get_book_data():
#     symbol = request.args.get('symbol')
#     url_data = f'https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/X:{symbol}USD/book?apiKey={API_KEY}'
#     response = requests.get(url_data)
#     if response.status_code == 200:
#         data = response.json()
#         return data
#     else:
#         print(f"Failed to fetch data. Status code: {response.status_code}")
#         return None


# @app.route('/api/crypto/exchanges', methods=['GET'])
# def get_exchanges():
#     url_exchanges = f'https://api.polygon.io/v3/reference/exchanges?asset_class=crypto&apiKey={API_KEY}'
#     response = requests.get(url_exchanges)
#     if response.status_code == 200:
#         exchange_data = response.json()
#         exchange_names = {exchange['id']: exchange['name'] for exchange in exchange_data['results']}
#         print(exchange_data)
#         return exchange_names
#     else:
#         print(f"Failed to fetch exchange names. Status code: {response.status_code}")
#         return None


# @app.route('/api/crypto/arb', methods=['GET'])
# def find_arbitrage():
#     print(request)
#     symbol = request.args.get('symbol')
#     response = find_arbitrage_opportunity(get_data(symbol))
#     return response


if __name__ == '__main__':
    app.run(port=5002)
    # listener = ngrok.forward(5000, authtoken_from_env=True)
    # print(f"Ingress established at {listener.url()}")


