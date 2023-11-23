from flask import Flask, request, jsonify
from datetime import datetime
import requests
from arb.arb import find_arbitrage_opportunity
from arb.arb import get_data


app = Flask(__name__)

API_KEY = 'IyLEoMj6Ms39XVhSW4SeLF6WZl9BO8XZ'


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


@app.route('/api/crypto')
def crypto_endpoint():
    return "Crypto endpoint methods: /book, /exchanges"


@app.route('/api/crypto/book', methods=['GET'])
def get_book_data():
    symbol = request.args.get('symbol')
    url_data = f'https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/X:{symbol}USD/book?apiKey={API_KEY}'
    response = requests.get(url_data)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None


@app.route('/api/crypto/exchanges', methods=['GET'])
def get_exchanges():
    url_exchanges = f'https://api.polygon.io/v3/reference/exchanges?asset_class=crypto&apiKey={API_KEY}'
    response = requests.get(url_exchanges)
    if response.status_code == 200:
        exchange_data = response.json()
        exchange_names = {exchange['id']: exchange['name'] for exchange in exchange_data['results']}
        return exchange_names
    else:
        print(f"Failed to fetch exchange names. Status code: {response.status_code}")
        return None
    
@app.route('/api/stock', methods=['GET'])
def get_stock_data():
    # Get parameters from the request
    stock_ticker = request.args.get('stock_ticker')
    start_datetime_str = request.args.get('start')
    end_datetime_str = request.args.get('end')

    # Validate parameters
    if not all([stock_ticker, start_datetime_str, end_datetime_str]):
        return jsonify({'error': 'Missing required parameters'}), 400

    try:
        # Parse datetime strings
        start_datetime = datetime.strptime(start_datetime_str, '%m/%d/%Y %H:%M:%S')
        end_datetime = datetime.strptime(end_datetime_str, '%m/%d/%Y %H:%M:%S')
    except ValueError:
        return jsonify({'error': 'Invalid datetime format'}), 400

    # Call Polygon API to get stock data
    polygon_url = f'https://api.polygon.io/v2/aggs/ticker/{stock_ticker}/range/1/minute/{start_datetime.timestamp()*1000}/{end_datetime.timestamp()*1000}?apiKey={API_KEY}'
    
    try:
        response = requests.get(polygon_url)
        response.raise_for_status()
        stock_data = response.json()
        return jsonify(stock_data)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Polygon API request failed: {str(e)}'}), 400

@app.route('/api/crypto/arb', methods=['GET'])
def find_arbitrage():
    symbol = request.args.get('symbol')
    response = find_arbitrage_opportunity(get_data(symbol))
    return response


if __name__ == '__main__':
    app.run()
