from flask import Flask, request, jsonify
import requests

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
    return "Available crypto endpoint methods: /book, /exchanges"


@app.route('/api/crypto/book', methods=['GET'])
def get_book_data():
    url_data = f'https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/X:BTCUSD/book?apiKey={API_KEY}'
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


if __name__ == '__main__':
    app.run()
