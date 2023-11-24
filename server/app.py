from flask import Flask, request, jsonify
import requests
from datetime import datetime
from arb.arb import find_arbitrage_opportunity
from arb.arb import get_data
from arb.simulate import simulate_crypto, get_prices


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


@app.route('/api/crypto/arb', methods=['GET'])
def find_arbitrage():
    symbol = request.args.get('symbol')
    response = find_arbitrage_opportunity(get_data(symbol))
    return response

@app.route('/api/cryptoarb/simulate', methods=['GET'])
def simulate_cryptoarb():
    # issue here is cannot specifiy which crytpo exchange we want to work with
    try:
        data = request.args
        exchange_one = data.get('exchange_1', '')
        exchange_two = data.get('exchange_2', '')
        currency = data.get('currency_symbol')
        capital = data.get('capital', 0)
        transactions = data.get('transactions', [])

        prices = {exchange_one: [], exchange_two: []}

        for transaction in transactions:
            start = datetime.strptime(transaction['timestart1'], 'dd/mm/yyyy hh:mm:ss') * 1000
            end = datetime.strptime(transaction['timestart2'], 'dd/mm/yyyy hh:mm:ss') * 1000
            exchange_one_prices = get_prices(currency, start, end)
            exchange_two_prices = get_prices(currency, start, end)
            prices[exchange_one].append[exchange_one_prices]
            prices[exchange_two].append[exchange_two_prices]

        opportunity_1 = simulate_crypto(capital, prices[exchange_one])
        opportunity_2 = simulate_crypto(capital, prices[exchange_two])

        return {
            'opportunity1': opportunity_1,
            'opportunity2': opportunity_2
        }
    except Exception as e:
        return jsonify({'error': 'Invalid JSON data.'}), 400
    
@app.route('/api/stock/simulate', methods=['GET'])
def simulate_stock():
    try:
        data = request.args
        exchange_one = data.get('exchange_1', '')
        stock = data.get('stock_ticker')
        capital = data.get('capital', 0)
        transactions = data.get('transactions', [])
        prices =  []
        for transaction in transactions:
            start = datetime.strptime(transaction['timestart1'], 'dd/mm/yyyy hh:mm:ss') * 1000
            end = datetime.strptime(transaction['timestart2'], 'dd/mm/yyyy hh:mm:ss') * 1000
            prices.append[get_prices(stock, start, end)]
        simulation = simulate_crypto(capital, prices)
        return {
            simulation
        }
    except Exception as e:
     jsonify({'error': 'Invalid JSON data.'}), 400


if __name__ == '__main__':
    app.run()
