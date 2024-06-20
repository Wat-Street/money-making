import requests
import json
from datetime import datetime


def get_data(symbol):
    book_data_url = 'http://127.0.0.1:5000/api/crypto/book'
    response = requests.get(book_data_url, params={'symbol': symbol})
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None


def get_exchanges():
    exchanges_url = 'http://127.0.0.1:5000/api/crypto/exchanges'
    response = requests.get(exchanges_url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch exchange names. Status code: {response.status_code}")
        return None


def find_arbitrage_opportunity(data):
    if data is None:
        return "No data available"
    arbitrage_found = False

    # Extracting asks data
    asks = data['data']['asks']

    # Dictionary to map exchange prices by their IDs
    exchange_prices = {}

    exchange_names = get_exchanges()
    print("exchanges", exchange_names)

    # Processing asks
    for ask in asks:
        for exchange_id, price in ask['x'].items():
            if exchange_id not in exchange_prices:
                exchange_prices[exchange_id] = {'ask': ask['p'], 'ask': None}
            else:
                exchange_prices[exchange_id]['ask'] = ask['p']

    opportunities = {}
    opportunity_count = 1

    # Check for arbitrage opportunities between different exchanges
    for exchange_id, prices in exchange_prices.items():
        print(exchange_id)
        print(exchange_prices)
        for other_exchange_id, other_prices in exchange_prices.items():
            if exchange_id != other_exchange_id:
                if prices['ask'] and other_prices['ask']:
                    spread = other_prices['ask'] - prices['ask']
                    if spread > 0:
                        arbitrage_found = True
                        print(
                            f"Arbitrage opportunity found between Exchange {exchange_id} and Exchange {other_exchange_id}: Buy at {prices['ask']} and sell at {other_prices['ask']}")

                        opportunity = {
                            'action1': 'buy',
                            'timestart': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                            'exchange_id_buy': exchange_id,
                            'price_buy': prices['ask'],
                            'action2': 'sell',
                            'timeend': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                            'exchange_id_sell': other_exchange_id,
                            'price_sell': other_prices['ask'],
                        }

                        opportunities[f'opportunity{opportunity_count}'] = opportunity
                        opportunity_count += 1

        json_output = {
            'exchanges': exchange_names,
            'arbitrage_opportunities': opportunities
        }
        print(json.dumps(json_output, indent=4))
        return json.dumps(json_output, indent=4)

    if not arbitrage_found:
        return "No arb found"


# Call the function with the symbol
symbol = 'BTC'
# find_arbitrage_opportunity(get_data(symbol))
