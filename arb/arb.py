import requests


def get_data():
    book_data_url = 'http://127.0.0.1:5000/api/crypto/book'
    response = requests.get(book_data_url)
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
    arbitrage_found = False

    # Extracting asks data
    asks = data['data']['asks']

    # Dictionary to map exchange prices by their IDs
    exchange_prices = {}

    # Processing asks
    for ask in asks:
        for exchange_id, price in ask['x'].items():
            if exchange_id not in exchange_prices:
                exchange_prices[exchange_id] = {'ask': None, 'bid': None}
            if ask['p'] < exchange_prices[exchange_id]['ask'] or exchange_prices[exchange_id]['ask'] is None:
                exchange_prices[exchange_id]['ask'] = ask['p']

    # Check for arbitrage opportunities between different exchanges
    for exchange_id, prices in exchange_prices.items():
        for other_exchange_id, other_prices in exchange_prices.items():
            if exchange_id != other_exchange_id:
                if prices['ask'] and other_prices['bid']:
                    spread = other_prices['bid'] - prices['ask']
                    if spread > 0:
                        arbitrage_found = True
                        print(
                            f"Arbitrage opportunity found between Exchange {exchange_id} and Exchange {other_exchange_id}: Buy at {prices['ask']} and sell at {other_prices['bid']}")

    if not arbitrage_found:
        print("No arbitrage opportunity found.")


# Call the function with the provided data
find_arbitrage_opportunity(get_data())
