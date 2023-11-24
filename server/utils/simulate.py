from requests import requests


API_KEY = 'IyLEoMj6Ms39XVhSW4SeLF6WZl9BO8XZ'

def get_prices(asset: str, start: int, end: int):
    url = f'https://api.polygon.io/v2/aggs/ticker/{asset}/range/5/minute/{start}/{end}?adjusted=true&sort=asc&limit=120&apiKey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        prices_data = response.json()
        return prices_data['results']
    else:
        print(f"Failed to fetch prices. Status code: {response.status_code}")
        return None
    
def simulate(initial_capital: int, transactions: [], prices: []) -> {}:
    """
    Simple algo to spend as much when we can buy, sell as much when we should
    """
    transactions_with_amounts = transactions
    max_profit = 0
    buy_price = 0
    quantity_bought = 0
    bought_price = 0
    capital = initial_capital
    for transaction in transactions_with_amounts:
        transaction['amount'] = 0
        transaction['quantity'] = 0
    
    for idx, price in enumerate(prices):
        if(transactions[idx]['action'] == 'buy'):
            buy_price = min(price, key=lambda x: x['l'])
            if(capital < buy_price):
                break
            quantity_to_buy = capital // buy_price
            quantity_bought += quantity_to_buy
            value = quantity_to_buy * buy_price
            capital -= value
            bought_price = buy_price
            transactions_with_amounts[idx]['quantity'] = quantity_to_buy
            transactions_with_amounts[idx]['amount'] = value
        else:
            sell_price = max(price, key=lambda x: x['h'])
            if(sell_price < bought_price):
                break
            capital += quantity_bought * sell_price
            quantity_bought = 0
    
    if(quantity_bought > 0):
        # sell everything at the end
        final_price = transactions[-1]['end_time']
        max_profit += quantity_bought * (final_price - buy_price)
    return {
        'profits': max_profit,
        'transactions': transactions
    }
