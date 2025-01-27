# Order Book Instances

## Overview
Order Book instances allow the ML team to forward-test their algorithms. An API and database are hosted on the PC that allows members to create instances of Order Books and tie specific algorithms to trade on them.

## API Features
1. **`create_orderbook`**
    To create an order book.

    Expected arguments:
    - `name`: **unique** name of algorithm
    - `tickerstotrack`: two tickers (e.g. (AAPL, GOOG))
    - `algo_path`: path to algorithm from the `projects` directory (ex. algo_path=harv-extension)
    - `updatetime`: time interval for updates (minutes)
    - `end`: lifespan of instance (days)

    Example command: `https://watstreet.orderbook.create?name=krishalgo&tickerstotrack=AAPL,GOOG&algo=linktoalgo&updatetime=1&end=100`

2. **`view_orderbook`**
    To retrieve details of a specific order book.

    Expected arguments:
    - `name`: name of algorithm

    Example command: `https://watstreet.orderbook.view?name=krishalgo`

3. **`delete_orderbook`**
    To delete an order book.

    Expected arguments:
    - `name`: name of algorithm

## Interaction with Models
The two standard commands are:
- `trade()`: runs the algorithm and, based on current ownership of stocks and balance, will return a trade
- `update_book_status(book_status)`: gives the trading algorithm the info it needs to make trades (portfolio and balance)

## Data Formats (TODO:)
**KEEP THIS UPDATED AT ALL TIMES.** 
#### Basic Model Output
```
{
  "trades": [
    {"type": "buy", "ticker": "AAPL", "price": 176, "quantity": 8},
    {"type": "sell", "ticker": "AAPL", "price": 157, "quantity": 7},
    {"type": "buy", "ticker": "GOOG", "price": 112, "quantity": 9}
  ],
}
```
#### Updating Book In Model
```
{
  "portfolio": {
    "GOOG": 9,
    "AAPL": 10,
    "MSFT": 34,
    "AMZN": 4
  },
  "balance": 68208
}

```

#### View Book
```
{
  "trades": [
    {"type": "buy", "ticker": "AAPL", "price": 176, "quantity": 8},
    {"type": "sell", "ticker": "AAPL", "price": 157, "quantity": 7},
    {"type": "buy", "ticker": "GOOG", "price": 112, "quantity": 9}
  ],
  “holding”: {
    "GOOG": 9,
    "AAPL": 10,
    "MSFT": 34,
    "AMZN": 4
  }.
  "value": [80000, 81000, 83000, 80000],
  "balance": 78549
}
```

## Design Components (wip)
1. API Backend (Flask)
2. Database (PostgreSQL, in Rebbi's local env)
Database name: `postgres` for now
Tables: (wip)
- `order_books`
- `trades`
3. Scheduler (Celery?) (wip)






