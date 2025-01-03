# Order Book Instances

## Overview
Order Book instances allow the ML team to forward-test their algorithms. An API and database are hosted on the PC that allows members to create instances of Order Books and tie specific algorithms to trade on them.

## API Features
### `create_orderbook`
To create an order book.

Expected arguments:
- `name`: **unique** name of algorithm
- `tickerstotrack`: two tickers (e.g. (AAPL, GOOG))
- `algo`: link to algorithm
- `updatetime`: time interval for updates (minutes)
- `end`: lifespan of instance (days)

Example command: `https://watstreet.orderbook.create?name=krishalgo&tickerstotrack=AAPL,GOOG&algo=linktoalgo&updatetime=1&end=100`

### `view_orderbook`
To retrieve details of a specific order book.

Expected arguments:
- `name`: name of algorithm

Example command: `https://watstreet.orderbook.view?name=krishalgo`

### `delete_orderbook`
To delete an order book.

Expected arguments:
- `name`: name of algorithm

## Interaction with Models
The two standard commands are:
- `trade()`: runs the algorithm and, based on current ownership of stocks and balance, will return a trade
- `update_book_status(book_status)`: gives the trading algorithm the info it needs to make trades (portfolio and balance)

## Data Formats (TODO:)
**KEEP THIS UPDATED AT ALL TIMES** 


## Design Components (wip)
1. API Backend (Flask)
2. Database (PostgreSQL, in Rebbi's local env)
Database name: `postgres` for now
Tables: (wip)
- `order_books`
- `trades`
3. Scheduler (Celery?) (wip)






