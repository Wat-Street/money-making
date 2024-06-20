# Money Making Server

Contains the database and handles REST API requests to use these models on a given subset of data and assess performance as needed.

## More Documentation to come



### Mean Reversion Endpoint
Basically runs our mean reversion model, and returns a series of buy/sell + confidence values. You can use the example below to structure your call and process the results.

URL: /api/meanreversion

Method: POST

Example request:
{
  "ticker": "AAPL",
  "trainstart": "2015-01-01",
  "trainend": "2022-01-01",
  "teststart": "2022-01-01",
  "testend": "2024-05-06",
  "maxholding": "100"
}

Example response:
[
  {
    "confidence": 0.9731735587120056,
    "datetime": "2022-12-06 00:00:00",
    "suggestion": "Buy"
  },
  {
    "confidence": 0.9925611019134521,
    "datetime": "2022-12-07 00:00:00",
    "suggestion": "Buy"
  },
  ...
]



### List Stocks

URL: /api/list_tickers

Method: GET

Description: Lists 10 frequent stock tickers. Currently hardcoded

Example Request:

Example Response:
[
    "AAPL",
    "AMZN",
    "GOOG",
    "META",
    "MSFT",
    "NVDA",
    "TSLA"
]

### Stock Data

URL: /api/stock_data

Method: GET

Description: Retrieves stock's daily information given a ticker, start date, and end date.

Example Request:

{
  "ticker": "AAPL",
  "start_date": "2015-01-01", # optional
  "end_date": "2022-01-01", # optional
}

Example Response:
[
    {
        "datetime": "2023-01-01 00:00:00",
        "open": 130.00,
        "high": 135.00,
        "low": 128.00,
        "close": 132.00,
        "volume": 100000
    },
    {
        "datetime": "2023-01-02 00:00:00",
        "open": 132.00,
        "high": 136.00,
        "low": 130.00,
        "close": 134.00,
        "volume": 110000
    }
]