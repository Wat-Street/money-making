# Money Making Server

This server hosts REST API endpoints to use machine learning models on a specific dataset and evaluate performance.

## Accessing Endpoints Publicly

1. Use the base URL: `https://immense-alert-osprey.ngrok-free.app`
2. Append the compatible endpoints listed below to this base URL and call with the correct requests.

## Running Locally

1. Navigate to `server/app.py`.
2. Comment out ngrok-related lines.
3. Run `python app.py`.
4. The server will run locally at `http://localhost:5000`.

---

## Endpoints

### Mean Reversion Endpoint

Runs the mean reversion model and returns a series of buy/sell suggestions with confidence values.

- **URL:** `/api/meanreversion`
- **Method:** POST

#### Example Request
```json
{
  "ticker": "AAPL",
  "trainstart": "2015-01-01",
  "trainend": "2022-01-01",
  "teststart": "2022-01-01",
  "testend": "2024-05-06",
  "maxholding": "100"
}
```

#### Example Response:
```
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
```


### List Stocks

Lists 10 frequent stock tickers. Currently hardcoded

- **URL:** `/api/list_tickers`
- **Method:** GET

#### Example Request
```
```

#### Example Response:
```
[
    "AAPL",
    "AMZN",
    "GOOG",
    "META",
    "MSFT",
    "NVDA",
    "TSLA"
]
```



### Stock Data

Retrieves stock's daily information given a ticker, start date, and end date.

- **URL:** `/api/stock_data`
- **Method:** GET

#### Example Request:
```
{
  "ticker": "AAPL",
  "start_date": "2015-01-01", # optional
  "end_date": "2022-01-01", # optional
}
```

#### Example Response:
```
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
```