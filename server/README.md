# Money Making Server

Contains the database and handles REST API requests to use these models on a given subset of data and assess performance as needed.

## More Documentation to come



### Mean Reversion Endpoint
Basically runs our mean reversion model, and returns a series of buy/sell + confidence values. You can use the example below to structure your call and process the results.
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
  {
    "confidence": 1.0062354803085327,
    "datetime": "2022-12-08 00:00:00",
    "suggestion": "Buy"
  },
  {
    "confidence": 1.0176501274108887,
    "datetime": "2022-12-09 00:00:00",
    "suggestion": "Buy"
  },
  {
    "confidence": 1.0209354162216187,
    "datetime": "2022-12-12 00:00:00",
    "suggestion": "Buy"
  },
  {
    "confidence": 1.0153659582138062,
    "datetime": "2022-12-13 00:00:00",
    "suggestion": "Buy"
  },
  {
    "confidence": 1.0154021978378296,
    "datetime": "2022-12-14 00:00:00",
    "suggestion": "Buy"
  },
  {
    "confidence": 1.0138174295425415,
    "datetime": "2022-12-15 00:00:00",
    "suggestion": "Buy"
  },
  {
    "confidence": 1.0009783506393433,
    "datetime": "2022-12-16 00:00:00",
    "suggestion": "Buy"
  },
  {
    "confidence": 0.9905188679695129,
    "datetime": "2022-12-19 00:00:00",
    "suggestion": "Buy"
  },
  {
    "confidence": 0.16695882380008698,
    "datetime": "2022-03-14 00:00:00",
    "suggestion": "Sell"
  },
  {
    "confidence": 0.169448122382164,
    "datetime": "2022-07-22 00:00:00",
    "suggestion": "Sell"
  },
  {
    "confidence": 0.1353330761194229,
    "datetime": "2022-07-25 00:00:00",
    "suggestion": "Sell"
  },
  {
    "confidence": 0.10780976712703705,
    "datetime": "2022-07-26 00:00:00",
    "suggestion": "Sell"
  },
  {
    "confidence": 0.07454611361026764,
    "datetime": "2022-07-27 00:00:00",
    "suggestion": "Sell"
  },
  {
    "confidence": 0.04149295389652252,
    "datetime": "2022-07-28 00:00:00",
    "suggestion": "Sell"
  },
  {
    "confidence": 0.035939738154411316,
    "datetime": "2022-07-29 00:00:00",
    "suggestion": "Sell"
  },
  {
    "confidence": 0.04654340445995331,
    "datetime": "2022-08-01 00:00:00",
    "suggestion": "Sell"
  },
  {
    "confidence": 0.07429952919483185,
    "datetime": "2022-08-02 00:00:00",
    "suggestion": "Sell"
  },
  {
    "confidence": 0.13511483371257782,
    "datetime": "2022-08-03 00:00:00",
    "suggestion": "Sell"
  }
]