-- example entry in order_books
INSERT INTO order_books (name, trades, worth, balance)
VALUES (
    'krishalgo',
    '[{"type": "buy", "ticker": "AAPL", "price": 176, "quantity": 8}]',
    '{15, 19, 40}',
    78549
);


-- example entry in order_books_v2
INSERT INTO order_books_v2 (name, tickers_to_track, algo_link, update_time, end_duration)
VALUES ('test', '{"AAPL","GOOG"}', 'hi', '1', '100');


-- table creation
CREATE TABLE order_books_v2 (
    name TEXT PRIMARY KEY,
    tickers_to_track TEXT[],
    algo_link TEXT NOT NULL,
    update_time INT NOT NULL,
    end_duration INT NOT NULL,
    trades JSONB DEFAULT '[]',
    worth NUMERIC[] DEFAULT '{}',
    balance NUMERIC NOT NULL DEFAULT 100000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);