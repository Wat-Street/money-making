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

-- creating user
CREATE USER reebxu WITH SUPERUSER PASSWORD 'watstreet';
-- database name: postgres


