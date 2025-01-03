-- table creation
CREATE TABLE order_books (
    name TEXT PRIMARY KEY,
    trades JSONB NOT NULL DEFAULT '[]',
    worth NUMERIC[] NOT NULL DEFAULT '{}',
    balance NUMERIC NOT NULL DEFAULT 100000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- creating user
CREATE USER reebxu WITH SUPERUSER PASSWORD 'watstreet';
-- database name: postgres


