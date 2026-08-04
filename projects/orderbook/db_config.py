import os

import psycopg2

DB_NAME = 'postgres'
DB_USER = 'reebxu'
DB_PASSWORD = os.getenv("ORDERBOOK_DB_PASSWORD")

def get_db_connection():
    if not DB_PASSWORD:
        raise RuntimeError("ORDERBOOK_DB_PASSWORD must be set")

    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host="localhost",
        port=5432
    )
