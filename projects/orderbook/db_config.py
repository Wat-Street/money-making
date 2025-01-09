import psycopg2

DB_NAME = 'postgres'
DB_USER = 'reebxu'
DB_PASSWORD = 'watstreet'

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host="localhost",
        port=5432
    )
