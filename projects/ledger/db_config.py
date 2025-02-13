from sqlalchemy import (ARRAY, NUMERIC, TIMESTAMP, Column, Integer, MetaData,
                        Table, Text, create_engine)
from sqlalchemy.dialects.postgresql import JSONB

DB_NAME = 'postgres'
DB_USER = 'reebxu'
DB_PASSWORD = 'watstreet'

engine = create_engine(
    f'postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/{DB_NAME}')

metadata = MetaData()

order_books = Table(
    'order_books_v2',
    metadata,
    Column('name', Text, primary_key=True),
    Column('tickers_to_track', ARRAY(Text)),
    Column('algo_link', Text, nullable=False),
    Column('update_time', Integer, nullable=False),
    Column('end_duration', Integer, nullable=False),
    Column('trades', JSONB, server_default='[]'),
    Column('worth', ARRAY(NUMERIC), server_default='{}'),
    Column('balance', NUMERIC, nullable=False, server_default='100000'),
    Column('created_at', TIMESTAMP, server_default='CURRENT_TIMESTAMP')
)


def get_db_connection():
    return engine.connect()
