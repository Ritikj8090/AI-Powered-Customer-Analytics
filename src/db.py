"""Shared database connection helper.

Reads DATABASE_URL from the environment (or a .env file), e.g.
    postgresql://postgres:postgres@localhost:5432/olist
"""
import csv
import io
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
        )
    # Hosted providers (Neon, Supabase) sometimes give "postgres://" URLs;
    # SQLAlchemy needs "postgresql://".
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    # Pin the psycopg2 driver: the fast COPY loader relies on it, and newer
    # SQLAlchemy versions otherwise default to a different driver.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(url)


def copy_insert(table, conn, keys, data_iter):
    """pandas.to_sql 'method' that uses PostgreSQL COPY.

    Much faster than row-by-row INSERTs (the geolocation file alone
    has ~1M rows). Use as: df.to_sql(..., method=copy_insert)
    """
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        buf = io.StringIO()
        csv.writer(buf).writerows(data_iter)
        buf.seek(0)
        columns = ", ".join(f'"{k}"' for k in keys)
        name = f"{table.schema}.{table.name}" if table.schema else table.name
        cur.copy_expert(f"COPY {name} ({columns}) FROM STDIN WITH CSV", buf)
