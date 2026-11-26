"""Load the 9 Olist CSV files into PostgreSQL.

Steps
  1. Create the schema   (sql/schema.sql)
  2. Clean each CSV      (rename typo columns, parse dates, drop duplicates)
  3. Bulk-load tables    (pandas.to_sql with PostgreSQL COPY, in FK order)
  4. Create views        (sql/views.sql)
  5. Print row counts and data-quality checks

Usage
    python src/load_data.py                  # CSVs in data/raw/
    python src/load_data.py --data-dir path/to/csvs
"""
import argparse
import csv
import io
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "sql"

# table name -> (csv file, date columns)
# Order matters: parents are loaded before the children that reference them.
TABLES = {
    "category_translation": ("product_category_name_translation.csv", []),
    "customers":            ("olist_customers_dataset.csv", []),
    "sellers":              ("olist_sellers_dataset.csv", []),
    "products":             ("olist_products_dataset.csv", []),
    "geolocation":          ("olist_geolocation_dataset.csv", []),
    "orders":               ("olist_orders_dataset.csv", [
                                "order_purchase_timestamp", "order_approved_at",
                                "order_delivered_carrier_date",
                                "order_delivered_customer_date",
                                "order_estimated_delivery_date"]),
    "order_items":          ("olist_order_items_dataset.csv", ["shipping_limit_date"]),
    "order_payments":       ("olist_order_payments_dataset.csv", []),
    "order_reviews":        ("olist_order_reviews_dataset.csv", [
                                "review_creation_date", "review_answer_timestamp"]),
}

# Columns that must stay text (zip prefixes have leading zeros).
TEXT_COLUMNS = {
    "customer_zip_code_prefix": str,
    "seller_zip_code_prefix": str,
    "geolocation_zip_code_prefix": str,
}

# Primary keys, used to drop duplicate rows before loading.
PRIMARY_KEYS = {
    "category_translation": ["product_category_name"],
    "customers": ["customer_id"],
    "sellers": ["seller_id"],
    "products": ["product_id"],
    "orders": ["order_id"],
    "order_items": ["order_id", "order_item_id"],
    "order_payments": ["order_id", "payment_sequential"],
    "order_reviews": ["review_id", "order_id"],
}


def copy_insert(table, conn, keys, data_iter):
    """pandas.to_sql 'method' that uses PostgreSQL COPY.

    Much faster than row-by-row INSERTs (the geolocation file alone
    has ~1M rows).
    """
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        buf = io.StringIO()
        csv.writer(buf).writerows(data_iter)
        buf.seek(0)
        columns = ", ".join(f'"{k}"' for k in keys)
        name = f"{table.schema}.{table.name}" if table.schema else table.name
        cur.copy_expert(f"COPY {name} ({columns}) FROM STDIN WITH CSV", buf)


def read_and_clean(table: str, path: Path, date_cols: list[str]) -> pd.DataFrame:
    # utf-8-sig strips the byte-order mark at the start of some Olist files
    df = pd.read_csv(path, dtype=TEXT_COLUMNS, encoding="utf-8-sig")

    # Fix Olist's misspelled column names ("lenght" -> "length")
    df.columns = [c.replace("lenght", "length") for c in df.columns]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if table in PRIMARY_KEYS:
        before = len(df)
        df = df.drop_duplicates(subset=PRIMARY_KEYS[table])
        dropped = before - len(df)
        if dropped:
            print(f"    dropped {dropped:,} duplicate rows on {PRIMARY_KEYS[table]}")

    # Product measurements have blanks, so pandas reads them as floats
    # ("40.0"), which PostgreSQL rejects for INTEGER columns.
    if table == "products":
        int_cols = ["product_name_length", "product_description_length",
                    "product_photos_qty"]
        df[int_cols] = df[int_cols].astype("Int64")

    return df


def run_sql_file(engine, path: Path) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=ROOT / "data" / "raw", type=Path)
    args = parser.parse_args()

    missing = [f for f, _ in TABLES.values() if not (args.data_dir / f).exists()]
    if missing:
        raise SystemExit(f"Missing CSV files in {args.data_dir}:\n  " + "\n  ".join(missing))

    engine = get_engine()
    start = time.time()

    print("1/4  Creating schema ...")
    run_sql_file(engine, SQL_DIR / "schema.sql")

    print("2/4  Loading tables ...")
    for table, (filename, date_cols) in TABLES.items():
        df = read_and_clean(table, args.data_dir / filename, date_cols)
        df.to_sql(table, engine, if_exists="append", index=False,
                  method=copy_insert, chunksize=50_000)
        print(f"    {table:<22} {len(df):>10,} rows")

    print("3/4  Creating analysis views ...")
    run_sql_file(engine, SQL_DIR / "views.sql")

    print("4/4  Data-quality checks ...")
    checks = {
        "orders without items":
            "SELECT COUNT(*) FROM orders o WHERE NOT EXISTS "
            "(SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)",
        "delivered orders missing delivery date":
            "SELECT COUNT(*) FROM orders WHERE order_status = 'delivered' "
            "AND order_delivered_customer_date IS NULL",
        "products without category":
            "SELECT COUNT(*) FROM products WHERE product_category_name IS NULL",
        "categories missing English name":
            "SELECT COUNT(DISTINCT p.product_category_name) FROM products p "
            "LEFT JOIN category_translation t USING (product_category_name) "
            "WHERE p.product_category_name IS NOT NULL "
            "AND t.product_category_name IS NULL",
        "real customers (customer_unique_id)":
            "SELECT COUNT(DISTINCT customer_unique_id) FROM customers",
    }
    with engine.connect() as conn:
        for label, sql in checks.items():
            print(f"    {label:<42} {conn.execute(text(sql)).scalar():>10,}")

    print(f"\nDone in {time.time() - start:.0f}s.")


if __name__ == "__main__":
    main()
