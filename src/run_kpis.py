"""Run every KPI query in sql/kpi_queries.sql and save each to outputs/.

Usage
    python src/run_kpis.py              # run all queries
    python src/run_kpis.py 05           # run only queries whose name contains "05"
"""
import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine

ROOT = Path(__file__).resolve().parent.parent
QUERY_FILE = ROOT / "sql" / "kpi_queries.sql"
OUT_DIR = ROOT / "outputs"


def parse_queries(path: Path) -> dict[str, str]:
    """Split the file on '-- name: <name>' markers."""
    parts = re.split(r"^--\s*name:\s*(\S+)\s*$", path.read_text(), flags=re.M)
    # parts = [header, name1, body1, name2, body2, ...]
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def main() -> None:
    queries = parse_queries(QUERY_FILE)
    if len(sys.argv) > 1:
        queries = {k: v for k, v in queries.items() if sys.argv[1] in k}

    OUT_DIR.mkdir(exist_ok=True)
    engine = get_engine()
    pd.set_option("display.width", 160, "display.max_columns", 20)

    for name, sql in queries.items():
        # text() escapes literal % signs (e.g. in comments) for the driver
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"\n=== {name}  ({len(df)} rows -> outputs/{name}.csv)")
        print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
