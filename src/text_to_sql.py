"""Phase 6: ask business questions in plain English, get SQL + answers.

How it works
  1. The LLM gets the database schema, business rules (what counts as
     revenue, which customer ID to use ...) and a few worked examples.
  2. It writes one PostgreSQL SELECT query.
  3. The query is checked (single read-only SELECT) and run inside a
     READ ONLY transaction with a timeout, so it can never change data.
  4. If PostgreSQL returns an error, the error is sent back to the LLM to
     fix the query (up to 2 retries).

Accuracy is measured on a set of questions with hand-written correct SQL
(eval/text_to_sql_questions.json): a generated query counts as correct
when it returns the same results.

Usage
    python src/text_to_sql.py "Which state has the highest average order value?"
    python src/text_to_sql.py --eval              # accuracy on the question set
    python src/text_to_sql.py --eval --limit 10
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

import llm
from db import get_engine, save_metrics

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "eval" / "text_to_sql_questions.json"
OUT_DIR = ROOT / "outputs"

# Tables/views the assistant may query (others are ignored even if present)
ALLOWED_OBJECTS = [
    "v_orders_enriched", "v_customer_summary", "orders", "order_items",
    "order_payments", "order_reviews", "customers", "products", "sellers",
    "category_translation", "customer_segments", "churn_scores",
    "revenue_forecasts", "review_sentiment",
]

BUSINESS_RULES = """
Business rules (follow them exactly):
- Revenue = item price + freight. For order-level questions use v_orders_enriched.order_total.
- Unless the question says otherwise, only count orders with order_status = 'delivered'
  for revenue, delivery and customer-behaviour metrics.
- customers.customer_id is created per ORDER. To count or analyse real people use
  customer_unique_id (available in v_orders_enriched and v_customer_summary).
- Product categories: show category_translation.product_category_name_english
  (LEFT JOIN on product_category_name; fall back to the Portuguese name).
- v_orders_enriched has one row per order with: customer_unique_id, customer_state,
  order_status, purchase_ts, order_month, delivered_ts, estimated_ts, n_items,
  items_revenue, freight_value, order_total, payment_value, main_payment_type,
  max_installments, review_score (average if several), delivery_days,
  days_vs_estimate (positive = late), is_late (1/0).
- v_customer_summary has one row per customer (delivered orders only): frequency,
  monetary, first_purchase_ts, last_purchase_ts, avg_review_score, late_share.
- States are two-letter Brazilian codes (SP = Sao Paulo, RJ = Rio de Janeiro ...).
- Percentages: multiply by 100.0. Round money and averages to 2 decimals.
"""

FEW_SHOT = """
Example questions and correct SQL:

Q: What is the monthly revenue in 2017?
SQL:
SELECT order_month, ROUND(SUM(order_total), 2) AS revenue
FROM v_orders_enriched
WHERE order_status = 'delivered'
  AND purchase_ts >= '2017-01-01' AND purchase_ts < '2018-01-01'
GROUP BY order_month
ORDER BY order_month;

Q: Which 3 categories have the most orders?
SQL:
SELECT COALESCE(t.product_category_name_english, p.product_category_name) AS category,
       COUNT(DISTINCT oi.order_id) AS orders
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY 1
ORDER BY orders DESC
LIMIT 3;

Q: How many customers bought more than once?
SQL:
SELECT COUNT(*) AS repeat_customers
FROM v_customer_summary
WHERE frequency > 1;
"""

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|"
    r"vacuum|reindex|cluster|comment|lock|call|do|pg_sleep|pg_read_file|"
    r"set_config|dblink)\b", re.IGNORECASE)


# ---------------------------------------------------------------------
# Schema context
# ---------------------------------------------------------------------
def schema_description(engine) -> str:
    sql = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = ANY(:names)
        ORDER BY table_name, ordinal_position
    """
    with engine.connect() as conn:
        cols = pd.read_sql(text(sql), conn, params={"names": ALLOWED_OBJECTS})
    lines = []
    for table in ALLOWED_OBJECTS:
        t = cols[cols["table_name"] == table]
        if len(t):
            fields = ", ".join(f"{c} ({d})" for c, d in zip(t["column_name"], t["data_type"]))
            lines.append(f"- {table}: {fields}")
    return "\n".join(lines)


def system_prompt(schema: str) -> str:
    return f"""You are an expert data analyst who writes PostgreSQL queries for an
e-commerce database (Olist, Brazil, 2016-2018).

Database objects:
{schema}
{BUSINESS_RULES}
{FEW_SHOT}
Rules for your answer:
- Write exactly ONE read-only PostgreSQL query (SELECT or WITH ... SELECT).
- Use only the tables and columns listed above.
- Give columns short, readable aliases.
- Add LIMIT 100 unless the result is naturally small (aggregates, top-N).
- Reply with the SQL only, inside a ```sql code block. No explanation."""


# ---------------------------------------------------------------------
# Generation, validation, execution
# ---------------------------------------------------------------------
def extract_sql(reply: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    sql = (m.group(1) if m else reply).strip()
    return sql.rstrip(";").strip()


def validate_sql(sql: str) -> None:
    body = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", body)
    if not re.match(r"^\s*(select|with)\b", no_strings, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    if ";" in no_strings.strip().rstrip(";"):
        raise ValueError("Only one statement is allowed.")
    bad = FORBIDDEN.search(no_strings)
    if bad:
        raise ValueError(f"Keyword not allowed: {bad.group(0).upper()}")


def run_query(engine, sql: str, max_rows: int = 1000, timeout_ms: int = 15000) -> pd.DataFrame:
    """Execute inside a READ ONLY transaction with a statement timeout.

    Even if validation missed something, PostgreSQL itself rejects any write.
    Uses a raw cursor so literal % signs (LIKE '%x%') need no escaping.
    """
    validate_sql(sql)
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        raw.rollback()
        raw.close()


class SQLAssistant:
    def __init__(self, engine=None, max_retries: int = 2):
        self.engine = engine or get_engine()
        self.max_retries = max_retries
        self.system = system_prompt(schema_description(self.engine))

    def generate(self, question: str, previous_sql: str | None = None,
                 error: str | None = None) -> str:
        user = f"Question: {question}"
        if previous_sql:
            user += (f"\n\nYour previous query failed.\nQuery:\n{previous_sql}\n"
                     f"Error:\n{error}\nWrite a corrected query.")
        return extract_sql(llm.complete(self.system, user))

    def ask(self, question: str) -> dict:
        """Returns {'sql', 'result' (DataFrame or None), 'error', 'attempts'}."""
        sql, error = None, None
        for attempt in range(1, self.max_retries + 2):
            try:
                sql = self.generate(question, sql, error)
                df = run_query(self.engine, sql)
                return {"sql": sql, "result": df, "error": None, "attempts": attempt}
            except Exception as exc:  # LLM, validation or database error
                error = str(exc).strip().splitlines()[0][:500]
        return {"sql": sql, "result": None, "error": error, "attempts": attempt}

    def summarize(self, question: str, df: pd.DataFrame) -> str:
        """Optional: one or two plain-English sentences about the result."""
        preview = df.head(20).to_string(index=False)
        return llm.complete(
            "You explain query results to business users in 1-2 short sentences. "
            "Use only the numbers given. No SQL, no preamble.",
            f"Question: {question}\n\nResult:\n{preview}", max_tokens=200).strip()


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def _normalize_value(v) -> str:
    # 2018-01-01, datetime(2018, 1, 1) and "2018-01-01 00:00:00" all match
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "isoformat"):
        ts = pd.Timestamp(v)
        return str(ts.date()) if ts == ts.normalize() else ts.isoformat()
    return str(v).strip().lower()


def _normalize_column(s: pd.Series):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all() and len(s):
        return ("num", np.sort(num.to_numpy(float)))
    return ("str", sorted(_normalize_value(v) for v in s))


def _same_column(a, b) -> bool:
    if a[0] != b[0] or len(a[1]) != len(b[1]):
        return False
    if a[0] == "num":
        return bool(np.allclose(a[1], b[1], rtol=1e-2, atol=1e-2))
    return a[1] == b[1]


def results_match(gold: pd.DataFrame, pred: pd.DataFrame) -> bool:
    """Execution match: same number of rows, and every gold column has a
    matching column in the prediction (any order, any name, 1% tolerance
    for numbers). Extra predicted columns are allowed."""
    if pred is None or len(gold) != len(pred):
        return False
    pred_cols = [_normalize_column(pred[c]) for c in pred.columns]
    used = set()
    for c in gold.columns:
        g = _normalize_column(gold[c])
        hit = next((i for i, p in enumerate(pred_cols)
                    if i not in used and _same_column(g, p)), None)
        if hit is None:
            return False
        used.add(hit)
    return True


def evaluate(limit: int | None = None) -> dict:
    questions = json.loads(EVAL_FILE.read_text())[:limit]
    assistant = SQLAssistant()
    rows = []
    for i, q in enumerate(questions, 1):
        gold = run_query(assistant.engine, q["sql"])
        start = time.time()
        out = assistant.ask(q["question"])
        ok = results_match(gold, out["result"])
        rows.append({"id": q["id"], "question": q["question"], "correct": ok,
                     "attempts": out["attempts"], "error": out["error"],
                     "seconds": round(time.time() - start, 1),
                     "generated_sql": out["sql"]})
        print(f"  [{i:>2}/{len(questions)}] {'PASS' if ok else 'FAIL'}  {q['question']}")
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "text_to_sql_eval.csv", index=False)
    summary = {"provider": llm.provider(), "model": llm.model_name(),
               "questions": len(df),
               "execution_accuracy": round(df["correct"].mean(), 4),
               "first_try_accuracy": round((df["correct"] & (df["attempts"] == 1)).mean(), 4),
               "failed_to_run": int(df["error"].notna().sum()),
               "avg_seconds": round(df["seconds"].mean(), 1)}
    (OUT_DIR / "text_to_sql_metrics.json").write_text(json.dumps(summary, indent=2))
    save_metrics(assistant.engine, "text_to_sql", summary)
    print("\n=== Text-to-SQL evaluation ===")
    for k, v in summary.items():
        print(f"  {k:<20} {v}")
    print("Per-question results: outputs/text_to_sql_eval.csv")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="a business question in English")
    parser.add_argument("--eval", action="store_true", help="run the accuracy evaluation")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.limit)
    elif args.question:
        a = SQLAssistant()
        out = a.ask(args.question)
        print(f"\nSQL ({out['attempts']} attempt(s)):\n{out['sql']}\n")
        if out["error"]:
            print(f"Error: {out['error']}")
        else:
            print(out["result"].head(20).to_string(index=False))
            print(f"\n{a.summarize(args.question, out['result'])}")
    else:
        parser.print_help()
