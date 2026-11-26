# Customer Intelligence Platform

End-to-end analytics on 100K+ real e-commerce orders from Olist, Brazil's largest
online marketplace: SQL KPIs, customer segmentation, churn prediction, revenue
forecasting, review sentiment, and an LLM-powered text-to-SQL assistant.

> **Status:** Phase 1 (database + SQL KPIs) complete. ML phases in progress.

## Business questions
1. How are revenue, order volume and basket size trending?
2. How reliable is delivery, and how much do late deliveries hurt satisfaction?
3. How many customers come back, and how does retention decay by cohort?
4. Which categories, states and sellers drive revenue?

## Architecture (Phase 1)
```
Olist CSVs (9 files) ──► load_data.py ──► PostgreSQL
                         (clean, dedupe,     ├── 9 normalized tables (PK/FK, indexes)
                          COPY bulk-load)    ├── v_orders_enriched   (1 row / order)
                                             └── v_customer_summary  (1 row / customer)
                                                        │
                                   kpi_queries.sql ◄────┘
                                   (10 KPI queries) ──► run_kpis.py ──► outputs/*.csv
```

## Project structure
```
customer-intelligence/
├── data/raw/              # Olist CSVs go here (not committed)
├── sql/
│   ├── schema.sql         # tables, keys, indexes
│   ├── views.sql          # reusable analysis views
│   └── kpi_queries.sql    # 10 business KPI queries
├── src/
│   ├── db.py              # database connection
│   ├── load_data.py       # CSV → PostgreSQL loader
│   └── run_kpis.py        # runs KPI queries, saves CSVs
├── outputs/               # query results (generated)
├── docker-compose.yml     # optional local PostgreSQL
└── requirements.txt
```

## Setup

**1. Get the data.** Download the
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
and unzip all 9 CSV files into `data/raw/`.

**2. Start PostgreSQL** (pick one):
- Docker: `docker compose up -d`
- Local install: create a database named `olist`
- Hosted (free): create a project on [Neon](https://neon.tech) or [Supabase](https://supabase.com) and copy its connection string

**3. Configure and install.**
```bash
cp .env.example .env          # then edit DATABASE_URL if needed
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**4. Load the data and run the KPIs.**
```bash
python src/load_data.py       # ~1 minute: creates schema, loads 9 tables, builds views
python src/run_kpis.py        # runs all 10 queries → outputs/*.csv
python src/run_kpis.py 05     # run a single query (cohort retention)
```

## KPI queries
| # | Query | SQL techniques |
|---|---|---|
| 01 | Business summary: orders, customers, revenue, AOV, late % | Aggregation over a view |
| 02 | Monthly revenue with MoM growth and running total | CTE, `LAG`, cumulative `SUM() OVER` |
| 03 | On-time delivery rate and delivery days by month | Date arithmetic, conditional averages |
| 04 | Late delivery vs. review score | `CASE` bucketing |
| 05 | Monthly cohort retention (months 0–12) | Multi-CTE, `AGE`, cohort analysis |
| 06 | Top 15 categories with revenue share and Pareto curve | `RANK`, `SUM() OVER ()`, `LEFT JOIN` + `COALESCE` |
| 07 | Revenue, AOV and delivery experience by state | Grouped KPIs |
| 08 | Top 20 sellers with review score and late % | `HAVING` threshold, multi-table join |
| 09 | Payment method mix and installments | Window share of total |
| 10 | One-time vs. repeat customers | Customer-level view, share of revenue |

## Data notes
- **`customer_id` is per order; `customer_unique_id` is the real customer.** All
  customer-level metrics use `customer_unique_id`.
- **Revenue** = item price + freight on delivered orders.
- **Analysis window** is Jan 2017 to Aug 2018; data outside it is too sparse.
- `review_id` is not unique in the raw file, so reviews are keyed on
  `(review_id, order_id)`.
- A few product categories have no English translation; queries fall back to the
  Portuguese name.

## Key findings
_Fill in after running the queries on the full dataset, for example:_
- Repeat customer rate: __%
- Average review score drops from __ (on time) to __ (7+ days late)
- Top 3 categories account for __% of revenue

## Roadmap
- [x] Phase 1: PostgreSQL data model + SQL KPIs
- [ ] Phase 2: RFM segmentation with K-Means
- [ ] Phase 3: Churn model (XGBoost) + SHAP explainability
- [ ] Phase 4: Category revenue forecasting (Prophet)
- [ ] Phase 5: Review sentiment (multilingual transformer)
- [ ] Phase 6: LLM text-to-SQL assistant
- [ ] Phase 7: Streamlit dashboard + deployment
