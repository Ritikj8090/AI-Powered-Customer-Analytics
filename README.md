# Customer Intelligence Platform

End-to-end analytics on 100K+ real e-commerce orders from Olist, Brazil's largest
online marketplace: SQL KPIs, customer segmentation, churn prediction, revenue
forecasting, review sentiment, and an LLM-powered text-to-SQL assistant.

> **Status:** Phases 1–3 complete: database + SQL KPIs, customer segmentation, churn and negative-review models.

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
├── notebooks/
│   ├── 02_segmentation.ipynb  # RFM + K-Means walkthrough with charts
│   └── 03_churn_model.ipynb   # XGBoost + SHAP churn and review models
├── src/
│   ├── db.py              # database connection + fast COPY loader
│   ├── load_data.py       # CSV → PostgreSQL loader
│   ├── run_kpis.py        # runs KPI queries, saves CSVs
│   ├── segmentation.py    # RFM scoring + K-Means segmentation
│   └── churn_model.py     # XGBoost + SHAP: churn and negative-review models
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
python src/segmentation.py    # Phase 2: segments → customer_segments table + charts
python src/churn_model.py     # Phase 3: churn model → churn_scores table + SHAP charts
python src/churn_model.py --target negative_review
```

To explore the analysis step by step, run `jupyter notebook` and open
`notebooks/02_segmentation.ipynb`.

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

## Customer segmentation (Phase 2)
Every customer gets two labels, saved to the `customer_segments` table:

- **Rule-based RFM segment** (Champions, Loyal Customers, New Big Spenders, Promising,
  At Risk High Value, Hibernating, Needs Attention). Recency and Monetary are scored
  1–5 by quintile. Frequency is scored on fixed order counts, because nearly every
  Olist customer orders once and quintiles would be meaningless.
- **K-Means cluster** on log-transformed, standardized R/F/M. *k* is the best
  silhouette score between 3 and 6. (k = 2 scores highest but only separates repeat
  from one-time buyers, which is too coarse to act on.) Clusters are named by what
  sets them apart: a cluster is "recent" or "high spend" only when its centre is
  more than 0.5 standard deviations from the average customer.

Outputs: `outputs/segment_profile_kmeans.csv`, `outputs/segment_profile_rfm.csv`,
and charts in `outputs/figures/` (k selection, segment shares, cluster scatter).

## Predictive models (Phase 3)
| Model | Unit | Target | Validation |
|---|---|---|---|
| Churn | Customer | Buys again within 180 days? | Train on snapshot T−360d, test on snapshot T−180d |
| Negative review | Order | Gets a 1–2 star review? | Train on older orders, test on the latest 5 months |

- **XGBoost** with early stopping, compared against a **logistic regression** baseline.
- **No leakage:** churn features use only data known at the snapshot date; deliveries
  and reviews after the snapshot are masked.
- **Positive class is "returns":** about 97% of Olist customers never buy again, so the
  model learns to find the rare returners. Churn risk = 1 − P(return).
- **No class re-weighting,** so predicted probabilities stay realistic.
- **SHAP** (TreeExplainer) explains predictions. Feature importance is also summed into
  driver groups (delivery, reviews, recency, spend, location...) to report what share
  of the model's explanation each business lever accounts for.
- The final churn model is retrained on the latest labelled snapshot and scores every
  customer into the `churn_scores` table (`return_probability`, `churn_probability`,
  `risk_tier`).

Metrics (ROC-AUC, PR-AUC, top-10% capture and lift) are saved to
`outputs/churn_metrics.json` and `outputs/negative_review_metrics.json`.

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
_Fill in after running on the full dataset, for example:_
- Repeat customer rate: __%
- Repeat buyers are __% of customers but __% of revenue
- __% of revenue sits with lapsed high spenders, the main win-back target
- Churn model ROC-AUC __ out-of-time (baseline __); top 10% of scores reach __% of returners
- Delivery features explain __% of negative-review predictions
- Average review score drops from __ (on time) to __ (7+ days late)
- Top 3 categories account for __% of revenue

## Roadmap
- [x] Phase 1: PostgreSQL data model + SQL KPIs
- [x] Phase 2: RFM segmentation with K-Means
- [x] Phase 3: Churn model (XGBoost) + SHAP explainability
- [ ] Phase 4: Category revenue forecasting (Prophet)
- [ ] Phase 5: Review sentiment (multilingual transformer)
- [ ] Phase 6: LLM text-to-SQL assistant
- [ ] Phase 7: Streamlit dashboard + deployment
