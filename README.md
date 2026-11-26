# Customer Intelligence Platform

End-to-end analytics and machine learning on **100K+ real e-commerce orders** from Olist,
Brazil's largest online marketplace: a PostgreSQL data model, SQL KPIs, customer
segmentation, churn prediction with explainable AI, revenue forecasting, NLP on customer
reviews, and an **LLM assistant that answers business questions in plain English**, all
brought together in an interactive dashboard.

**Live demo:** _add your Streamlit link_ · **Stack:** Python, SQL, PostgreSQL, scikit-learn,
XGBoost, SHAP, Prophet, Hugging Face Transformers, LLMs, Streamlit, Plotly

<!-- Add a dashboard screenshot: docs/dashboard.png -->

## Business questions
1. How are revenue, delivery performance and customer satisfaction trending?
2. Who are our customers, and which segments drive revenue?
3. Which customers are likely to come back, and what drives churn?
4. How much revenue should each category expect next quarter?
5. What do unhappy customers complain about?
6. Can non-technical teams get answers without writing SQL?

## Results
_Fill in after running the pipeline on the full dataset. Each number is saved to the file shown._

| Area | Result | Source |
|---|---|---|
| Data | __ orders, __ customers, 9 relational tables | `src/load_data.py` output |
| Retention | __% of customers buy again; repeat buyers drive __% of revenue | `outputs/10_repeat_customer_rate.csv` |
| Satisfaction | Avg review falls from __ (on time) to __ (7+ days late) | `outputs/04_late_delivery_vs_review.csv` |
| Segmentation | __ segments; __ hold __% of revenue | `outputs/segment_profile_kmeans.csv` |
| Churn model | ROC-AUC __ out-of-time (baseline __); top 10% of scores reach __% of returners | `outputs/churn_metrics.json` |
| Review-risk model | ROC-AUC __; delivery factors explain __% of predictions | `outputs/negative_review_metrics.json` |
| Forecasting | Total revenue WAPE __% vs. __% naive (__% better); MAPE __% | `outputs/forecast_metrics.csv` |
| Sentiment | Model agrees with star ratings on __% of reviews; __% of complaints are about delivery | `outputs/sentiment_metrics.json` |
| AI SQL assistant | __% execution accuracy on 30 benchmark questions (__ model) | `outputs/text_to_sql_metrics.json` |

## Architecture
```
Olist CSVs (9 files)
      │  load_data.py: clean, dedupe, bulk COPY
      ▼
PostgreSQL ── 9 normalized tables + analysis views (v_orders_enriched, v_customer_summary)
      │
      ├── kpi_queries.sql ─────────► 10 KPI queries (cohorts, growth, delivery, Pareto ...)
      ├── segmentation.py ─────────► customer_segments   (RFM + K-Means)
      ├── churn_model.py ──────────► churn_scores        (XGBoost + SHAP)
      ├── forecasting.py ──────────► revenue_forecasts   (Prophet)
      ├── sentiment.py ────────────► review_sentiment    (multilingual BERT)
      └── text_to_sql.py ◄──LLM───► natural-language questions → safe read-only SQL
      │
      ▼
Streamlit dashboard (app/streamlit_app.py) ── 6 tabs incl. "Ask the data" chat
```

## Quick start
```bash
# 1. Data: download https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
#    and unzip the 9 CSVs into data/raw/

# 2. Database: local PostgreSQL (macOS):
brew install postgresql@16 && brew services start postgresql@16 && createdb olist
#    or Docker: docker compose up -d   ·   or a free Neon/Supabase database

# 3. Environment (Python 3.10-3.12 recommended)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set DATABASE_URL, and an LLM provider for Phase 6

# 4. Run everything, then open the dashboard
./run_pipeline.sh
streamlit run app/streamlit_app.py
```

Or run phases one at a time:

| Phase | Command | Creates |
|---|---|---|
| 1. Data + KPIs | `python src/load_data.py` then `python src/run_kpis.py` | tables, views, `outputs/*.csv` |
| 2. Segmentation | `python src/segmentation.py` | `customer_segments` |
| 3. Churn + review models | `python src/churn_model.py` (and `--target negative_review`) | `churn_scores` |
| 4. Forecasting | `python src/forecasting.py` | `revenue_forecasts` |
| 5. Sentiment | `python src/sentiment.py --limit 5000` | `review_sentiment` |
| 6. AI SQL assistant | `python src/text_to_sql.py "your question"` / `--eval` | accuracy report |
| 7. Dashboard | `streamlit run app/streamlit_app.py` | web app |

Each phase also has a notebook in `notebooks/` that walks through the analysis with charts.

## How each phase works

### 1. Data model and SQL KPIs
Nine tables with primary/foreign keys and indexes, loaded with PostgreSQL `COPY`. Two
views hold the business logic in one place: `v_orders_enriched` (one row per order with
revenue, delivery delay and review score) and `v_customer_summary` (one row per customer).
The 10 KPI queries use CTEs, window functions (`LAG`, running totals, `RANK`, share of
total) and cohort analysis.

### 2. Customer segmentation
Every customer gets a **rule-based RFM segment** (Champions, At Risk High Value ...) and a
**K-Means cluster** on log-scaled, standardized R/F/M. *k* is the best silhouette score
between 3 and 6: k = 2 scores highest but only separates repeat from one-time buyers, which
is too coarse to act on. Clusters are named by what sets them apart (centre more than
0.5 SD from the average customer).

### 3. Churn and review-risk models
- **Churn:** will a customer buy again within 180 days? Features use only data known at
  the snapshot date (no leakage). The model trains on one snapshot and is **tested on a
  later one** (out-of-time). Because ~97% of customers never return, the model learns to
  find the rare returners; churn risk = 1 − P(return).
- **Negative review:** will a delivered order get 1–2 stars? A practical early-warning model.
- Both use **XGBoost** with early stopping vs. a **logistic regression baseline**, report
  ROC-AUC, PR-AUC and top-10% lift, and are explained with **SHAP**, including the share of
  importance per business driver (delivery, reviews, spend, recency ...).
- No class re-weighting, so predicted probabilities stay realistic.

### 4. Revenue forecasting
Weekly **Prophet** models for total revenue and the top 8 categories, with a Black Friday
holiday effect. Backtested on the last 12 weeks against a naive baseline; reports MAPE and
WAPE, then forecasts the next 12 weeks with 80% intervals.

### 5. Review sentiment
Reviews are in Portuguese, so a **multilingual BERT** model
(`nlptown/bert-base-multilingual-uncased-sentiment`) predicts 1–5 stars from the text. It
is validated against the stars customers actually gave. A Portuguese keyword dictionary
tags complaint themes, and sentiment is linked to repeat-purchase behaviour.

### 6. AI SQL assistant (text-to-SQL)
An LLM gets the schema, business rules and worked examples, and writes one PostgreSQL query.
- **Safe by design:** only single `SELECT` statements pass validation; queries run inside a
  `READ ONLY` transaction with a 15-second timeout; the deployed app uses a read-only user
  (`sql/readonly_user.sql`).
- **Self-correcting:** database errors are sent back to the LLM to fix (up to 2 retries).
- **Measured, not assumed:** `eval/text_to_sql_questions.json` has 30 questions with
  hand-written correct SQL. A generated query counts as correct when it returns the same
  results (any column order or name, 1% numeric tolerance).
- **Any provider:** Ollama (free, local), Google Gemini (free tier), Anthropic or OpenAI;
  set it in `.env`.

### 7. Dashboard
Streamlit + Plotly, six tabs: Overview (KPIs, cohorts), Segments, Churn risk (with a
downloadable win-back target list), Forecast, Reviews, and **Ask the data** (chat with the
SQL assistant). Tabs whose phase hasn't run show a note instead of failing.

## Deploy the dashboard (free)
1. **Database:** create a free project on [Neon](https://neon.tech). Put its connection
   string in your local `.env` as `DATABASE_URL` and run `./run_pipeline.sh`, so every table
   is built in the cloud database. (Olist fits within Neon's free storage.)
2. **Read-only user:** in Neon's SQL editor, run `sql/readonly_user.sql` (change the password).
3. **Push to GitHub.** `.env`, the CSVs and `outputs/` are git-ignored.
4. **Streamlit Community Cloud** ([share.streamlit.io](https://share.streamlit.io)): New app →
   your repo → main file `app/streamlit_app.py`. It installs the lightweight
   `app/requirements.txt`.
5. **Secrets:** in the app's Settings → Secrets, paste the values from
   `.streamlit/secrets.toml.example`, using the **read-only** user's connection string and
   your LLM key (Gemini's free tier works well here).

## Project structure
```
customer-intelligence/
├── app/
│   ├── streamlit_app.py       # dashboard
│   └── requirements.txt       # lightweight deps for deployment
├── data/raw/                  # Olist CSVs (not committed)
├── eval/text_to_sql_questions.json   # 30-question SQL benchmark
├── notebooks/                 # 02 segmentation · 03 churn · 04 forecasting · 05 sentiment
├── sql/
│   ├── schema.sql · views.sql · kpi_queries.sql
│   └── readonly_user.sql
├── src/
│   ├── db.py                  # connection, fast COPY loader, metrics store
│   ├── load_data.py · run_kpis.py
│   ├── segmentation.py · churn_model.py · forecasting.py · sentiment.py
│   ├── llm.py                 # provider-agnostic LLM client
│   └── text_to_sql.py         # SQL assistant + evaluation
├── outputs/                   # generated CSVs, metrics and charts (not committed)
├── run_pipeline.sh · docker-compose.yml · requirements.txt · .env.example
```

## Data notes
- `customer_id` is created **per order**; `customer_unique_id` identifies the real person.
  Every customer-level metric uses `customer_unique_id`.
- Revenue = item price + freight on **delivered** orders. KPI window: Jan 2017 – Aug 2018
  (data outside it is sparse).
- `review_id` isn't unique in the raw file, so reviews are keyed on `(review_id, order_id)`.
- A few categories have no English translation; queries fall back to the Portuguese name.

## Dataset
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0).
