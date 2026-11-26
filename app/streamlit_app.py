"""Customer Intelligence Platform: interactive dashboard.

Run locally:   streamlit run app/streamlit_app.py
Each tab reads tables created by the pipeline phases; a tab whose phase
hasn't been run yet shows a note instead of failing.
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

# On Streamlit Community Cloud, settings come from the app's Secrets.
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:  # no secrets file locally: .env is used instead
    pass

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from sqlalchemy import text  # noqa: E402

from db import get_engine  # noqa: E402
from run_kpis import QUERY_FILE, parse_queries  # noqa: E402

st.set_page_config(page_title="Customer Intelligence Platform",
                   page_icon="📊", layout="wide")


# ---------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------
@st.cache_resource
def engine():
    return get_engine()


@st.cache_data(ttl=3600, show_spinner=False)
def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


@st.cache_data(ttl=3600, show_spinner=False)
def kpi(name: str) -> pd.DataFrame:
    return query(parse_queries(QUERY_FILE)[name])


@st.cache_data(ttl=3600, show_spinner=False)
def table_exists(name: str) -> bool:
    return bool(query("SELECT to_regclass(:n) IS NOT NULL AS ok", {"n": name})["ok"].iloc[0])


def metrics(name: str):
    if not table_exists("model_metrics"):
        return None
    df = query("SELECT metrics FROM model_metrics WHERE name = :n", {"n": name})
    return df["metrics"].iloc[0] if len(df) else None


def missing(phase: str, command: str) -> None:
    st.info(f"Run {phase} to fill this tab:\n\n`{command}`")


def money(x) -> str:
    return f"R$ {x:,.0f}"


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Customer Intelligence Platform")
st.caption("Olist Brazilian e-commerce, 100K+ orders (2017–2018) · SQL analytics · "
           "segmentation · churn prediction · forecasting · NLP · AI SQL assistant")

try:
    summary = kpi("01_business_summary").iloc[0]
except Exception as exc:
    st.error(f"Could not connect to the database. Check DATABASE_URL.\n\n{exc}")
    st.stop()

c = st.columns(6)
c[0].metric("Revenue", money(summary["total_revenue"]))
c[1].metric("Delivered orders", f"{summary['delivered_orders']:,.0f}")
c[2].metric("Customers", f"{summary['customers']:,.0f}")
c[3].metric("Avg order value", money(summary["avg_order_value"]))
c[4].metric("Avg review", f"{summary['avg_review_score']:.2f} ★")
c[5].metric("Late deliveries", f"{summary['late_delivery_pct']:.1f}%")

tabs = st.tabs(["📈 Overview", "👥 Segments", "⚠️ Churn risk", "🔮 Forecast",
                "💬 Reviews", "🤖 Ask the data"])

# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
with tabs[0]:
    monthly = kpi("02_monthly_revenue_growth")
    left, right = st.columns(2)
    left.plotly_chart(px.line(monthly, x="order_month", y="revenue", markers=True,
                              title="Monthly revenue"), width="stretch")
    right.plotly_chart(px.bar(monthly, x="order_month", y="avg_order_value",
                              title="Average order value"), width="stretch")

    left, right = st.columns(2)
    delivery = kpi("03_delivery_performance_monthly")
    left.plotly_chart(px.line(delivery, x="order_month", y="on_time_pct", markers=True,
                              title="On-time delivery rate (%)"), width="stretch")
    late = kpi("04_late_delivery_vs_review")
    right.plotly_chart(px.bar(late, x="delivery_bucket", y="avg_review_score",
                              text="avg_review_score",
                              title="Late deliveries destroy review scores"),
                       width="stretch")

    left, right = st.columns(2)
    cats = kpi("06_revenue_by_category").sort_values("revenue")
    left.plotly_chart(px.bar(cats, x="revenue", y="category", orientation="h",
                             title="Top categories by revenue", height=500),
                      width="stretch")
    states = kpi("07_revenue_by_state")
    right.plotly_chart(px.scatter(states, x="avg_delivery_days", y="avg_review_score",
                                  size="revenue", text="customer_state", height=500,
                                  title="By state: slower delivery, lower reviews"),
                       width="stretch")

    cohort = kpi("05_cohort_retention")
    pivot = (cohort[cohort["months_since_first"].between(1, 12)]
             .pivot(index="cohort_month", columns="months_since_first", values="retention_pct"))
    st.plotly_chart(px.imshow(pivot, text_auto=".1f", aspect="auto",
                              color_continuous_scale="Blues",
                              labels={"x": "Months after first purchase", "y": "Cohort",
                                      "color": "Retention %"},
                              title="Cohort retention: % of customers buying again"),
                    width="stretch")

# ---------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------
with tabs[1]:
    if not table_exists("customer_segments"):
        missing("Phase 2", "python src/segmentation.py")
    else:
        prof = query("""
            SELECT cluster_name AS segment, COUNT(*) AS customers,
                   ROUND(AVG(recency_days)) AS avg_recency_days,
                   ROUND(AVG(frequency)::NUMERIC, 2) AS avg_orders,
                   ROUND(AVG(monetary)::NUMERIC, 2) AS avg_spend,
                   ROUND(SUM(monetary)::NUMERIC, 2) AS revenue
            FROM customer_segments GROUP BY 1 ORDER BY revenue DESC""")
        prof["% customers"] = (100 * prof["customers"] / prof["customers"].sum()).round(1)
        prof["% revenue"] = (100 * prof["revenue"] / prof["revenue"].sum()).round(1)

        left, right = st.columns([3, 2])
        shares = prof.melt(id_vars="segment", value_vars=["% customers", "% revenue"])
        left.plotly_chart(px.bar(shares, x="value", y="segment", color="variable",
                                 barmode="group", orientation="h",
                                 title="K-Means segments: share of customers vs. revenue"),
                          width="stretch")
        rfm = query("SELECT rfm_segment AS segment, COUNT(*) AS customers "
                    "FROM customer_segments GROUP BY 1 ORDER BY 2 DESC")
        right.plotly_chart(px.pie(rfm, names="segment", values="customers", hole=0.45,
                                  title="Rule-based RFM segments"),
                           width="stretch")
        st.dataframe(prof, hide_index=True, width="stretch")

        sample = query("""SELECT recency_days, monetary, cluster_name AS segment
                          FROM customer_segments ORDER BY md5(customer_unique_id) LIMIT 4000""")
        st.plotly_chart(px.scatter(sample, x="recency_days", y="monetary", color="segment",
                                   log_y=True, opacity=0.6,
                                   title="Recency vs. lifetime spend (4,000-customer sample)"),
                        width="stretch")

# ---------------------------------------------------------------------
# Churn risk
# ---------------------------------------------------------------------
with tabs[2]:
    if not table_exists("churn_scores"):
        missing("Phase 3", "python src/churn_model.py")
    else:
        m = metrics("churn")
        if m:
            x = m["xgboost"]
            c = st.columns(4)
            c[0].metric("ROC-AUC (out-of-time)", f"{x['roc_auc']:.3f}",
                        f"{x['roc_auc'] - m['logistic_regression']['roc_auc']:+.3f} vs. baseline")
            c[1].metric("PR-AUC", f"{x['pr_auc']:.3f}",
                        f"random = {x['positive_rate']:.3f}", delta_color="off")
            c[2].metric("Top-10% capture", f"{100 * x['top10_capture']:.0f}%",
                        help="Share of all returning customers found in the 10% highest scores")
            c[3].metric("Lift", f"{x['top10_lift']:.1f}x")
            groups = pd.Series(m["driver_group_share_pct"]).sort_values()
            st.plotly_chart(px.bar(groups, orientation="h",
                                   labels={"value": "% of SHAP importance", "index": ""},
                                   title="What drives the churn model (SHAP, by driver group)"),
                            width="stretch")

        tiers = query("""SELECT risk_tier, COUNT(*) AS customers,
                                ROUND(AVG(return_probability)::NUMERIC * 100, 2) AS avg_return_pct
                         FROM churn_scores GROUP BY 1 ORDER BY 3 DESC""")
        st.dataframe(tiers, hide_index=True, width="stretch")

        if table_exists("customer_segments"):
            st.subheader("Win-back target list")
            st.caption("High-value customers the model thinks can still be won back.")
            n = st.slider("Customers to show", 10, 200, 25, step=5)
            targets = query("""
                SELECT s.customer_unique_id, s.cluster_name AS segment, s.customer_state AS state,
                       s.recency_days, s.frequency AS orders,
                       ROUND(s.monetary::NUMERIC, 2) AS lifetime_spend,
                       ROUND(c.return_probability::NUMERIC * 100, 2) AS return_prob_pct
                FROM churn_scores c JOIN customer_segments s USING (customer_unique_id)
                WHERE c.risk_tier <> 'Likely to churn'
                ORDER BY c.return_probability * s.monetary DESC
                LIMIT :n""", {"n": n})
            st.dataframe(targets, hide_index=True, width="stretch")
            st.download_button("Download list (CSV)", targets.to_csv(index=False),
                               "win_back_targets.csv", "text/csv")

    review_m = metrics("negative_review")
    if review_m:
        st.subheader("Negative-review early warning model")
        c = st.columns(3)
        c[0].metric("ROC-AUC", f"{review_m['xgboost']['roc_auc']:.3f}")
        c[1].metric("PR-AUC", f"{review_m['xgboost']['pr_auc']:.3f}")
        top = max(review_m["driver_group_share_pct"], key=review_m["driver_group_share_pct"].get)
        c[2].metric("Top driver", top,
                    f"{review_m['driver_group_share_pct'][top]:.0f}% of importance",
                    delta_color="off")

# ---------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------
with tabs[3]:
    if not table_exists("revenue_forecasts"):
        missing("Phase 4", "python src/forecasting.py")
    else:
        series = query("SELECT DISTINCT category FROM revenue_forecasts ORDER BY 1")["category"]
        choice = st.selectbox("Category", series,
                              index=int((series == "ALL CATEGORIES").idxmax()))
        fc = query("SELECT * FROM revenue_forecasts WHERE category = :c ORDER BY week",
                   {"c": choice})
        fut = fc[fc["is_forecast"]]
        fig = go.Figure([
            go.Scatter(x=fc["week"], y=fc["actual"], name="Actual", line=dict(color="black")),
            go.Scatter(x=fut["week"], y=fut["forecast_upper"], line=dict(width=0),
                       showlegend=False, hoverinfo="skip"),
            go.Scatter(x=fut["week"], y=fut["forecast_lower"], fill="tonexty", line=dict(width=0),
                       fillcolor="rgba(255,127,14,0.25)", name="80% interval"),
            go.Scatter(x=fut["week"], y=fut["forecast"], name="Forecast",
                       line=dict(color="#ff7f0e", width=3)),
        ])
        fig.update_layout(title=f"Weekly revenue: {choice}", yaxis_title="Revenue (R$)")
        st.plotly_chart(fig, width="stretch")

        m = metrics("forecast")
        if m:
            mdf = pd.DataFrame(m).set_index("series")
            row = mdf.loc[choice]
            c = st.columns(3)
            c[0].metric("Backtest MAPE", f"{row['prophet_mape']:.1f}%")
            c[1].metric("Backtest WAPE", f"{row['prophet_wape']:.1f}%",
                        f"{row['naive_wape'] - row['prophet_wape']:+.1f} pts vs. naive",
                        delta_color="inverse" if row["naive_wape"] < row["prophet_wape"] else "normal")
            c[2].metric("Next 12 weeks (forecast)", money(fut["forecast"].sum()))
            with st.expander("Accuracy for every series (last 12 weeks held out)"):
                st.dataframe(mdf, width="stretch")

# ---------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------
with tabs[4]:
    if not table_exists("review_sentiment"):
        missing("Phase 5", "python src/sentiment.py --limit 5000")
    else:
        m = metrics("sentiment")
        if m:
            c = st.columns(4)
            c[0].metric("Reviews analysed", f"{m['reviews_scored']:,}")
            c[1].metric("Agrees with stars (3-class)", f"{100 * m['label_accuracy']:.0f}%")
            c[2].metric("Negative when on time", f"{m['negative_pct_on_time']}%")
            c[3].metric("Negative when late", f"{m['negative_pct_late']}%")

        theme_cols = [c for c in query("SELECT * FROM review_sentiment LIMIT 0").columns
                      if c.startswith("theme_")]
        sums = ", ".join(f"ROUND(100.0 * AVG({c}), 1) AS {c}" for c in theme_cols)
        themes = query(f"SELECT {sums} FROM review_sentiment "
                       "WHERE sentiment_label = 'negative'").T.reset_index()
        themes.columns = ["theme", "pct"]
        themes["theme"] = themes["theme"].str.replace("theme_", "").str.replace("_", " ").str.title()
        left, right = st.columns(2)
        left.plotly_chart(px.bar(themes.sort_values("pct"), x="pct", y="theme", orientation="h",
                                 labels={"pct": "% of negative reviews"},
                                 title="What negative reviews complain about"),
                          width="stretch")
        dist = query("""SELECT review_score AS stars, sentiment_label AS model, COUNT(*) AS n
                        FROM review_sentiment GROUP BY 1, 2""")
        right.plotly_chart(px.bar(dist, x="stars", y="n", color="model", barmode="stack",
                                  color_discrete_map={"negative": "#d62728", "neutral": "#7f7f7f",
                                                      "positive": "#2ca02c"},
                                  title="Model sentiment by customer star rating"),
                           width="stretch")

        theme_pick = st.selectbox("Read negative reviews about", theme_cols,
                                  format_func=lambda c: c.replace("theme_", "").replace("_", " ").title())
        examples = query(f"""
            SELECT r.review_score AS stars, r.review_comment_message AS review
            FROM review_sentiment s JOIN order_reviews r USING (review_id, order_id)
            WHERE s.sentiment_label = 'negative' AND s.{theme_pick} = 1
            ORDER BY md5(s.review_id) LIMIT 10""")
        st.dataframe(examples, hide_index=True, width="stretch")
        st.caption("Reviews are in Portuguese, as written by customers.")

# ---------------------------------------------------------------------
# Ask the data
# ---------------------------------------------------------------------
with tabs[5]:
    st.write("Ask a business question in plain English. An LLM writes the SQL, "
             "which runs read-only against the database.")
    m = metrics("text_to_sql")
    if m:
        st.caption(f"Measured accuracy: {100 * m['execution_accuracy']:.0f}% on "
                   f"{m['questions']} benchmark questions ({m['model']}).")

    try:
        import llm
        llm.provider()
        ready = True
    except Exception as exc:
        ready = False
        st.info(f"The assistant isn't configured: {exc}")

    if ready:
        @st.cache_resource
        def assistant():
            from text_to_sql import SQLAssistant
            return SQLAssistant(engine())

        examples = ["Which 5 states have the highest average order value?",
                    "What share of delivered orders were late in each month of 2018?",
                    "Which categories have the worst average review score?"]
        cols = st.columns(len(examples))
        clicked = next((q for col, q in zip(cols, examples)
                        if col.button(q, width="stretch")), None)
        question = st.chat_input("e.g. How did revenue change month over month in 2018?") or clicked

        if "history" not in st.session_state:
            st.session_state.history = []
        if question:
            with st.spinner("Writing and running SQL ..."):
                out = assistant().ask(question)
                summary_text = None
                if out["result"] is not None and len(out["result"]):
                    try:
                        summary_text = assistant().summarize(question, out["result"])
                    except Exception:
                        pass
            st.session_state.history.insert(0, {"q": question, **out, "summary": summary_text})

        for item in st.session_state.history[:5]:
            with st.chat_message("user"):
                st.write(item["q"])
            with st.chat_message("assistant"):
                if item["error"]:
                    st.error(f"Couldn't answer: {item['error']}")
                else:
                    if item["summary"]:
                        st.write(item["summary"])
                    st.dataframe(item["result"], hide_index=True, width="stretch")
                if item["sql"]:
                    with st.expander("SQL"):
                        st.code(item["sql"], language="sql")

st.divider()
st.caption("Built with PostgreSQL, scikit-learn, XGBoost, SHAP, Prophet, Hugging Face "
           "Transformers, an LLM text-to-SQL agent and Streamlit.")
