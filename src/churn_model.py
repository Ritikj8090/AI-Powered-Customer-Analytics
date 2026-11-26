"""Phase 3: churn prediction with XGBoost and SHAP explanations.

Two prediction targets share the same training, evaluation and SHAP code:

  churn            Will a customer buy again within the next 180 days?
                   Customer-level, built from point-in-time snapshots.
  negative_review  Will a delivered order get a 1-2 star review?
                   Order-level; lets support teams step in before the
                   customer posts a bad review and leaves.

Why "will return" instead of "will churn": on Olist ~97% of customers never
buy again, so the model learns to spot the rare customers who DO come back.
Churn risk is simply 1 - P(return). ROC-AUC is the same either way.

Validation is out-of-time: the model trains on an earlier snapshot and is
tested on a later one, exactly how it would be used in production.

Usage
    python src/churn_model.py                         # churn model
    python src/churn_model.py --target negative_review
    python src/churn_model.py --horizon 120           # churn window in days
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from xgboost import XGBClassifier

from db import copy_insert, get_engine

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RANDOM_STATE = 42
TOP_CATEGORIES = 15  # other categories are grouped as "other"


# =====================================================================
# Data
# =====================================================================
ORDERS_SQL = """
WITH main_item AS (           -- the most expensive item decides the category
    SELECT DISTINCT ON (oi.order_id)
           oi.order_id,
           COALESCE(t.product_category_name_english,
                    p.product_category_name, 'unknown') AS category,
           s.seller_state
    FROM order_items oi
    JOIN products p ON p.product_id = oi.product_id
    JOIN sellers  s ON s.seller_id  = oi.seller_id
    LEFT JOIN category_translation t
           ON t.product_category_name = p.product_category_name
    ORDER BY oi.order_id, oi.price DESC
),
sellers_per_order AS (
    SELECT order_id, COUNT(DISTINCT seller_id) AS n_sellers
    FROM order_items GROUP BY order_id
)
SELECT e.order_id, e.customer_unique_id, e.customer_state,
       e.purchase_ts, e.delivered_ts, e.estimated_ts,
       e.order_total::FLOAT       AS order_total,
       e.items_revenue::FLOAT     AS items_revenue,
       e.freight_value::FLOAT     AS freight_value,
       e.n_items, e.max_installments, e.main_payment_type,
       e.review_score::FLOAT      AS review_score,
       e.delivery_days::FLOAT     AS delivery_days,
       e.days_vs_estimate::FLOAT  AS days_vs_estimate,
       e.is_late,
       m.category, m.seller_state, s.n_sellers,
       (EXTRACT(EPOCH FROM o.order_approved_at - o.order_purchase_timestamp)
            / 3600.0)::FLOAT     AS approval_hours,
       (EXTRACT(EPOCH FROM o.order_delivered_carrier_date - o.order_purchase_timestamp)
            / 86400.0)::FLOAT    AS days_to_carrier,
       (EXTRACT(EPOCH FROM o.order_estimated_delivery_date - o.order_purchase_timestamp)
            / 86400.0)::FLOAT    AS promised_days
FROM v_orders_enriched e
JOIN orders o USING (order_id)
LEFT JOIN main_item m USING (order_id)
LEFT JOIN sellers_per_order s USING (order_id)
WHERE e.order_status = 'delivered'
"""


def load_orders(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(ORDERS_SQL), conn,
                         parse_dates=["purchase_ts", "delivered_ts", "estimated_ts"])
    top = df["category"].value_counts().index[:TOP_CATEGORIES]
    df["category"] = df["category"].where(df["category"].isin(top), "other")
    return df


# =====================================================================
# Target 1: churn (customer level, point-in-time snapshots)
# =====================================================================
CHURN_GROUPS = {
    "Delivery experience": ["avg_delivery_days", "late_share", "max_days_late",
                            "last_is_late", "last_delivery_days", "last_days_vs_estimate"],
    "Review scores":       ["avg_review_score", "min_review_score",
                            "any_negative_review", "last_review_score"],
    "Recency & tenure":    ["recency_days", "tenure_days"],
    "Spend & basket":      ["frequency", "monetary", "avg_order_value", "avg_items",
                            "freight_ratio", "last_order_value", "avg_installments",
                            "n_categories"],
    "Payment method":      ["main_payment_type_"],
    "Customer location":   ["customer_state_"],
    "Product category":    ["last_category_"],
}


def customer_snapshot(orders: pd.DataFrame, cutoff: pd.Timestamp,
                      horizon_days: int | None) -> pd.DataFrame:
    """Features using only what was known at `cutoff`; label from the window after.

    Delivery and review facts for an order only count if the order was
    delivered before the cutoff, otherwise they would leak future info.
    """
    hist = orders[orders["purchase_ts"] < cutoff].copy()
    known = hist["delivered_ts"] < cutoff
    for col in ["review_score", "delivery_days", "days_vs_estimate", "is_late"]:
        hist.loc[~known, col] = np.nan
    hist["negative_review"] = (hist["review_score"] <= 2).astype(float)
    hist.loc[hist["review_score"].isna(), "negative_review"] = np.nan
    hist = hist.sort_values("purchase_ts")

    g = hist.groupby("customer_unique_id")
    last = g.tail(1).set_index("customer_unique_id")
    feats = pd.DataFrame({
        "recency_days":      (cutoff - g["purchase_ts"].max()).dt.days,
        "tenure_days":       (cutoff - g["purchase_ts"].min()).dt.days,
        "frequency":         g.size(),
        "monetary":          g["order_total"].sum(),
        "avg_order_value":   g["order_total"].mean(),
        "avg_items":         g["n_items"].mean(),
        "freight_ratio":     g["freight_value"].sum() / g["order_total"].sum(),
        "avg_installments":  g["max_installments"].mean(),
        "n_categories":      g["category"].nunique(),
        "avg_review_score":  g["review_score"].mean(),
        "min_review_score":  g["review_score"].min(),
        "any_negative_review": g["negative_review"].max(),
        "avg_delivery_days": g["delivery_days"].mean(),
        "late_share":        g["is_late"].mean(),
        "max_days_late":     g["days_vs_estimate"].max(),
    })
    feats = feats.join(last[["review_score", "is_late", "delivery_days",
                             "days_vs_estimate", "order_total", "customer_state",
                             "main_payment_type", "category"]]
                       .rename(columns={"review_score": "last_review_score",
                                        "is_late": "last_is_late",
                                        "delivery_days": "last_delivery_days",
                                        "days_vs_estimate": "last_days_vs_estimate",
                                        "order_total": "last_order_value",
                                        "category": "last_category"}))

    if horizon_days is not None:
        end = cutoff + pd.Timedelta(days=horizon_days)
        future = orders[(orders["purchase_ts"] >= cutoff) & (orders["purchase_ts"] < end)]
        feats["target"] = feats.index.isin(future["customer_unique_id"]).astype(int)
    return feats


def churn_datasets(orders: pd.DataFrame, horizon: int):
    data_end = orders["purchase_ts"].max().normalize() + pd.Timedelta(days=1)
    test_cut = data_end - pd.Timedelta(days=horizon)
    train_cut = test_cut - pd.Timedelta(days=horizon)
    print(f"  train snapshot {train_cut.date()} | test snapshot {test_cut.date()} "
          f"| scoring snapshot {data_end.date()} | window {horizon} days")
    train = customer_snapshot(orders, train_cut, horizon)
    test = customer_snapshot(orders, test_cut, horizon)
    current = customer_snapshot(orders, data_end, None)
    return train, test, current


# =====================================================================
# Target 2: negative review (order level)
# =====================================================================
REVIEW_GROUPS = {
    "Delivery experience": ["delivery_days", "days_vs_estimate", "is_late",
                            "days_to_carrier", "promised_days", "approval_hours"],
    "Price & basket":      ["order_total", "freight_ratio", "n_items", "n_sellers",
                            "max_installments"],
    "Location":            ["customer_state_", "seller_state_", "same_state"],
    "Payment method":      ["main_payment_type_"],
    "Product category":    ["category_"],
}


def review_datasets(orders: pd.DataFrame, test_months: int = 5):
    df = orders.dropna(subset=["review_score", "delivered_ts"]).copy()
    df["target"] = (df["review_score"] <= 2).astype(int)
    df["freight_ratio"] = df["freight_value"] / df["order_total"]
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    cols = REVIEW_GROUPS["Delivery experience"] + REVIEW_GROUPS["Price & basket"] + [
        "same_state", "customer_state", "seller_state", "main_payment_type",
        "category", "target"]
    df = df.set_index("order_id")
    split = df["purchase_ts"].max().normalize() - pd.DateOffset(months=test_months)
    print(f"  train: orders before {split.date()} | test: orders from {split.date()}")
    return df.loc[df["purchase_ts"] < split, cols], df.loc[df["purchase_ts"] >= split, cols]


# =====================================================================
# Shared: encoding, training, evaluation, SHAP
# =====================================================================
CATEGORICAL = ["customer_state", "seller_state", "main_payment_type",
               "category", "last_category"]


def encode(train: pd.DataFrame, *others: pd.DataFrame):
    """One-hot encode categoricals; align every frame to train's columns."""
    cats = [c for c in CATEGORICAL if c in train.columns]
    X_train = pd.get_dummies(train.drop(columns="target", errors="ignore"),
                             columns=cats, dtype=float).astype(float)
    out = [X_train]
    for df in others:
        X = pd.get_dummies(df.drop(columns="target", errors="ignore"),
                           columns=cats, dtype=float).astype(float)
        out.append(X.reindex(columns=X_train.columns, fill_value=0.0))
    return out


def train_xgb(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        # No class re-weighting: it would barely change AUC but would push
        # predicted probabilities toward 0.5, making them meaningless.
        eval_metric="auc", early_stopping_rounds=50,
        random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  XGBoost stopped at {model.best_iteration + 1} trees")
    return model


def train_baseline(X: pd.DataFrame, y: pd.Series):
    """Logistic regression, to prove XGBoost is worth its complexity."""
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
    return model.fit(X, y)


def evaluate(y: pd.Series, p: np.ndarray) -> dict:
    top = np.argsort(-p)[: max(1, len(p) // 10)]
    capture = y.iloc[top].sum() / max(y.sum(), 1)
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "positive_rate": round(y.mean(), 4),
        "top10_capture": round(capture, 4),   # share of positives in top 10% scores
        "top10_lift": round(capture / 0.10, 2),
    }


def shap_analysis(model, X: pd.DataFrame, groups: dict, target: str):
    sample = X.sample(min(len(X), 5000), random_state=RANDOM_STATE)
    values = shap.TreeExplainer(model).shap_values(sample)
    importance = pd.Series(np.abs(values).mean(axis=0), index=sample.columns)

    def group_of(col):
        for name, members in groups.items():
            if any(col == m or (m.endswith("_") and col.startswith(m)) for m in members):
                return name
        return "Other"

    grouped = (importance.groupby(importance.index.map(group_of)).sum()
                         .sort_values(ascending=False))
    grouped = (100 * grouped / grouped.sum()).round(1)

    label = "P(return)" if target == "churn" else "P(negative review)"
    plt.figure()
    shap.summary_plot(values, sample, max_display=15, show=False)
    plt.title(f"SHAP: feature impact on {label}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{target}_shap_beeswarm.png", dpi=150)
    plt.close("all")

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(grouped) + 1.5))
    grouped.sort_values().plot.barh(ax=ax)
    ax.set(title=f"Share of model explanation by driver group ({target})",
           xlabel="% of mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{target}_shap_groups.png", dpi=150)
    plt.close(fig)
    return importance.sort_values(ascending=False), grouped


def plot_roc(y, p_xgb, p_base, target):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in [("XGBoost", p_xgb), ("Logistic regression", p_base)]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set(title=f"ROC curve, out-of-time test ({target})",
           xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{target}_roc.png", dpi=150)
    plt.close(fig)


def save_churn_scores(model, X_current, current, engine) -> pd.DataFrame:
    p_return = model.predict_proba(X_current)[:, 1]
    scores = pd.DataFrame({
        "customer_unique_id": current.index,
        "return_probability": p_return.round(4),
        "churn_probability": (1 - p_return).round(4),
    })
    rank = scores["return_probability"].rank(pct=True)
    scores["risk_tier"] = np.select([rank >= 0.9, rank >= 0.6],
                                    ["Likely to return", "Could return"],
                                    "Likely to churn")
    scores.to_sql("churn_scores", engine, if_exists="replace", index=False,
                  method=copy_insert, chunksize=50_000)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE churn_scores ADD PRIMARY KEY (customer_unique_id)"))
    return scores


# =====================================================================
# Main
# =====================================================================
def run(target: str = "churn", horizon: int = 180) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    print("1/5  Loading orders ...")
    orders = load_orders(engine)

    print("2/5  Building datasets ...")
    if target == "churn":
        train, test, current = churn_datasets(orders, horizon)
        X_train, X_test, X_current = encode(train, test, current)
        groups = CHURN_GROUPS
    else:
        train, test = review_datasets(orders)
        X_train, X_test = encode(train, test)
        groups = REVIEW_GROUPS
    y_train, y_test = train["target"], test["target"]
    print(f"  train rows {len(X_train):,} (positive {y_train.mean():.1%}) | "
          f"test rows {len(X_test):,} (positive {y_test.mean():.1%}) | "
          f"features {X_train.shape[1]}")
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise SystemExit("Not enough positive examples to train. Try a longer --horizon.")

    print("3/5  Training XGBoost and baseline ...")
    xgb = train_xgb(X_train, y_train)
    base = train_baseline(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    p_base = base.predict_proba(X_test)[:, 1]
    metrics = {"target": target,
               "xgboost": evaluate(y_test, p_xgb),
               "logistic_regression": evaluate(y_test, p_base),
               "train_rows": len(X_train), "test_rows": len(X_test)}
    if target == "churn":
        metrics["horizon_days"] = horizon
    plot_roc(y_test, p_xgb, p_base, target)

    print("4/5  Explaining with SHAP ...")
    importance, grouped = shap_analysis(xgb, X_test, groups, target)
    metrics["driver_group_share_pct"] = grouped.to_dict()
    metrics["top_features"] = importance.head(10).round(4).to_dict()

    print("5/5  Saving results ...")
    if target == "churn":
        # Final model: retrain on the most recent labelled snapshot, then
        # score every customer as of the end of the data.
        test_X, = encode(test)
        final = train_xgb(test_X, y_test)
        X_now = X_current.reindex(columns=test_X.columns, fill_value=0.0)
        scores = save_churn_scores(final, X_now, current, engine)
        print(f"  churn_scores table: {len(scores):,} customers")
    (OUT_DIR / f"{target}_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n=== Results: {target} (out-of-time test set) ===")
    print(pd.DataFrame({k: metrics[k] for k in ["xgboost", "logistic_regression"]}).to_string())
    print("\nShare of model explanation by driver group (%):")
    print(grouped.to_string())
    print(f"\nSaved outputs/{target}_metrics.json and charts in outputs/figures/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["churn", "negative_review"], default="churn")
    parser.add_argument("--horizon", type=int, default=180,
                        help="churn window in days (churn target only)")
    args = parser.parse_args()
    run(args.target, args.horizon)
