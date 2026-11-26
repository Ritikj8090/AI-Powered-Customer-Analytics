"""Phase 5: sentiment analysis of customer review text.

Olist reviews are written in Portuguese, so this uses a multilingual BERT
model fine-tuned on product reviews
(nlptown/bert-base-multilingual-uncased-sentiment), which predicts 1-5 stars.

Steps
  1. Score every review that has a comment.
  2. Validate the model against the stars customers actually gave.
  3. Tag complaint themes (late delivery, wrong item, damaged product ...)
     with a Portuguese keyword dictionary.
  4. Link sentiment to churn: do customers who write negative comments
     come back less often?

Results go to the `review_sentiment` table and outputs/.

Usage
    python src/sentiment.py                 # all reviews with comments (~40K)
    python src/sentiment.py --limit 5000    # quick run on a sample
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sqlalchemy import text

from db import copy_insert, get_engine, save_metrics

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# Keywords are matched after lower-casing and removing accents.
THEMES = {
    # specific phrases only: plain "entrega"/"prazo" also appear in praise
    # ("entrega rapida", "chegou antes do prazo")
    "Late / missing delivery": ["atras", "nao chegou", "nao recebi", "ainda nao",
                                "fora do prazo", "nao foi entregue", "nao entreg",
                                "demor", "aguardando", "extravi"],
    "Wrong or incomplete item": ["errado", "diferente", "faltou", "falta", "incomplet",
                                 "so recebi", "veio outro", "apenas um", "trocado"],
    "Damaged / defective":      ["quebrad", "defeito", "danificad", "estragad",
                                 "nao funciona", "rasgad", "amassad"],
    "Product quality":          ["qualidade", "pessim", "ruim", "fragil", "frágil",
                                 "material", "fraco"],
    "Seller / support":         ["vendedor", "atendimento", "contato", "resposta",
                                 "reclam", "devolu", "reembols", "estorno"],
}


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_reviews(engine, limit: int | None) -> pd.DataFrame:
    sql = """
        SELECT r.review_id, r.order_id, r.review_score,
               CONCAT_WS(' ', NULLIF(TRIM(r.review_comment_title), ''),
                              NULLIF(TRIM(r.review_comment_message), '')) AS review_text,
               c.customer_unique_id, e.is_late
        FROM order_reviews r
        JOIN orders o    ON o.order_id = r.order_id
        JOIN customers c ON c.customer_id = o.customer_id
        JOIN v_orders_enriched e ON e.order_id = r.order_id
        WHERE COALESCE(TRIM(r.review_comment_message), '') <> ''
           OR COALESCE(TRIM(r.review_comment_title), '') <> ''
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    if limit and len(df) > limit:
        # stratified sample so every star rating stays represented
        df = (df.groupby("review_score", group_keys=False)
                .apply(lambda g: g.sample(max(1, int(limit * len(g) / len(df))),
                                          random_state=42)))
    print(f"Reviews with text: {len(df):,}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def load_classifier():
    """Download (first run only, ~650 MB) and load the transformer."""
    import torch
    from transformers import pipeline

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():   # Apple Silicon GPU
        device = "mps"
    else:
        device = -1
    print(f"Loading {MODEL_NAME} on {device if device != -1 else 'cpu'} ...")
    return pipeline("sentiment-analysis", model=MODEL_NAME, device=device,
                    truncation=True, max_length=256, top_k=None)


def score_texts(texts: list[str], classifier, batch_size: int = 32) -> pd.DataFrame:
    """Predicted stars (argmax) and expected stars (probability-weighted)."""
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for preds in classifier(batch, batch_size=batch_size):
            probs = {int(p["label"][0]): p["score"] for p in preds}  # "4 stars" -> 4
            stars = np.array([probs.get(s, 0.0) for s in range(1, 6)])
            rows.append({"predicted_stars": int(stars.argmax() + 1),
                         "expected_stars": float((stars * np.arange(1, 6)).sum()),
                         "confidence": float(stars.max())})
        done = min(start + batch_size, len(texts))
        if done % (batch_size * 50) == 0 or done == len(texts):
            print(f"  scored {done:,}/{len(texts):,}")
    return pd.DataFrame(rows)


def to_label(stars: pd.Series) -> pd.Series:
    return pd.cut(stars, bins=[0, 2, 3, 5], labels=["negative", "neutral", "positive"]
                  ).astype(str)


# ---------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def tag_themes(texts: pd.Series) -> pd.DataFrame:
    norm = texts.map(normalize)
    out = {}
    for theme, words in THEMES.items():
        pattern = "|".join(re.escape(normalize(w)) for w in words)
        out[theme] = norm.str.contains(pattern, regex=True).astype(int)
    return pd.DataFrame(out, index=texts.index)


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------
def validate(df: pd.DataFrame) -> dict:
    true_label, pred_label = to_label(df["review_score"]), df["sentiment_label"]
    exact = accuracy_score(df["review_score"], df["predicted_stars"])
    within_one = float((df["review_score"] - df["predicted_stars"]).abs().le(1).mean())
    label_acc = accuracy_score(true_label, pred_label)
    corr = float(np.corrcoef(df["review_score"], df["expected_stars"])[0, 1])
    neg_true = true_label == "negative"
    neg_pred = pred_label == "negative"
    recall = float((neg_true & neg_pred).sum() / max(neg_true.sum(), 1))
    precision = float((neg_true & neg_pred).sum() / max(neg_pred.sum(), 1))

    labels = ["negative", "neutral", "positive"]
    cm = pd.DataFrame(confusion_matrix(true_label, pred_label, labels=labels),
                      index=[f"true {l}" for l in labels],
                      columns=[f"pred {l}" for l in labels])
    cm.to_csv(OUT_DIR / "sentiment_confusion_matrix.csv")
    return {"reviews_scored": len(df),
            "label_accuracy": round(label_acc, 4),
            "exact_star_accuracy": round(exact, 4),
            "within_one_star": round(within_one, 4),
            "correlation_with_stars": round(corr, 4),
            "negative_precision": round(precision, 4),
            "negative_recall": round(recall, 4)}


def theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    neg = df[df["sentiment_label"] == "negative"]
    rows = []
    for theme in THEMES:
        col = df[theme]
        rows.append({"theme": theme,
                     "share_of_negative_reviews_pct": round(100 * neg[theme].mean(), 1),
                     "avg_stars_when_mentioned": round(df.loc[col == 1, "review_score"].mean(), 2),
                     "avg_stars_otherwise": round(df.loc[col == 0, "review_score"].mean(), 2),
                     "reviews": int(col.sum())})
    return pd.DataFrame(rows).sort_values("share_of_negative_reviews_pct", ascending=False)


def sentiment_vs_return(df: pd.DataFrame, engine) -> pd.DataFrame | None:
    """Repeat-purchase rate by the sentiment of the customer's FIRST review."""
    with engine.connect() as conn:
        cust = pd.read_sql(text("SELECT customer_unique_id, frequency "
                                "FROM v_customer_summary"), conn)
    first = (df.merge(cust, on="customer_unique_id")
               .sort_values("review_id").drop_duplicates("customer_unique_id"))
    if first.empty:
        return None
    return (first.groupby("sentiment_label")
                 .agg(customers=("customer_unique_id", "count"),
                      repeat_rate_pct=("frequency", lambda f: round(100 * (f > 1).mean(), 2)))
                 .reindex(["negative", "neutral", "positive"]).dropna())


def plot_results(df: pd.DataFrame, themes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ct = pd.crosstab(df["review_score"], df["sentiment_label"], normalize="index") * 100
    ct = ct.reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    ct.plot.bar(stacked=True, ax=axes[0],
                color=["tab:red", "tab:gray", "tab:green"], rot=0)
    axes[0].set(title="Model sentiment vs. stars the customer gave",
                xlabel="Customer's star rating", ylabel="% of reviews")
    axes[0].legend(title="Model says", fontsize=8)

    t = themes.sort_values("share_of_negative_reviews_pct")
    axes[1].barh(t["theme"], t["share_of_negative_reviews_pct"], color="tab:red")
    axes[1].set(title="What negative reviews complain about",
                xlabel="% of negative reviews mentioning the theme")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sentiment_overview.png", dpi=150)
    plt.close(fig)


def run(limit: int | None = None, classifier=None) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    print("1/5  Loading reviews ...")
    df = load_reviews(engine, limit)

    print("2/5  Scoring sentiment ...")
    classifier = classifier or load_classifier()
    scores = score_texts(df["review_text"].tolist(), classifier)
    df = pd.concat([df, scores], axis=1)
    df["sentiment_label"] = to_label(df["predicted_stars"])

    print("3/5  Validating against star ratings ...")
    metrics = validate(df)

    print("4/5  Tagging complaint themes ...")
    df = pd.concat([df, tag_themes(df["review_text"])], axis=1)
    themes = theme_summary(df)
    themes.to_csv(OUT_DIR / "sentiment_themes.csv", index=False)
    late = df.groupby("is_late")["sentiment_label"].apply(
        lambda s: round(100 * (s == "negative").mean(), 1))
    metrics["negative_pct_on_time"] = late.get(0)
    metrics["negative_pct_late"] = late.get(1)
    ret = sentiment_vs_return(df, engine)
    if ret is not None:
        ret.to_csv(OUT_DIR / "sentiment_vs_repeat.csv")
        metrics["repeat_rate_by_first_review_sentiment"] = ret["repeat_rate_pct"].to_dict()
    plot_results(df, themes)

    print("5/5  Saving review_sentiment table ...")
    theme_cols = {t: "theme_" + re.sub(r"\W+", "_", t.lower()).strip("_") for t in THEMES}
    out = df.rename(columns=theme_cols)[
        ["review_id", "order_id", "review_score", "predicted_stars", "expected_stars",
         "confidence", "sentiment_label", *theme_cols.values()]]
    out["expected_stars"] = out["expected_stars"].round(3)
    out["confidence"] = out["confidence"].round(3)
    out.to_sql("review_sentiment", engine, if_exists="replace", index=False,
               method=copy_insert, chunksize=50_000)
    (OUT_DIR / "sentiment_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    save_metrics(engine, "sentiment", metrics)

    print("\n=== Sentiment model vs. customer star ratings ===")
    for k, v in metrics.items():
        print(f"  {k:<40} {v}")
    print("\nComplaint themes in negative reviews:\n", themes.to_string(index=False))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="score a stratified sample of N reviews")
    run(parser.parse_args().limit)
