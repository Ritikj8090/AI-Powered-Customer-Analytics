"""Phase 2: RFM customer segmentation with K-Means.

Two complementary views of every customer:
  1. Rule-based RFM scores (1-5) and named segments that a marketing team
     can read at a glance ("Champions", "At Risk High Value", ...).
  2. K-Means clusters on log-scaled RFM features, with k chosen from
     the elbow curve and silhouette score.

Results are written to the `customer_segments` table (used by the churn
model and the dashboard) and charts are saved to outputs/figures/.

Usage
    python src/segmentation.py            # pick k automatically (3-6)
    python src/segmentation.py --k 5      # force a number of clusters
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from db import copy_insert, get_engine

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. Build RFM table
# ---------------------------------------------------------------------
def load_rfm(engine) -> pd.DataFrame:
    """One row per real customer with Recency, Frequency, Monetary.

    Recency is measured from a snapshot date one day after the last
    purchase in the data, as if we were running the analysis that day.
    """
    sql = """
        SELECT customer_unique_id,
               customer_state,
               frequency,
               monetary::FLOAT          AS monetary,
               last_purchase_ts,
               avg_review_score::FLOAT  AS avg_review_score,
               late_share::FLOAT        AS late_share
        FROM v_customer_summary
        WHERE monetary IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, parse_dates=["last_purchase_ts"])

    snapshot = df["last_purchase_ts"].max().normalize() + pd.Timedelta(days=1)
    df["recency_days"] = (snapshot - df["last_purchase_ts"]).dt.days
    print(f"Customers: {len(df):,}   snapshot date: {snapshot.date()}")
    return df


# ---------------------------------------------------------------------
# 2. Rule-based RFM scores
# ---------------------------------------------------------------------
def score_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Score R and M by quintile (1 = worst, 5 = best).

    Frequency cannot be split into quintiles on Olist because almost every
    customer has exactly one order, so F is scored on fixed order counts.
    """
    df = df.copy()
    # rank(method="first") breaks ties so qcut always gets 5 equal bins
    df["r_score"] = pd.qcut(df["recency_days"].rank(method="first"), 5,
                            labels=[5, 4, 3, 2, 1]).astype(int)
    df["m_score"] = pd.qcut(df["monetary"].rank(method="first"), 5,
                            labels=[1, 2, 3, 4, 5]).astype(int)
    df["f_score"] = df["frequency"].clip(upper=5).astype(int)
    df["rfm_code"] = (df["r_score"].astype(str) + df["f_score"].astype(str)
                      + df["m_score"].astype(str))
    df["rfm_segment"] = df.apply(_rfm_segment, axis=1)
    return df


def _rfm_segment(row) -> str:
    r, f, m = row["r_score"], row["f_score"], row["m_score"]
    if f >= 2 and r >= 4:
        return "Champions"
    if f >= 2:
        return "Loyal Customers"
    if r >= 4 and m >= 4:
        return "New Big Spenders"
    if r >= 4:
        return "Promising"
    if r <= 2 and m >= 4:
        return "At Risk High Value"
    if r <= 2:
        return "Hibernating"
    return "Needs Attention"


# ---------------------------------------------------------------------
# 3. K-Means clustering
# ---------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> np.ndarray:
    """Log-transform (RFM values are heavily right-skewed), then scale."""
    X = np.log1p(df[["recency_days", "frequency", "monetary"]].to_numpy(float))
    return StandardScaler().fit_transform(X)


def evaluate_k(X: np.ndarray, k_values=range(2, 9)) -> pd.DataFrame:
    """Inertia (elbow) and silhouette score for each k."""
    rows = []
    sample = min(len(X), 10_000)  # silhouette is O(n^2); sample for speed
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=sample,
                               random_state=RANDOM_STATE)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  k={k}  inertia={km.inertia_:>12,.0f}  silhouette={sil:.3f}")
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame, k_min=3, k_max=6) -> int:
    """Best silhouette within a range the business can act on.

    k=2 often scores highest but just splits "recent" from "old", which
    is too coarse for marketing. More than 6 segments is hard to staff.
    """
    window = scores[(scores.k >= k_min) & (scores.k <= k_max)]
    return int(window.loc[window.silhouette.idxmax(), "k"])


def name_clusters(df: pd.DataFrame, X: np.ndarray, cut: float = 0.5) -> dict[int, str]:
    """Give each cluster a business name from what makes it different.

    Uses each cluster's centre in the scaled feature space: a cluster is
    "recent" or "lapsed" only if its average recency is more than `cut`
    standard deviations from the average customer, and likewise for spend.
    A cluster whose customers average 1.5+ orders is a repeat-buyer group.
    """
    centres = (pd.DataFrame(X, columns=["r_z", "f_z", "m_z"])
                 .assign(cluster=df["cluster"].to_numpy())
                 .groupby("cluster").mean())
    avg_freq = df.groupby("cluster")["frequency"].mean()

    def label(r_z, m_z, freq):
        if freq >= 1.5:
            return "Loyal Repeat Buyers" if r_z <= 0 else "Lapsing Repeat Buyers"
        rec = "recent" if r_z < -cut else "lapsed" if r_z > cut else "mid"
        spend = "high" if m_z > cut else "low" if m_z < -cut else "mid"
        return {("recent", "high"): "Recent High Spenders",
                ("recent", "mid"):  "Recent Buyers",
                ("recent", "low"):  "Recent Low Spenders",
                ("mid", "high"):    "High Spenders",
                ("mid", "mid"):     "Average Customers",
                ("mid", "low"):     "Low Spenders",
                ("lapsed", "high"): "At-Risk High Spenders",
                ("lapsed", "mid"):  "Lapsed Buyers",
                ("lapsed", "low"):  "Lapsed Low Spenders"}[(rec, spend)]

    names = {}
    # name the highest-spending clusters first so they get the plain names
    for c in centres.sort_values("m_z", ascending=False).index:
        name = label(centres.at[c, "r_z"], centres.at[c, "m_z"], avg_freq[c])
        base, n = name, 2
        while name in names.values():  # keep names unique
            name, n = f"{base} {n}", n + 1
        names[c] = name
    return names


def profile_segments(df: pd.DataFrame, by: str) -> pd.DataFrame:
    total_rev = df["monetary"].sum()
    prof = (df.groupby(by)
              .agg(customers=("customer_unique_id", "count"),
                   avg_recency_days=("recency_days", "mean"),
                   avg_frequency=("frequency", "mean"),
                   avg_monetary=("monetary", "mean"),
                   revenue=("monetary", "sum"),
                   avg_review_score=("avg_review_score", "mean"),
                   late_share=("late_share", "mean"))
              .sort_values("revenue", ascending=False))
    prof["customer_share_pct"] = 100 * prof["customers"] / prof["customers"].sum()
    prof["revenue_share_pct"] = 100 * prof["revenue"] / total_rev
    return prof.round(2)


# ---------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------
def plot_k_selection(scores: pd.DataFrame, best_k: int) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(scores.k, scores.inertia, marker="o")
    a1.set(title="Elbow method", xlabel="k (clusters)", ylabel="Inertia")
    a2.plot(scores.k, scores.silhouette, marker="o", color="tab:orange")
    a2.axvline(best_k, ls="--", color="grey")
    a2.set(title=f"Silhouette score (chosen k = {best_k})",
           xlabel="k (clusters)", ylabel="Silhouette")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "k_selection.png", dpi=150)
    plt.close(fig)


def plot_segment_shares(profile: pd.DataFrame, title: str, fname: str) -> None:
    p = profile.sort_values("revenue_share_pct")
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(p) + 1.5))
    ax.barh(y - 0.2, p["customer_share_pct"], 0.4, label="% of customers")
    ax.barh(y + 0.2, p["revenue_share_pct"], 0.4, label="% of revenue")
    ax.set_yticks(y, p.index)
    ax.set(title=title, xlabel="Share (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def plot_clusters(df: pd.DataFrame) -> None:
    s = df.sample(min(len(df), 5_000), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, g in s.groupby("cluster_name"):
        ax.scatter(g["recency_days"], g["monetary"], s=8, alpha=0.5, label=name)
    ax.set_yscale("log")
    ax.set(title="Customer clusters: recency vs. lifetime spend",
           xlabel="Days since last purchase", ylabel="Lifetime spend (log scale)")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clusters_scatter.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------
def save_to_db(df: pd.DataFrame, engine) -> None:
    cols = ["customer_unique_id", "customer_state", "recency_days", "frequency",
            "monetary", "r_score", "f_score", "m_score", "rfm_code",
            "rfm_segment", "cluster", "cluster_name"]
    df[cols].to_sql("customer_segments", engine, if_exists="replace",
                    index=False, method=copy_insert, chunksize=50_000)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE customer_segments "
                          "ADD PRIMARY KEY (customer_unique_id)"))


def run(k: int | None = None) -> pd.DataFrame:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    print("1/5  Building RFM table ...")
    df = score_rfm(load_rfm(engine))

    print("2/5  Evaluating k ...")
    X = build_features(df)
    scores = evaluate_k(X)
    best_k = k or choose_k(scores)
    print(f"     -> using k = {best_k}")

    print("3/5  Fitting K-Means and naming clusters ...")
    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X)
    df["cluster"] = km.labels_
    df["cluster_name"] = df["cluster"].map(name_clusters(df, X))

    print("4/5  Profiling and plotting ...")
    cluster_prof = profile_segments(df, "cluster_name")
    rfm_prof = profile_segments(df, "rfm_segment")
    cluster_prof.to_csv(OUT_DIR / "segment_profile_kmeans.csv")
    rfm_prof.to_csv(OUT_DIR / "segment_profile_rfm.csv")
    scores.round(4).to_csv(OUT_DIR / "k_selection.csv", index=False)
    plot_k_selection(scores, best_k)
    plot_segment_shares(cluster_prof, "K-Means segments: customers vs. revenue",
                        "kmeans_segment_shares.png")
    plot_segment_shares(rfm_prof, "RFM segments: customers vs. revenue",
                        "rfm_segment_shares.png")
    plot_clusters(df)

    print("5/5  Saving customer_segments table ...")
    save_to_db(df, engine)

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\nK-Means segment profile:\n", cluster_prof.to_string())
    print("\nRFM segment profile:\n", rfm_prof.to_string())
    print(f"\nCharts saved to {FIG_DIR.relative_to(ROOT)}/")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, help="number of clusters (default: auto)")
    run(parser.parse_args().k)
