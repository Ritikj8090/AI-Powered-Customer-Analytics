"""Phase 4: weekly revenue forecasting by product category with Prophet.

For the total business and each top category:
  1. Aggregate delivered-order revenue by week.
  2. Hold out the last 12 weeks, fit Prophet on the rest, and measure
     MAPE and WAPE against a naive baseline (mean of the last 4 weeks).
  3. Refit on all history and forecast the next 12 weeks.

Results go to the `revenue_forecasts` table (used by the dashboard),
outputs/forecast_metrics.csv and charts in outputs/figures/.

Usage
    python src/forecasting.py                 # total + top 8 categories
    python src/forecasting.py --top 5 --horizon 8
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sqlalchemy import text

from db import copy_insert, get_engine, save_metrics

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TOTAL = "ALL CATEGORIES"

# Prophet and its Stan backend are very chatty
for name in ["prophet", "cmdstanpy"]:
    logging.getLogger(name).setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True

# Black Friday drives Olist's biggest spike (Nov 2017); telling Prophet
# about it stops the spike from distorting the trend.
BLACK_FRIDAY = pd.DataFrame({
    "holiday": "black_friday",
    "ds": pd.to_datetime(["2017-11-24", "2018-11-23"]),
    "lower_window": -3, "upper_window": 3,
})

WEEKLY_SQL = """
SELECT DATE_TRUNC('week', o.order_purchase_timestamp)::DATE   AS week,
       COALESCE(t.product_category_name_english,
                p.product_category_name, 'unknown')           AS category,
       SUM(oi.price + oi.freight_value)::FLOAT                AS revenue
FROM order_items oi
JOIN orders o   ON o.order_id = oi.order_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation t
       ON t.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
  AND o.order_purchase_timestamp >= '2017-01-02'
GROUP BY 1, 2
"""


def load_weekly(engine, top: int) -> pd.DataFrame:
    """Weekly revenue for the total and the top-N categories (wide format)."""
    with engine.connect() as conn:
        long = pd.read_sql(text(WEEKLY_SQL), conn, parse_dates=["week"])

    # Drop the last, incomplete week and the sparse tail of the data
    last_full_week = long["week"].max() - pd.Timedelta(days=7)
    long = long[long["week"] <= last_full_week]

    wide = long.pivot_table(index="week", columns="category", values="revenue",
                            aggfunc="sum", fill_value=0.0)
    wide = wide.asfreq("W-MON", fill_value=0.0)
    top_cats = wide.sum().sort_values(ascending=False).index[:top]
    out = wide[top_cats].copy()
    out.insert(0, TOTAL, wide.sum(axis=1))
    print(f"Weeks: {len(out)} ({out.index.min().date()} to {out.index.max().date()})")
    return out


def fit_prophet(series: pd.Series) -> Prophet:
    df = series.rename("y").rename_axis("ds").reset_index()
    model = Prophet(
        holidays=BLACK_FRIDAY,
        yearly_seasonality=False,   # under 2 years of data: not enough to learn it
        weekly_seasonality=False,   # data is already weekly
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        interval_width=0.8,
    )
    return model.fit(df)


def predict(model: Prophet, weeks: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=weeks, freq="W-MON")
    fc = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    fc[["yhat", "yhat_lower", "yhat_upper"]] = fc[["yhat", "yhat_lower",
                                                   "yhat_upper"]].clip(lower=0)
    return fc.set_index("ds")


def error_metrics(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    """MAPE (average % error per week) and WAPE (total error / total revenue).

    WAPE is reported alongside MAPE because MAPE explodes on low-revenue weeks.
    """
    a, p = actual.to_numpy(float), pred.to_numpy(float)
    nonzero = a > 0
    mape = float(np.mean(np.abs(a[nonzero] - p[nonzero]) / a[nonzero]) * 100)
    wape = float(np.abs(a - p).sum() / max(a.sum(), 1e-9) * 100)
    return round(mape, 2), round(wape, 2)


def backtest(series: pd.Series, horizon: int) -> dict:
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    fc = predict(fit_prophet(train), horizon).loc[test.index, "yhat"]
    naive = pd.Series(train.iloc[-4:].mean(), index=test.index)
    p_mape, p_wape = error_metrics(test, fc)
    n_mape, n_wape = error_metrics(test, naive)
    return {"prophet_mape": p_mape, "prophet_wape": p_wape,
            "naive_mape": n_mape, "naive_wape": n_wape,
            "wape_improvement_pct": round(100 * (n_wape - p_wape) / n_wape, 1)
            if n_wape else None}


def plot_forecast(name: str, history: pd.Series, fc: pd.DataFrame, ax) -> None:
    ax.plot(history.index, history.values, color="black", lw=1, label="Actual")
    future = fc[fc.index > history.index.max()]
    fitted = fc[fc.index <= history.index.max()]
    ax.plot(fitted.index, fitted["yhat"], color="tab:blue", lw=1, alpha=0.6, label="Model fit")
    ax.plot(future.index, future["yhat"], color="tab:orange", lw=2, label="Forecast")
    ax.fill_between(future.index, future["yhat_lower"], future["yhat_upper"],
                    color="tab:orange", alpha=0.25, label="80% interval")
    ax.set_title(name, fontsize=10)
    ax.tick_params(labelsize=8)


def run(top: int = 8, horizon: int = 12) -> pd.DataFrame:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    print("1/4  Loading weekly revenue ...")
    wide = load_weekly(engine, top)
    if len(wide) < horizon * 3:
        raise SystemExit(f"Only {len(wide)} weeks of data; need at least {horizon * 3}.")

    print(f"2/4  Backtesting on the last {horizon} weeks ...")
    metrics, forecasts = [], []
    for name in wide.columns:
        series = wide[name]
        m = backtest(series, horizon)
        m.update({"series": name, "weekly_avg_revenue": round(series.mean(), 2)})
        metrics.append(m)
        print(f"  {name:<32} Prophet MAPE {m['prophet_mape']:>6.1f}%  "
              f"WAPE {m['prophet_wape']:>6.1f}%  | naive WAPE {m['naive_wape']:>6.1f}%")

        fc = predict(fit_prophet(series), horizon)  # refit on all history
        fc["actual"] = series.reindex(fc.index)
        fc["series"] = name
        fc["is_forecast"] = fc.index > series.index.max()
        forecasts.append(fc)

    print("3/4  Plotting ...")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_forecast(f"{TOTAL}: weekly revenue", wide[TOTAL], forecasts[0], ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_total.png", dpi=150)
    plt.close(fig)

    cats = wide.columns[1:]
    ncols = 3
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), squeeze=False)
    for ax, name, fc in zip(axes.flat, cats, forecasts[1:]):
        plot_forecast(name, wide[name], fc, ax)
    for ax in axes.flat[len(cats):]:
        ax.axis("off")
    fig.suptitle(f"Weekly revenue forecast by category (next {horizon} weeks)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_categories.png", dpi=150)
    plt.close(fig)

    print("4/4  Saving ...")
    metrics_df = pd.DataFrame(metrics).set_index("series")
    metrics_df.to_csv(OUT_DIR / "forecast_metrics.csv")
    save_metrics(engine, "forecast", metrics_df.reset_index().to_dict(orient="records"))

    table = (pd.concat(forecasts).rename_axis("week").reset_index()
               .rename(columns={"series": "category", "yhat": "forecast",
                                "yhat_lower": "forecast_lower",
                                "yhat_upper": "forecast_upper"}))
    table = table[["category", "week", "actual", "forecast", "forecast_lower",
                   "forecast_upper", "is_forecast"]]
    table[["forecast", "forecast_lower", "forecast_upper"]] = \
        table[["forecast", "forecast_lower", "forecast_upper"]].round(2)
    table.to_sql("revenue_forecasts", engine, if_exists="replace", index=False,
                 method=copy_insert, chunksize=50_000)

    cat_rows = metrics_df.drop(index=TOTAL)
    print(f"\nTotal revenue:   MAPE {metrics_df.loc[TOTAL, 'prophet_mape']}%, "
          f"WAPE {metrics_df.loc[TOTAL, 'prophet_wape']}% "
          f"(naive WAPE {metrics_df.loc[TOTAL, 'naive_wape']}%)")
    print(f"Category median: MAPE {cat_rows['prophet_mape'].median():.1f}%, "
          f"WAPE {cat_rows['prophet_wape'].median():.1f}%")
    print("Saved revenue_forecasts table, outputs/forecast_metrics.csv and charts.")
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=8, help="number of categories")
    parser.add_argument("--horizon", type=int, default=12, help="weeks to forecast")
    args = parser.parse_args()
    run(args.top, args.horizon)
