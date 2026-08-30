"""
Sales / demand prediction using simple linear regression on daily order
counts and daily revenue, aggregated from real order history. Kept
deliberately simple per the brief -- day index, weekend flag, and previous
day's count/sales as features.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.connection import execute


def build_daily_sales() -> pd.DataFrame:
    rows = execute(
        """SELECT date(order_date) as day, COUNT(*) as order_count, SUM(total_amount) as revenue
           FROM orders WHERE status != 'cancelled'
           GROUP BY date(order_date) ORDER BY day""",
        fetch=True,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["day"])
    full_range = pd.date_range(df["day"].min(), df["day"].max())
    df = df.set_index("day").reindex(full_range, fill_value=0).rename_axis("day").reset_index()
    df["is_weekend"] = df["day"].dt.dayofweek.isin([5, 6]).astype(int)
    df["day_index"] = range(len(df))
    df["prev_order_count"] = df["order_count"].shift(1).fillna(0)
    df["prev_revenue"] = df["revenue"].shift(1).fillna(0)
    return df


def train_demand_model():
    df = build_daily_sales()
    if len(df) < 5:
        return None, None, "Not enough daily order history yet to train a demand model."

    features = ["day_index", "is_weekend", "prev_order_count", "prev_revenue"]
    X = df[features]
    y_orders = df["order_count"]
    y_revenue = df["revenue"]

    split = max(1, int(len(df) * 0.8))
    order_model = LinearRegression().fit(X[:split], y_orders[:split])
    revenue_model = LinearRegression().fit(X[:split], y_revenue[:split])

    if split < len(df):
        order_mae = mean_absolute_error(y_orders[split:], order_model.predict(X[split:]))
        revenue_mae = mean_absolute_error(y_revenue[split:], revenue_model.predict(X[split:]))
    else:
        order_mae = revenue_mae = None

    return {"order_model": order_model, "revenue_model": revenue_model, "features": features, "last_row": df.iloc[-1]}, \
           {"order_mae": order_mae, "revenue_mae": revenue_mae}, None


def predict_next_day():
    models, metrics, error = train_demand_model()
    if error:
        return {"error": error}
    last = models["last_row"]
    next_features = pd.DataFrame([{
        "day_index": last["day_index"] + 1,
        "is_weekend": 1 if (last["day"] + pd.Timedelta(days=1)).dayofweek in (5, 6) else 0,
        "prev_order_count": last["order_count"],
        "prev_revenue": last["revenue"],
    }])[models["features"]]

    predicted_orders = max(0, round(models["order_model"].predict(next_features)[0]))
    predicted_revenue = max(0.0, round(models["revenue_model"].predict(next_features)[0], 2))
    return {
        "predicted_orders": predicted_orders,
        "predicted_revenue": predicted_revenue,
        "order_mae": metrics["order_mae"],
        "revenue_mae": metrics["revenue_mae"],
    }


if __name__ == "__main__":
    print(predict_next_day())
