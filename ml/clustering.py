"""
Customer segmentation using K-Means, based on real order history: number of
orders, average order value, and visit frequency (days since first order /
number of orders, as a proxy). Falls back gracefully when there isn't
enough order history yet (common early in a fresh DB).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.connection import execute


def build_customer_features() -> pd.DataFrame:
    rows = execute(
        """SELECT user_id, COUNT(*) as num_orders, AVG(total_amount) as avg_order_value,
                  SUM(total_amount) as total_spent
           FROM orders WHERE status != 'cancelled'
           GROUP BY user_id""",
        fetch=True,
    )
    return pd.DataFrame(rows)


def segment_customers(n_clusters: int = 4) -> pd.DataFrame:
    df = build_customer_features()
    if len(df) < n_clusters:
        return pd.DataFrame()  # not enough customers with orders yet to cluster meaningfully

    features = df[["num_orders", "avg_order_value", "total_spent"]].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled)

    # Label clusters using their own computed centroid stats (not a fixed
    # ranking), so labels stay honest about what each cluster represents:
    # split on order frequency (median) then on spend (median) within each half.
    cluster_stats = df.groupby("cluster")[["num_orders", "avg_order_value"]].mean()
    freq_median = cluster_stats["num_orders"].median()
    value_median = cluster_stats["avg_order_value"].median()

    def label_for(row):
        frequent = row["num_orders"] >= freq_median
        high_value = row["avg_order_value"] >= value_median
        if frequent and high_value:
            return "Frequent Customers"
        if frequent and not high_value:
            return "Budget Customers"
        if not frequent and high_value:
            return "Premium Customers"
        return "Occasional Customers"

    label_map = {cluster_id: label_for(stats) for cluster_id, stats in cluster_stats.iterrows()}
    df["segment"] = df["cluster"].map(label_map)
    return df


if __name__ == "__main__":
    result = segment_customers()
    if result.empty:
        print("Not enough customer order history to cluster yet.")
    else:
        print(result[["user_id", "num_orders", "avg_order_value", "segment"]])
