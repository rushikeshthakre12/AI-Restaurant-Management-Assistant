"""Admin dashboard: KPIs, popular items, sales charts, customer analytics,
review sentiment, and ML analytics (intent classifier report, clusters,
demand prediction)."""
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from services.admin_service import get_kpis, get_popular_items, get_sentiment_breakdown, get_daily_sales, get_repeat_customer_rate
from ml.clustering import segment_customers
from ml.prediction import predict_next_day
from ui.components import render_kpi_row

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_REPORT_PATH = BASE_DIR / "ml" / "saved_models" / "intent_eval_report.json"


def render_admin_app():
    st.title("📊 Admin Dashboard")

    kpis = get_kpis()
    render_kpi_row(kpis)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Popular Items")
        popular = get_popular_items()
        if popular:
            st.bar_chart(pd.DataFrame(popular).set_index("name"))
        else:
            st.info("No order data yet.")

    with col2:
        st.subheader("😊 Review Sentiment")
        sentiment = get_sentiment_breakdown()
        if sentiment:
            st.bar_chart(pd.Series(sentiment))
        else:
            st.info("No reviews yet.")

    st.divider()
    st.subheader("📈 Daily Sales (last 14 days with orders)")
    daily = get_daily_sales()
    if daily:
        df = pd.DataFrame(daily).sort_values("day")
        st.line_chart(df.set_index("day")[["revenue"]])
    else:
        st.info("No sales data yet.")

    st.divider()
    st.subheader("👥 Customer Analytics")
    st.metric("Repeat Customer Rate", f"{get_repeat_customer_rate()}%")
    segments = segment_customers()
    if not segments.empty:
        st.dataframe(segments[["user_id", "num_orders", "avg_order_value", "segment"]])
        st.bar_chart(segments["segment"].value_counts())
    else:
        st.info("Not enough order history yet to compute customer segments (K-Means needs a few customers with order history).")

    st.divider()
    st.subheader("🔮 Demand Prediction (next day)")
    forecast = predict_next_day()
    if "error" in forecast:
        st.info(forecast["error"])
    else:
        c1, c2 = st.columns(2)
        c1.metric("Predicted Orders Tomorrow", forecast["predicted_orders"])
        c2.metric("Predicted Revenue Tomorrow", f"₹{forecast['predicted_revenue']:,.0f}")
        st.caption(
            f"Model mean absolute error on held-out days -- orders: {forecast['order_mae']:.2f}, "
            f"revenue: ₹{forecast['revenue_mae']:.2f}. Small dataset, so treat this as a "
            f"directional estimate, not a precise forecast."
        )

    st.divider()
    st.subheader("🤖 ML Analytics — Intent Classifier")
    if EVAL_REPORT_PATH.exists():
        with open(EVAL_REPORT_PATH) as f:
            report = json.load(f)
        for model_name, metrics in report.items():
            st.write(f"**{model_name}**: accuracy={metrics['accuracy']:.3f}, "
                     f"F1 (macro)={metrics['f1_macro']:.3f}")
        with st.expander("Full classification report"):
            for model_name, metrics in report.items():
                st.text(f"--- {model_name} ---\n{metrics['classification_report']}")
    else:
        st.info("Run `python -m ml.train_intent` to generate the evaluation report.")
