"""Small reusable Streamlit rendering helpers shared by customer.py and admin.py."""
import streamlit as st


def render_menu_item_card(item: dict):
    veg_tag = "🟢 Veg" if item.get("vegetarian") else "🔴 Non-Veg"
    spicy_tag = "🌶️ Spicy" if item.get("spicy") else ""
    st.markdown(f"**{item['name']}** — ₹{item['price']:.0f} · {veg_tag} {spicy_tag}")
    st.caption(item.get("description", ""))


def render_kpi_row(kpis: dict):
    cols = st.columns(3)
    cols[0].metric("Total Orders", kpis["total_orders"])
    cols[1].metric("Total Revenue", f"₹{kpis['total_revenue']:,.0f}")
    cols[2].metric("Total Bookings", kpis["total_bookings"])
    cols2 = st.columns(3)
    cols2[0].metric("Total Customers", kpis["total_users"])
    cols2[1].metric("Cancelled Orders", kpis["cancelled_orders"])
    cols2[2].metric("Avg Rating", f"{kpis['avg_rating']:.2f} ⭐" if kpis["avg_rating"] else "N/A")
