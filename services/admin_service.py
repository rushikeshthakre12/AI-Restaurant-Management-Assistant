"""Aggregation queries backing the admin dashboard KPIs."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.connection import execute


def get_kpis() -> dict:
    total_users = execute("SELECT COUNT(*) c FROM users WHERE role='customer'", fetch=True)[0]["c"]
    total_orders = execute("SELECT COUNT(*) c FROM orders", fetch=True)[0]["c"]
    total_revenue = execute("SELECT COALESCE(SUM(total_amount),0) r FROM orders WHERE status != 'cancelled'", fetch=True)[0]["r"]
    total_bookings = execute("SELECT COUNT(*) c FROM bookings", fetch=True)[0]["c"]
    cancelled_orders = execute("SELECT COUNT(*) c FROM orders WHERE status='cancelled'", fetch=True)[0]["c"]
    avg_rating = execute("SELECT AVG(rating) r FROM reviews", fetch=True)[0]["r"]
    return {
        "total_users": total_users,
        "total_orders": total_orders,
        "total_revenue": round(total_revenue or 0, 2),
        "total_bookings": total_bookings,
        "cancelled_orders": cancelled_orders,
        "avg_rating": round(avg_rating, 2) if avg_rating else 0,
    }


def get_popular_items(limit: int = 5) -> list[dict]:
    return execute(
        """SELECT mi.name, SUM(oi.quantity) as total_sold
           FROM order_items oi JOIN menu_items mi ON oi.item_id = mi.item_id
           GROUP BY mi.name ORDER BY total_sold DESC LIMIT ?""",
        (limit,), fetch=True,
    )


def get_sentiment_breakdown() -> dict:
    rows = execute("SELECT sentiment, COUNT(*) c FROM reviews GROUP BY sentiment", fetch=True)
    total = sum(r["c"] for r in rows) or 1
    return {r["sentiment"]: round(100 * r["c"] / total, 1) for r in rows}


def get_daily_sales(limit_days: int = 14) -> list[dict]:
    return execute(
        """SELECT date(order_date) as day, COUNT(*) as orders, SUM(total_amount) as revenue
           FROM orders WHERE status != 'cancelled'
           GROUP BY date(order_date) ORDER BY day DESC LIMIT ?""",
        (limit_days,), fetch=True,
    )


def get_repeat_customer_rate() -> float:
    rows = execute(
        "SELECT user_id, COUNT(*) c FROM orders WHERE status != 'cancelled' GROUP BY user_id", fetch=True
    )
    if not rows:
        return 0.0
    repeat = sum(1 for r in rows if r["c"] > 1)
    return round(100 * repeat / len(rows), 1)
