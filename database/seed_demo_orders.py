"""
Optional: seed realistic demo ORDERS and BOOKINGS (separate from seed.py's
core menu/users/reviews) so the admin dashboard's charts, K-Means
segmentation, and sales prediction have real historical data to compute
from. Safe to skip -- the app runs fine with zero orders, just with empty
analytics until real orders are placed.

Run with: python -m database.seed_demo_orders
"""
import random
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.connection import execute, DB_BACKEND

random.seed(7)


def run():
    menu = execute("SELECT item_id, price FROM menu_items", fetch=True)
    user_ids = list(range(2, 16))  # demo customers seeded in seed.py

    # Give each user a random number of past orders spread over the last 60 days,
    # with a skew so some users look "frequent" and others "occasional" --
    # this drives real variance for K-Means rather than uniform noise.
    order_count = 0
    for uid in user_ids:
        num_orders = random.choice([1, 1, 2, 2, 3, 3, 4, 6, 8])
        for _ in range(num_orders):
            days_ago = random.randint(0, 60)
            order_date = (date.today() - timedelta(days=days_ago)).isoformat()
            cart = random.sample(menu, k=random.randint(1, 3))
            subtotal = sum(item["price"] * random.randint(1, 2) for item in cart)
            tax = round(subtotal * 0.05, 2)
            discount = round(subtotal * 0.10, 2) if random.random() < 0.3 else 0.0
            total = round(subtotal + tax - discount, 2)

            order_id = execute(
                "INSERT INTO orders (user_id, order_date, total_amount, tax, discount, status) VALUES (?,?,?,?,?, 'completed')",
                (uid, order_date, total, tax, discount),
            )
            for item in cart:
                qty = random.randint(1, 2)
                execute(
                    "INSERT INTO order_items (order_id, item_id, quantity, price) VALUES (?,?,?,?)",
                    (order_id, item["item_id"], qty, item["price"]),
                )
            order_count += 1

    # A handful of bookings too, for the admin KPI cards
    booking_count = 0
    for uid in random.sample(user_ids, k=8):
        days_ahead = random.randint(-30, 10)
        b_date = (date.today() + timedelta(days=days_ahead)).isoformat()
        b_time = random.choice(["19:00", "19:30", "20:00", "20:30", "21:00"])
        guests = random.choice([2, 2, 3, 4, 4, 6])
        status = "completed" if days_ahead < 0 else "confirmed"
        execute(
            "INSERT INTO bookings (user_id, booking_date, booking_time, guests, status) VALUES (?,?,?,?,?)",
            (uid, b_date, b_time, guests, status),
        )
        booking_count += 1

    print(f"Seeded {order_count} demo orders and {booking_count} demo bookings for dashboard analytics.")


if __name__ == "__main__":
    run()
