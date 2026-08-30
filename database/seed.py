"""
Seed the database with sample data: an admin user, a demo customer, the full
menu from data/menu.csv, and the sample reviews from data/reviews.csv (run
through the real sentiment model so sentiment labels are computed, not
hard-coded).

Run with:  python -m database.seed
"""
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.connection import init_sqlite_schema, execute, DB_BACKEND
from utils.helpers import hash_password

BASE_DIR = Path(__file__).resolve().parent.parent


def seed_users():
    execute(
        "INSERT OR IGNORE INTO users (user_id, name, email, password_hash, role) VALUES (?,?,?,?,?)"
        if DB_BACKEND != "mysql" else
        "INSERT IGNORE INTO users (user_id, name, email, password_hash, role) VALUES (%s,%s,%s,%s,%s)",
        (1, "Admin", "admin@restaurant.com", hash_password("admin123"), "admin"),
    )
    execute(
        "INSERT OR IGNORE INTO users (user_id, name, email, password_hash, role) VALUES (?,?,?,?,?)"
        if DB_BACKEND != "mysql" else
        "INSERT IGNORE INTO users (user_id, name, email, password_hash, role) VALUES (%s,%s,%s,%s,%s)",
        (2, "Demo Customer", "customer@example.com", hash_password("customer123"), "customer"),
    )
    # Extra demo customers (3-15) so the sample reviews.csv (which references
    # these user_ids) can be seeded with valid foreign keys.
    for uid in range(3, 16):
        execute(
            "INSERT OR IGNORE INTO users (user_id, name, email, password_hash, role) VALUES (?,?,?,?,?)"
            if DB_BACKEND != "mysql" else
            "INSERT IGNORE INTO users (user_id, name, email, password_hash, role) VALUES (%s,%s,%s,%s,%s)",
            (uid, f"Customer {uid}", f"customer{uid}@example.com", hash_password("password123"), "customer"),
        )
    print("Seeded users: admin@restaurant.com / admin123, customer@example.com / customer123 (+ 13 demo customers)")


def seed_menu():
    path = BASE_DIR / "data" / "menu.csv"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    ph = ",".join(["?"] * 10) if DB_BACKEND != "mysql" else ",".join(["%s"] * 10)
    for row in rows:
        execute(
            f"""INSERT OR IGNORE INTO menu_items
                (item_id, name, category, description, price, vegetarian, spicy, ingredients, rating, available)
                VALUES ({ph})""" if DB_BACKEND != "mysql" else
            f"""INSERT IGNORE INTO menu_items
                (item_id, name, category, description, price, vegetarian, spicy, ingredients, rating, available)
                VALUES ({ph})""",
            (
                int(row["item_id"]), row["name"], row["category"], row["description"],
                float(row["price"]), int(row["vegetarian"]), int(row["spicy"]),
                row["ingredients"], float(row["rating"]), int(row["available"]),
            ),
        )
    print(f"Seeded {len(rows)} menu items")


def seed_reviews():
    from ml.sentiment import analyze_sentiment  # real model, not a hard-coded label

    path = BASE_DIR / "data" / "reviews.csv"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        sentiment = analyze_sentiment(row["comment"])["overall"]
        ph = ",".join(["?"] * 6) if DB_BACKEND != "mysql" else ",".join(["%s"] * 6)
        execute(
            f"""INSERT OR IGNORE INTO reviews (review_id, user_id, item_id, rating, comment, sentiment)
                VALUES ({ph})""" if DB_BACKEND != "mysql" else
            f"""INSERT IGNORE INTO reviews (review_id, user_id, item_id, rating, comment, sentiment)
                VALUES ({ph})""",
            (int(row["review_id"]), int(row["user_id"]), int(row["item_id"]),
             int(row["rating"]), row["comment"], sentiment),
        )
    print(f"Seeded {len(rows)} reviews with computed sentiment labels")


def seed_offers():
    execute(
        "INSERT OR IGNORE INTO offers (offer_id, title, description, discount, valid_from, valid_until) VALUES (?,?,?,?,?,?)"
        if DB_BACKEND != "mysql" else
        "INSERT IGNORE INTO offers (offer_id, title, description, discount, valid_from, valid_until) VALUES (%s,%s,%s,%s,%s,%s)",
        (1, "Frequent Diner Discount", "10% off for customers with 3+ past orders", 10.0, "2025-01-01", "2026-12-31"),
    )
    execute(
        "INSERT OR IGNORE INTO offers (offer_id, title, description, discount, valid_from, valid_until) VALUES (?,?,?,?,?,?)"
        if DB_BACKEND != "mysql" else
        "INSERT IGNORE INTO offers (offer_id, title, description, discount, valid_from, valid_until) VALUES (%s,%s,%s,%s,%s,%s)",
        (2, "Big Order Discount", "5% off on orders above 500", 5.0, "2025-01-01", "2026-12-31"),
    )
    print("Seeded offers")


if __name__ == "__main__":
    if DB_BACKEND != "mysql":
        init_sqlite_schema()
    seed_users()
    seed_menu()
    seed_offers()
    seed_reviews()
    print("Seeding complete.")
