"""
Database connection layer.

Default: SQLite, stored at data/restaurant.db. This requires no server and
runs anywhere, which is what makes the project runnable out of the box.

To use a real MySQL server instead (recommended for your final submission,
since the brief specifies MySQL), set DB_BACKEND=mysql in your .env file
and fill in DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD. The rest of
the codebase talks to this module only through get_connection() and
execute(), so no other file needs to change.
"""
import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_PATH = BASE_DIR / "data" / "restaurant.db"

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").lower()


def get_connection():
    """Return a live DB connection. Caller is responsible for closing it."""
    if DB_BACKEND == "mysql":
        import mysql.connector
        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            database=os.getenv("DB_NAME", "restaurant_assistant"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
        )
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _placeholder():
    return "%s" if DB_BACKEND == "mysql" else "?"


def execute(query: str, params: tuple = (), fetch: bool = False, many: bool = False):
    """
    Run a parameterized query. Never build SQL with string formatting of
    user input elsewhere in the codebase -- always call this with a
    placeholder query + params tuple.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        if many:
            cur.executemany(query, params)
        else:
            cur.execute(query, params)
        if fetch:
            rows = cur.fetchall()
            result = [dict(row) for row in rows] if DB_BACKEND != "mysql" else rows
            return result
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def init_sqlite_schema():
    """Create the SQLite-compatible tables (used only for the sqlite backend)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'customer',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS menu_items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            vegetarian INTEGER NOT NULL DEFAULT 1,
            spicy INTEGER NOT NULL DEFAULT 0,
            ingredients TEXT,
            rating REAL DEFAULT 0,
            available INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS bookings (
            booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            booking_date TEXT NOT NULL,
            booking_time TEXT NOT NULL,
            guests INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'confirmed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_amount REAL NOT NULL DEFAULT 0,
            tax REAL NOT NULL DEFAULT 0,
            discount REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'placed',
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS order_items (
            order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            item_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (item_id) REFERENCES menu_items(item_id)
        );

        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            item_id INTEGER,
            rating INTEGER NOT NULL,
            comment TEXT NOT NULL,
            sentiment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (item_id) REFERENCES menu_items(item_id)
        );

        CREATE TABLE IF NOT EXISTS offers (
            offer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            discount REAL NOT NULL,
            valid_from TEXT NOT NULL,
            valid_until TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);
        CREATE INDEX IF NOT EXISTS idx_bookings_user ON bookings(user_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_item ON reviews(item_id);
        CREATE INDEX IF NOT EXISTS idx_menu_category ON menu_items(category);
        """
    )
    conn.commit()
    conn.close()
