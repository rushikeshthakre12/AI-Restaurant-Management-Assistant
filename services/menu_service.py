"""Menu browsing: search, filter, sort -- backed by real DB queries."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.connection import execute


def get_all_menu(only_available: bool = True) -> list[dict]:
    q = "SELECT * FROM menu_items"
    if only_available:
        q += " WHERE available = 1"
    q += " ORDER BY category, name"
    return execute(q, fetch=True)


def search_menu(keyword: str) -> list[dict]:
    like = f"%{keyword.lower()}%"
    return execute(
        "SELECT * FROM menu_items WHERE available = 1 AND "
        "(LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(ingredients) LIKE ?) "
        "ORDER BY name",
        (like, like, like), fetch=True,
    )


def filter_menu(vegetarian: bool | None = None, spicy: bool | None = None,
                 max_price: float | None = None, category: str | None = None) -> list[dict]:
    q = "SELECT * FROM menu_items WHERE available = 1"
    params = []
    if vegetarian is not None:
        q += " AND vegetarian = ?"
        params.append(1 if vegetarian else 0)
    if spicy is not None:
        q += " AND spicy = ?"
        params.append(1 if spicy else 0)
    if max_price is not None:
        q += " AND price <= ?"
        params.append(max_price)
    if category is not None:
        q += " AND LOWER(category) = ?"
        params.append(category.lower())
    q += " ORDER BY price"
    return execute(q, tuple(params), fetch=True)


def sort_menu_by_price(descending: bool = False) -> list[dict]:
    order = "DESC" if descending else "ASC"
    return execute(f"SELECT * FROM menu_items WHERE available = 1 ORDER BY price {order}", fetch=True)


def get_item_by_id(item_id: int) -> dict | None:
    rows = execute("SELECT * FROM menu_items WHERE item_id = ?", (item_id,), fetch=True)
    return rows[0] if rows else None


def get_item_by_name(name: str) -> dict | None:
    rows = execute("SELECT * FROM menu_items WHERE LOWER(name) = ?", (name.lower(),), fetch=True)
    return rows[0] if rows else None
