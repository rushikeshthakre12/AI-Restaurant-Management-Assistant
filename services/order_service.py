"""
Food ordering business logic: cart handling, placing/cancelling/modifying
orders, and bill calculation (subtotal, tax, discount, total).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.connection import execute

TAX_RATE = 0.05  # 5% GST-style flat tax, kept simple and documented


def calculate_bill(cart_items: list[dict], user_id: int | None = None) -> dict:
    """cart_items: [{'item_id': int, 'name': str, 'price': float, 'quantity': int}, ...]"""
    subtotal = sum(i["price"] * i["quantity"] for i in cart_items)
    tax = round(subtotal * TAX_RATE, 2)
    discount = 0.0

    if user_id is not None:
        past_orders = execute(
            "SELECT COUNT(*) as c FROM orders WHERE user_id = ? AND status != 'cancelled'",
            (user_id,), fetch=True,
        )[0]["c"]
        if past_orders >= 3:
            discount += round(subtotal * 0.10, 2)  # frequent customer discount
    if subtotal > 500:
        discount += round(subtotal * 0.05, 2)      # big order discount

    total = round(subtotal + tax - discount, 2)
    return {"subtotal": round(subtotal, 2), "tax": tax, "discount": round(discount, 2), "total": total}


def place_order(user_id: int, cart_items: list[dict]) -> dict:
    if not cart_items:
        return {"success": False, "message": "Cart is empty."}
    bill = calculate_bill(cart_items, user_id)
    order_id = execute(
        "INSERT INTO orders (user_id, total_amount, tax, discount, status) VALUES (?,?,?,?, 'placed')",
        (user_id, bill["total"], bill["tax"], bill["discount"]),
    )
    for item in cart_items:
        execute(
            "INSERT INTO order_items (order_id, item_id, quantity, price) VALUES (?,?,?,?)",
            (order_id, item["item_id"], item["quantity"], item["price"]),
        )
    return {"success": True, "order_id": order_id, "bill": bill}


def cancel_order(order_id: int, user_id: int) -> dict:
    rows = execute("SELECT * FROM orders WHERE order_id = ? AND user_id = ?", (order_id, user_id), fetch=True)
    if not rows:
        return {"success": False, "message": "Order not found."}
    if rows[0]["status"] == "completed":
        return {"success": False, "message": "Completed orders cannot be cancelled."}
    execute("UPDATE orders SET status = 'cancelled' WHERE order_id = ?", (order_id,))
    return {"success": True, "message": "Order cancelled."}


def get_order_items(order_id: int) -> list[dict]:
    return execute(
        """SELECT oi.*, mi.name FROM order_items oi
           JOIN menu_items mi ON oi.item_id = mi.item_id
           WHERE oi.order_id = ?""",
        (order_id,), fetch=True,
    )


def get_user_orders(user_id: int) -> list[dict]:
    return execute("SELECT * FROM orders WHERE user_id = ? ORDER BY order_date DESC", (user_id,), fetch=True)


def get_user_previous_item_names(user_id: int) -> list[str]:
    """Used by the recommendation service to personalize suggestions."""
    rows = execute(
        """SELECT DISTINCT mi.name FROM order_items oi
           JOIN orders o ON oi.order_id = o.order_id
           JOIN menu_items mi ON oi.item_id = mi.item_id
           WHERE o.user_id = ?""",
        (user_id,), fetch=True,
    )
    return [r["name"] for r in rows]
