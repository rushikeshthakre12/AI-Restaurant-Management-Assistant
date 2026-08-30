import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from services.order_service import calculate_bill, place_order, cancel_order


def test_calculate_bill_basic():
    cart = [{"item_id": 1, "name": "Paneer Tikka", "price": 249, "quantity": 2}]
    bill = calculate_bill(cart)
    assert bill["subtotal"] == 498
    assert bill["tax"] == round(498 * 0.05, 2)


def test_calculate_bill_big_order_discount():
    cart = [{"item_id": 1, "name": "Veg Thali", "price": 349, "quantity": 2}]
    bill = calculate_bill(cart)
    assert bill["discount"] > 0  # subtotal 698 > 500 -> big order discount applies


def test_place_and_cancel_order():
    cart = [{"item_id": 2, "name": "Veg Biryani", "price": 199, "quantity": 1}]
    outcome = place_order(user_id=2, cart_items=cart)
    assert outcome["success"] is True
    cancel_outcome = cancel_order(outcome["order_id"], user_id=2)
    assert cancel_outcome["success"] is True


def test_place_order_empty_cart_fails():
    outcome = place_order(user_id=2, cart_items=[])
    assert outcome["success"] is False
