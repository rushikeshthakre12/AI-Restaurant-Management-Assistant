"""
The chatbot orchestrator: ties NLP entity extraction + ML intent
classification + business logic services + response generation into one
call. This is the function the Streamlit UI calls for every chat message.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nlp.ner import extract_entities
from ml.predict_intent import predict_intent
from transformer.chatbot import generate_reply
from services.booking_service import create_booking, cancel_booking, resolve_relative_date
from services.order_service import place_order, get_user_orders, get_order_items
from services.recommendation_service import get_recommendations
from services.menu_service import get_item_by_name, filter_menu


def handle_message(user_id: int, text: str) -> dict:
    """Returns {'intent', 'entities', 'reply', 'data'} where 'data' holds any
    structured result (booking confirmation, order confirmation, menu rows,
    recommendation rows) the UI can render alongside the chat reply."""
    result = predict_intent(text)
    intent = result["intent"]
    entities = extract_entities(text)
    data = None

    if intent == "table_booking" and entities["NUMBER_OF_PEOPLE"] and entities["TIME"]:
        booking_date = resolve_relative_date(entities["DATE"][0] if entities["DATE"] else None)
        booking_time = _normalize_time(entities["TIME"][0])
        outcome = create_booking(user_id, booking_date, booking_time, entities["NUMBER_OF_PEOPLE"])
        data = outcome
        if outcome["success"]:
            reply = f"Your table for {entities['NUMBER_OF_PEOPLE']} has been booked for {booking_time} on {booking_date}."
        else:
            alt = outcome.get("alternatives") or []
            reply = outcome["message"] + (f" How about {' or '.join(alt)}?" if alt else "")

    elif intent == "cancel_booking":
        reply = generate_reply(intent, entities)

    elif intent == "place_order" and entities["FOOD_ITEM"]:
        cart = []
        for name in entities["FOOD_ITEM"]:
            item = get_item_by_name(name)
            if item:
                cart.append({"item_id": item["item_id"], "name": item["name"], "price": item["price"], "quantity": 1})
        if cart:
            outcome = place_order(user_id, cart)
            data = outcome
            if outcome["success"]:
                names = ", ".join(c["name"] for c in cart)
                reply = f"Your order for {names} has been placed. Total: ₹{outcome['bill']['total']:.2f}"
            else:
                reply = outcome["message"]
        else:
            reply = generate_reply(intent, entities)

    elif intent == "food_recommendation":
        df, mode = get_recommendations(text, user_id=user_id)
        data = df.to_dict("records") if df is not None and not df.empty else []
        reply = "Here are a few dishes you might enjoy:" if data else "I couldn't find a matching dish -- try a different price or category."

    elif intent == "menu_query":
        vegetarian = True if "vegetarian" in text.lower() or "veg" in text.lower() else None
        rows = filter_menu(vegetarian=vegetarian)
        data = rows[:10]
        reply = generate_reply(intent, entities)

    elif intent == "previous_order":
        orders = get_user_orders(user_id)
        data = orders
        reply = "Here is your order history." if orders else "You haven't placed any orders yet."

    else:
        reply = generate_reply(intent, entities)

    return {"intent": intent, "entities": entities, "reply": reply, "data": data}


def _normalize_time(time_str: str) -> str:
    """Convert a free-text time like '8 pm' or '7:30 pm' into 24-hour HH:MM."""
    time_str = time_str.strip().lower().replace(".", "")
    is_pm = "pm" in time_str
    is_am = "am" in time_str
    time_str = time_str.replace("pm", "").replace("am", "").strip()
    if ":" in time_str:
        hour, minute = time_str.split(":")
    else:
        hour, minute = time_str, "00"
    hour = int(hour)
    if is_pm and hour != 12:
        hour += 12
    if is_am and hour == 12:
        hour = 0
    return f"{hour:02d}:{int(minute):02d}"


if __name__ == "__main__":
    tests = [
        "book a table for 4 people tomorrow at 8 pm",
        "i want two paneer pizzas",
        "suggest something spicy under 300",
        "show me vegetarian dishes",
    ]
    for t in tests:
        print(t, "->", handle_message(user_id=2, text=t))
        print()
