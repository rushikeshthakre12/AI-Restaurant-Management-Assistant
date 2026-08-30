"""Customer-facing Streamlit pages: chatbot, menu browsing, cart, bookings, orders, reviews."""
import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from services.menu_service import get_all_menu, search_menu, filter_menu, sort_menu_by_price, get_item_by_id
from services.booking_service import get_user_bookings, cancel_booking as cancel_booking_service
from services.order_service import calculate_bill, place_order, get_user_orders, get_order_items, cancel_order as cancel_order_service
from services.chatbot_service import handle_message
from ml.sentiment import analyze_sentiment
from utils.pdf_bill import generate_bill_pdf
from database.connection import execute
from ui.components import render_menu_item_card


def render_chat_tab(user_id: int):
    st.subheader("💬 Chat with our AI assistant")
    st.caption("Try: \"book a table for 4 at 8pm\", \"suggest something spicy under 300\", \"i want two paneer pizzas\"")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        result = handle_message(user_id, user_input)
        reply = result["reply"]
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        if result["data"] and isinstance(result["data"], list) and result["intent"] in ("food_recommendation", "menu_query"):
            for row in result["data"][:5]:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"• {row.get('name')} — ₹{row.get('price', 0):.0f}",
                })
        st.rerun()


def render_menu_tab():
    st.subheader("🍽️ Menu")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        veg_filter = st.selectbox("Diet", ["All", "Vegetarian", "Non-Vegetarian"])
    with col2:
        spicy_filter = st.selectbox("Spice", ["All", "Spicy", "Mild"])
    with col3:
        max_price = st.number_input("Max price (₹)", min_value=0, value=0, step=50)
    with col4:
        sort_by_price = st.selectbox("Sort", ["Default", "Price: Low to High", "Price: High to Low"])

    veg = True if veg_filter == "Vegetarian" else False if veg_filter == "Non-Vegetarian" else None
    spicy = True if spicy_filter == "Spicy" else False if spicy_filter == "Mild" else None
    items = filter_menu(vegetarian=veg, spicy=spicy, max_price=max_price or None)

    if sort_by_price == "Price: Low to High":
        items = sorted(items, key=lambda x: x["price"])
    elif sort_by_price == "Price: High to Low":
        items = sorted(items, key=lambda x: -x["price"])

    if "cart" not in st.session_state:
        st.session_state.cart = []

    for item in items:
        c1, c2 = st.columns([4, 1])
        with c1:
            render_menu_item_card(item)
        with c2:
            if st.button("Add to cart", key=f"add_{item['item_id']}"):
                st.session_state.cart.append(item)
                st.toast(f"Added {item['name']} to cart")


def render_cart_and_order_tab(user_id: int):
    st.subheader("🛒 Cart & Checkout")
    cart = st.session_state.get("cart", [])
    if not cart:
        st.info("Your cart is empty. Add items from the Menu tab.")
        return

    from collections import Counter
    counts = Counter(item["item_id"] for item in cart)
    unique_items = {item["item_id"]: item for item in cart}
    cart_items = [
        {"item_id": iid, "name": unique_items[iid]["name"], "price": unique_items[iid]["price"], "quantity": qty}
        for iid, qty in counts.items()
    ]

    for ci in cart_items:
        st.write(f"{ci['name']} × {ci['quantity']} — ₹{ci['price'] * ci['quantity']:.0f}")

    bill = calculate_bill(cart_items, user_id=user_id)
    st.divider()
    st.write(f"Subtotal: ₹{bill['subtotal']:.2f}")
    st.write(f"Tax: ₹{bill['tax']:.2f}")
    st.write(f"Discount: -₹{bill['discount']:.2f}")
    st.markdown(f"### Total: ₹{bill['total']:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Place Order", type="primary"):
            outcome = place_order(user_id, cart_items)
            if outcome["success"]:
                st.session_state.cart = []
                pdf_path = generate_bill_pdf(outcome["order_id"], cart_items, outcome["bill"])
                st.success(f"Order #{outcome['order_id']} placed! Total ₹{outcome['bill']['total']:.2f}")
                with open(pdf_path, "rb") as f:
                    st.download_button("Download Bill (PDF)", f, file_name=pdf_path.name)
            else:
                st.error(outcome["message"])
    with col2:
        if st.button("Clear Cart"):
            st.session_state.cart = []
            st.rerun()


def render_booking_tab(user_id: int):
    st.subheader("📅 Table Booking")
    col1, col2, col3 = st.columns(3)
    with col1:
        booking_date = st.date_input("Date")
    with col2:
        booking_time = st.time_input("Time")
    with col3:
        guests = st.number_input("Guests", min_value=1, max_value=20, value=2)

    if st.button("Book Table", type="primary"):
        from services.booking_service import create_booking
        outcome = create_booking(user_id, str(booking_date), booking_time.strftime("%H:%M"), int(guests))
        if outcome["success"]:
            st.success(outcome["message"])
        else:
            st.error(outcome["message"])
            if outcome.get("alternatives"):
                st.info(f"Available alternative times: {', '.join(outcome['alternatives'])}")

    st.divider()
    st.write("**Your bookings**")
    bookings = get_user_bookings(user_id)
    for b in bookings:
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.write(f"{b['booking_date']} at {b['booking_time']} — {b['guests']} guests ({b['status']})")
        if b["status"] == "confirmed":
            if c2.button("Cancel", key=f"cancel_booking_{b['booking_id']}"):
                cancel_booking_service(b["booking_id"], user_id)
                st.rerun()


def render_orders_tab(user_id: int):
    st.subheader("📦 Your Orders")
    orders = get_user_orders(user_id)
    if not orders:
        st.info("No orders yet.")
        return
    for o in orders:
        with st.expander(f"Order #{o['order_id']} — ₹{o['total_amount']:.2f} ({o['status']}) — {o['order_date']}"):
            for item in get_order_items(o["order_id"]):
                st.write(f"{item['name']} × {item['quantity']} — ₹{item['price'] * item['quantity']:.2f}")
            if o["status"] not in ("completed", "cancelled"):
                if st.button("Cancel Order", key=f"cancel_order_{o['order_id']}"):
                    cancel_order_service(o["order_id"], user_id)
                    st.rerun()


def render_reviews_tab(user_id: int):
    st.subheader("⭐ Leave a Review")
    items = get_all_menu()
    item_names = [i["name"] for i in items]
    selected_name = st.selectbox("Which dish?", item_names)
    rating = st.slider("Rating", 1, 5, 4)
    comment = st.text_area("Your review")

    if st.button("Submit Review"):
        if comment.strip():
            item = next(i for i in items if i["name"] == selected_name)
            sentiment = analyze_sentiment(comment)
            execute(
                "INSERT INTO reviews (user_id, item_id, rating, comment, sentiment) VALUES (?,?,?,?,?)",
                (user_id, item["item_id"], rating, comment, sentiment["overall"]),
            )
            st.success(f"Thanks for your review! Detected sentiment: {sentiment}")
        else:
            st.warning("Please write a comment before submitting.")


def render_customer_app(user_id: int, user_name: str):
    st.title(f"🍴 Welcome, {user_name}")
    tabs = st.tabs(["Chat", "Menu", "Cart", "Bookings", "Orders", "Reviews"])
    with tabs[0]:
        render_chat_tab(user_id)
    with tabs[1]:
        render_menu_tab()
    with tabs[2]:
        render_cart_and_order_tab(user_id)
    with tabs[3]:
        render_booking_tab(user_id)
    with tabs[4]:
        render_orders_tab(user_id)
    with tabs[5]:
        render_reviews_tab(user_id)
