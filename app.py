"""
Entry point. Run with:  streamlit run app.py

Handles registration/login/logout and routes to the customer app or admin
dashboard based on role.
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from database.connection import init_sqlite_schema, execute, DB_BACKEND
from utils.helpers import hash_password, verify_password
from ui.customer import render_customer_app
from ui.admin import render_admin_app

st.set_page_config(page_title="AI Restaurant Assistant", page_icon="🍴", layout="wide")

if DB_BACKEND != "mysql":
    init_sqlite_schema()


def get_user_by_email(email: str):
    rows = execute("SELECT * FROM users WHERE email = ?", (email,), fetch=True)
    return rows[0] if rows else None


def register_user(name: str, email: str, password: str) -> tuple[bool, str]:
    if get_user_by_email(email):
        return False, "An account with this email already exists."
    execute(
        "INSERT INTO users (name, email, password_hash, role) VALUES (?,?,?, 'customer')",
        (name, email, hash_password(password)),
    )
    return True, "Account created! You can now log in."


def render_login():
    st.title("🍴 AI Restaurant Management Assistant")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            user = get_user_by_email(email)
            if user and verify_password(password, user["password_hash"]):
                st.session_state.user = dict(user)
                st.rerun()
            else:
                st.error("Invalid email or password.")
        st.caption("Demo accounts: admin@restaurant.com / admin123, customer@example.com / customer123")

    with tab_register:
        name = st.text_input("Name")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Create Account"):
            if name and reg_email and reg_password:
                success, message = register_user(name, reg_email, reg_password)
                (st.success if success else st.error)(message)
            else:
                st.warning("Please fill in all fields.")


def main():
    if "user" not in st.session_state:
        render_login()
        return

    user = st.session_state.user
    with st.sidebar:
        st.write(f"Logged in as **{user['name']}** ({user['role']})")
        if st.button("Logout"):
            del st.session_state["user"]
            st.session_state.pop("chat_history", None)
            st.session_state.pop("cart", None)
            st.rerun()

    if user["role"] == "admin":
        render_admin_app()
    else:
        render_customer_app(user["user_id"], user["name"])


if __name__ == "__main__":
    main()
