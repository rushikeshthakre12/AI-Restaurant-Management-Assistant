"""Small shared utilities: password hashing and currency formatting."""
import hashlib
import hmac
import os
import secrets

_ITERATIONS = 100_000


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 password hashing with a random per-password salt.
    Stored format: iterations$salt_hex$hash_hex
    """
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS)
    return f"{_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        iterations_s, salt, hash_hex = stored_hash.split("$")
        iterations = int(iterations_s)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations)
    return hmac.compare_digest(dk.hex(), hash_hex)


def format_currency(amount: float) -> str:
    return f"₹{amount:,.2f}"
