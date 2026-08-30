"""
Table booking business logic: availability check, create, cancel, modify,
view. A simple capacity rule is used for availability (max 8 concurrent
bookings per 30-minute slot) -- documented so it's easy to explain/adjust
in a viva, not hidden magic.
"""
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database.connection import execute

MAX_BOOKINGS_PER_SLOT = 8


def resolve_relative_date(date_word: str | None) -> str:
    today = date.today()
    if not date_word:
        return today.isoformat()
    word = date_word.lower()
    if word == "today":
        return today.isoformat()
    if word == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if word in weekdays:
        target = weekdays.index(word)
        days_ahead = (target - today.weekday()) % 7
        days_ahead = days_ahead or 7
        return (today + timedelta(days=days_ahead)).isoformat()
    return today.isoformat()


def check_availability(booking_date: str, booking_time: str) -> bool:
    rows = execute(
        "SELECT COUNT(*) as c FROM bookings WHERE booking_date = ? AND booking_time = ? AND status = 'confirmed'",
        (booking_date, booking_time), fetch=True,
    )
    return rows[0]["c"] < MAX_BOOKINGS_PER_SLOT


def suggest_alternative_times(booking_date: str, booking_time: str) -> list[str]:
    """If the requested slot is full, suggest the nearest 30-min slots either side."""
    try:
        hour, minute = map(int, booking_time.split(":"))
    except ValueError:
        return []
    candidates = []
    for delta in (-30, 30, -60, 60):
        total_minutes = hour * 60 + minute + delta
        if 0 <= total_minutes < 24 * 60:
            h, m = divmod(total_minutes, 60)
            candidate = f"{h:02d}:{m:02d}"
            if check_availability(booking_date, candidate):
                candidates.append(candidate)
    return candidates


def create_booking(user_id: int, booking_date: str, booking_time: str, guests: int) -> dict:
    if not check_availability(booking_date, booking_time):
        alternatives = suggest_alternative_times(booking_date, booking_time)
        return {"success": False, "message": "That time is fully booked.", "alternatives": alternatives}

    booking_id = execute(
        "INSERT INTO bookings (user_id, booking_date, booking_time, guests, status) VALUES (?,?,?,?, 'confirmed')",
        (user_id, booking_date, booking_time, guests),
    )
    return {"success": True, "booking_id": booking_id, "message": "Booking confirmed."}


def cancel_booking(booking_id: int, user_id: int) -> dict:
    rows = execute("SELECT * FROM bookings WHERE booking_id = ? AND user_id = ?", (booking_id, user_id), fetch=True)
    if not rows:
        return {"success": False, "message": "Booking not found."}
    execute("UPDATE bookings SET status = 'cancelled' WHERE booking_id = ?", (booking_id,))
    return {"success": True, "message": "Booking cancelled."}


def modify_booking(booking_id: int, user_id: int, new_date: str | None = None,
                    new_time: str | None = None, new_guests: int | None = None) -> dict:
    rows = execute("SELECT * FROM bookings WHERE booking_id = ? AND user_id = ?", (booking_id, user_id), fetch=True)
    if not rows:
        return {"success": False, "message": "Booking not found."}
    current = rows[0]
    booking_date = new_date or current["booking_date"]
    booking_time = new_time or current["booking_time"]
    guests = new_guests or current["guests"]
    if (booking_date, booking_time) != (current["booking_date"], current["booking_time"]) and \
            not check_availability(booking_date, booking_time):
        return {"success": False, "message": "New time slot is unavailable."}
    execute(
        "UPDATE bookings SET booking_date=?, booking_time=?, guests=? WHERE booking_id=?",
        (booking_date, booking_time, guests, booking_id),
    )
    return {"success": True, "message": "Booking updated."}


def get_user_bookings(user_id: int) -> list[dict]:
    return execute(
        "SELECT * FROM bookings WHERE user_id = ? ORDER BY booking_date DESC, booking_time DESC",
        (user_id,), fetch=True,
    )
