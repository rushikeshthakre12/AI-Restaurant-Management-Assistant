import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from services.booking_service import create_booking, cancel_booking, check_availability, resolve_relative_date


def test_resolve_relative_date_today():
    from datetime import date
    assert resolve_relative_date("today") == date.today().isoformat()


def test_create_and_cancel_booking():
    outcome = create_booking(user_id=2, booking_date="2027-01-01", booking_time="20:00", guests=2)
    assert outcome["success"] is True
    booking_id = outcome["booking_id"]

    cancel_outcome = cancel_booking(booking_id, user_id=2)
    assert cancel_outcome["success"] is True


def test_booking_fills_up_and_suggests_alternatives():
    date_str = "2027-06-15"
    time_str = "19:00"
    for _ in range(8):
        create_booking(user_id=2, booking_date=date_str, booking_time=time_str, guests=2)
    assert check_availability(date_str, time_str) is False
    outcome = create_booking(user_id=2, booking_date=date_str, booking_time=time_str, guests=2)
    assert outcome["success"] is False
