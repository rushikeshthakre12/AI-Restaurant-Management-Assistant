"""
Custom restaurant-domain entity extraction (Practical 1, CO1).

General-purpose NER (spaCy's pretrained model) is not reliable for
restaurant-specific slots like "8 pm", "4 people", or matching a free-text
food name against your exact menu -- these need domain rules, which is
exactly what the brief allows ("If standard NER cannot detect
restaurant-specific entities, implement custom entity extraction").

Entities extracted: DATE, TIME, NUMBER_OF_PEOPLE, FOOD_ITEM, MONEY.
"""
import re
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
_MENU_PATH = BASE_DIR / "data" / "menu.csv"

_DATE_WORDS = {
    "today": "today", "tonight": "today", "tomorrow": "tomorrow",
    "monday": "monday", "tuesday": "tuesday", "wednesday": "wednesday",
    "thursday": "thursday", "friday": "friday", "saturday": "saturday",
    "sunday": "sunday",
}

_TIME_RE = re.compile(
    r"\b(1[0-2]|0?[1-9])(:[0-5][0-9])?\s*(am|pm|a\.m\.|p\.m\.)\b", re.IGNORECASE
)
_MONEY_RE = re.compile(r"(?:₹|rs\.?|inr)\s?(\d+(?:\.\d{1,2})?)", re.IGNORECASE)
_MONEY_UNDER_RE = re.compile(r"under\s+(?:₹|rs\.?)?\s?(\d+)", re.IGNORECASE)
_PEOPLE_RE = re.compile(
    r"(\d+)\s*(?:people|persons?|guests?|pax|members?)|(?:for|of)\s+(\d+)\b",
    re.IGNORECASE,
)


def _load_menu_names() -> list[str]:
    if not _MENU_PATH.exists():
        return []
    with open(_MENU_PATH, newline="", encoding="utf-8") as f:
        return [row["name"].lower() for row in csv.DictReader(f)]


_MENU_NAMES = _load_menu_names()


def extract_dates(text: str) -> list[str]:
    lower = text.lower()
    return [canon for word, canon in _DATE_WORDS.items() if re.search(rf"\b{word}\b", lower)]


def extract_times(text: str) -> list[str]:
    return [m.group(0).strip() for m in _TIME_RE.finditer(text)]


def extract_money(text: str) -> list[str]:
    found = [m.group(1) for m in _MONEY_RE.finditer(text)]
    found += [m.group(1) for m in _MONEY_UNDER_RE.finditer(text)]
    # dedupe while preserving order (same amount can match both patterns)
    seen = []
    for f in found:
        if f not in seen:
            seen.append(f)
    return seen


def extract_people_count(text: str) -> int | None:
    m = _PEOPLE_RE.search(text)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    return int(val) if val else None


def extract_food_items(text: str) -> list[str]:
    lower = text.lower()
    found = [name for name in _MENU_NAMES if name in lower]
    # keep longest matches only (avoid "pizza" matching inside "paneer pizza" twice)
    found.sort(key=len, reverse=True)
    deduped = []
    for f in found:
        if not any(f in other for other in deduped):
            deduped.append(f)
    return deduped


def extract_entities(text: str) -> dict:
    return {
        "DATE": extract_dates(text),
        "TIME": extract_times(text),
        "NUMBER_OF_PEOPLE": extract_people_count(text),
        "FOOD_ITEM": extract_food_items(text),
        "MONEY": extract_money(text),
    }


if __name__ == "__main__":
    tests = [
        "I want to book a table for 4 people tomorrow at 8 PM.",
        "Suggest something spicy under ₹300.",
        "I want two paneer pizzas and one coke.",
        "Book a table for 6 people on saturday at 7:30 pm.",
    ]
    for t in tests:
        print(t)
        print(" ->", extract_entities(t))
        print()
