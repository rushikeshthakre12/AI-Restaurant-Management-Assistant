import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.recommendation import recommend_from_query, recommend_similar_items, recommend_from_order_history


def test_recommend_vegetarian_only():
    df = recommend_from_query("suggest vegetarian food")
    assert not df.empty
    ids = set(df["item_id"])
    # spot check: Chicken Biryani (non-veg) should not appear
    names = set(df["name"].str.lower())
    assert "chicken biryani" not in names


def test_recommend_similar_items_excludes_self():
    df = recommend_similar_items("Paneer Tikka")
    assert not df.empty
    assert "Paneer Tikka" not in set(df["name"])


def test_recommend_from_order_history():
    df = recommend_from_order_history(["Paneer Pizza"])
    assert not df.empty
    assert "Paneer Pizza" not in set(df["name"])
