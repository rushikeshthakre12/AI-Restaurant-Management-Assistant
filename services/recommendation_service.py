"""Thin service wrapper around ml/recommendation.py that also folds in a
user's order history when available (personalization)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ml.recommendation import recommend_from_query, recommend_from_order_history
from services.order_service import get_user_previous_item_names


def get_recommendations(query: str, user_id: int | None = None, top_n: int = 3):
    df = recommend_from_query(query, top_n=top_n)
    if not df.empty:
        return df, "content"

    if user_id is not None:
        previous = get_user_previous_item_names(user_id)
        if previous:
            df = recommend_from_order_history(previous, top_n=top_n)
            if not df.empty:
                return df, "personalized"

    return df, "none"
