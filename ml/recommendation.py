"""
Content-based food recommendation system.

Builds a text profile per menu item from category + description +
ingredients + veg/spicy flags, vectorizes with TF-IDF, and ranks items by
cosine similarity -- either against a free-text query ("suggest something
spicy under 300") or against another item ("what's similar to Paneer
Tikka"). Also implements simple personalization from order history.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nlp.ner import extract_money, extract_food_items

BASE_DIR = Path(__file__).resolve().parent.parent
MENU_PATH = BASE_DIR / "data" / "menu.csv"


def _load_menu() -> pd.DataFrame:
    df = pd.read_csv(MENU_PATH)
    df["profile_text"] = (
        df["category"] + " " + df["description"] + " " + df["ingredients"].str.replace(";", " ")
        + " " + df["vegetarian"].map({1: "vegetarian", 0: "non-vegetarian"})
        + " " + df["spicy"].map({1: "spicy", 0: "mild"})
    )
    return df


_MENU_DF = _load_menu()
_VECTORIZER = TfidfVectorizer(stop_words="english")
_ITEM_VECTORS = _VECTORIZER.fit_transform(_MENU_DF["profile_text"])


def recommend_from_query(query: str, top_n: int = 3) -> pd.DataFrame:
    """Content-based recommendation from free text, e.g.
    'suggest vegetarian spicy food under 300'."""
    df = _MENU_DF.copy()

    lower = query.lower()
    if "vegetarian" in lower or "veg " in lower or lower.startswith("veg"):
        df = df[df["vegetarian"] == 1]
    if "non-vegetarian" in lower or "non veg" in lower or "meat" in lower or "chicken" in lower or "fish" in lower:
        df = df[df["vegetarian"] == 0]
    if "spicy" in lower:
        df = df[df["spicy"] == 1]
    if "mild" in lower or "not spicy" in lower or "not too spicy" in lower:
        df = df[df["spicy"] == 0]

    price_caps = extract_money(query)
    if price_caps:
        df = df[df["price"] <= float(price_caps[0])]

    if df.empty:
        return df

    df_vectors = _VECTORIZER.transform(df["profile_text"])
    query_vector = _VECTORIZER.transform([query])
    sims = cosine_similarity(query_vector, df_vectors).flatten()
    df = df.assign(similarity=sims).sort_values(["similarity", "rating"], ascending=False)
    return df.head(top_n)[["item_id", "name", "price", "rating", "similarity"]]


def recommend_similar_items(item_name: str, top_n: int = 3) -> pd.DataFrame:
    """Item-to-item content-based recommendation, e.g. 'similar to Paneer Tikka'."""
    matches = _MENU_DF[_MENU_DF["name"].str.lower() == item_name.lower()]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    sims = cosine_similarity(_ITEM_VECTORS[idx], _ITEM_VECTORS).flatten()
    df = _MENU_DF.assign(similarity=sims).drop(index=idx).sort_values("similarity", ascending=False)
    return df.head(top_n)[["item_id", "name", "price", "rating", "similarity"]]


def recommend_from_order_history(previous_item_names: list[str], top_n: int = 3) -> pd.DataFrame:
    """Simple personalization: average the TF-IDF profile of a user's past
    orders, then rank the rest of the menu by similarity to that average.
    This is a straightforward average-profile approach, not a
    sophisticated collaborative-filtering model."""
    matches = _MENU_DF[_MENU_DF["name"].str.lower().isin([n.lower() for n in previous_item_names])]
    if matches.empty:
        return pd.DataFrame()
    profile_vector = _ITEM_VECTORS[matches.index].mean(axis=0)
    import numpy as np
    profile_vector = np.asarray(profile_vector)
    sims = cosine_similarity(profile_vector, _ITEM_VECTORS).flatten()
    df = _MENU_DF.assign(similarity=sims).drop(index=matches.index).sort_values("similarity", ascending=False)
    return df.head(top_n)[["item_id", "name", "price", "rating", "similarity"]]


if __name__ == "__main__":
    print("Query: 'suggest vegetarian spicy food under 300'")
    print(recommend_from_query("suggest vegetarian spicy food under 300"))

    print("\nSimilar to 'Paneer Tikka':")
    print(recommend_similar_items("Paneer Tikka"))

    print("\nPersonalized (past orders: Paneer Pizza, Spicy Veg Pasta):")
    print(recommend_from_order_history(["Paneer Pizza", "Spicy Veg Pasta"]))
