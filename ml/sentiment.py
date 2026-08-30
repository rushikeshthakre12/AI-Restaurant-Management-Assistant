"""
Sentiment analysis with basic aspect detection (food vs service), trained
on data/reviews.csv.

Approach: a small labeled lexicon-based rule scorer is used to *derive*
sentiment labels for the sample reviews (since reviews.csv only has star
ratings, not sentiment labels), and a real Multinomial Naive Bayes
classifier is then trained on (review text -> derived label) so there is an
actual trained ML model backing predictions, not just the lexicon at
inference time. This mirrors how a real project would bootstrap labels
before training a classifier.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nlp.preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
REVIEWS_PATH = BASE_DIR / "data" / "reviews.csv"
MODEL_DIR = BASE_DIR / "ml" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_POSITIVE_WORDS = {
    "excellent", "amazing", "delicious", "great", "loved", "love", "best",
    "outstanding", "fantastic", "perfectly", "perfect", "fresh", "crispy",
    "juicy", "wonderful", "friendly", "quick", "tasty", "refreshing",
    "recommend", "good",
}
_NEGATIVE_WORDS = {
    "bad", "cold", "bland", "undercooked", "disappointing", "disappointed",
    "slow", "rude", "oily", "soggy", "worst", "poor", "off", "watered",
    "overpriced", "lacked", "barely",
}
_FOOD_WORDS = {
    "food", "taste", "tasted", "dish", "biryani", "pizza", "pasta", "curry",
    "paneer", "chicken", "dosa", "brownie", "coffee", "lassi", "rice",
    "flavor", "spice", "spicy", "ingredients",
}
_SERVICE_WORDS = {"service", "staff", "delivery", "waiter", "wait", "delay", "delayed", "order"}


def _lexicon_score(tokens: set[str]) -> str:
    pos = len(tokens & _POSITIVE_WORDS)
    neg = len(tokens & _NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def rule_based_sentiment(text: str) -> dict:
    """Aspect-level rule scoring: split the sentence on connectors like
    'but'/'although' and score food-related vs service-related clauses
    separately, then combine into an overall label."""
    lower = text.lower()
    clauses = re.split(r"\bbut\b|\balthough\b|,", lower)

    food_tokens, service_tokens = set(), set()
    for clause in clauses:
        words = set(re.findall(r"[a-z]+", clause))
        if words & _FOOD_WORDS:
            food_tokens |= words
        if words & _SERVICE_WORDS:
            service_tokens |= words

    all_words = set(re.findall(r"[a-z]+", lower))
    food_sentiment = _lexicon_score(food_tokens) if food_tokens else None
    service_sentiment = _lexicon_score(service_tokens) if service_tokens else None
    overall = _lexicon_score(all_words)

    if food_sentiment and service_sentiment and food_sentiment != service_sentiment:
        overall = "mixed"

    return {"food": food_sentiment, "service": service_sentiment, "overall": overall}


def train_sentiment_classifier():
    """Bootstrap labels with the rule scorer, then train a real ML
    classifier on top so predictions come from a trained model."""
    df = pd.read_csv(REVIEWS_PATH)
    df["label"] = df["comment"].apply(lambda t: rule_based_sentiment(t)["overall"])
    df["clean"] = df["comment"].apply(clean_text)

    # collapse 'mixed' into overall lexicon polarity for a clean 3-class ML task
    df["label_3class"] = df["comment"].apply(
        lambda t: _lexicon_score(set(re.findall(r"[a-z]+", t.lower())))
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label_3class"], test_size=0.25, random_state=42
    )
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Sentiment classifier test accuracy: {acc:.4f} (n_test={len(y_test)})")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(vectorizer, MODEL_DIR / "sentiment_vectorizer.joblib")
    joblib.dump(model, MODEL_DIR / "sentiment_model.joblib")
    return acc


def analyze_sentiment(text: str) -> dict:
    """Public API used by the rest of the app: aspect-level rule result,
    which is what powers the 'Food -> Positive / Service -> Negative'
    style output described in the brief."""
    return rule_based_sentiment(text)


def analyze_sentiment_ml(text: str) -> str:
    """Overall label from the trained ML classifier (used for dashboard
    stats once ml/saved_models/sentiment_model.joblib exists)."""
    vec_path = MODEL_DIR / "sentiment_vectorizer.joblib"
    model_path = MODEL_DIR / "sentiment_model.joblib"
    if not (vec_path.exists() and model_path.exists()):
        return analyze_sentiment(text)["overall"]
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    vec = vectorizer.transform([clean_text(text)])
    return model.predict(vec)[0]


if __name__ == "__main__":
    samples = [
        "The food was excellent but the service was very slow",
        "The staff was rude and the order was delayed",
        "Amazing biryani, loved every bite",
        "The pizza was cold and bland",
    ]
    for s in samples:
        print(s, "->", analyze_sentiment(s))
    print()
    train_sentiment_classifier()
