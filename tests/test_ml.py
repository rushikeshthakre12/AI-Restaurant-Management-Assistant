import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.predict_intent import predict_intent
from ml.sentiment import analyze_sentiment
from ml.recommendation import recommend_from_query


def test_predict_intent_booking():
    result = predict_intent("book a table for 4 at 8 pm")
    assert result["intent"] == "table_booking"


def test_predict_intent_greeting():
    result = predict_intent("hello there")
    assert result["intent"] == "greeting"


def test_sentiment_positive():
    result = analyze_sentiment("The food was excellent and delicious")
    assert result["overall"] == "positive"


def test_sentiment_negative():
    result = analyze_sentiment("The food was cold and bland")
    assert result["overall"] == "negative"


def test_sentiment_mixed_aspects():
    result = analyze_sentiment("The food was excellent but the service was slow")
    assert result["food"] == "positive"
    assert result["service"] == "negative"


def test_recommendation_respects_price_cap():
    df = recommend_from_query("suggest vegetarian food under 200")
    assert not df.empty
    assert (df["price"] <= 200).all()
