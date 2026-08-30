"""Load the trained intent model and classify new text."""
import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nlp.preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml" / "saved_models"

_vectorizer = None
_model = None


def _load():
    global _vectorizer, _model
    if _vectorizer is None or _model is None:
        _vectorizer = joblib.load(MODEL_DIR / "intent_vectorizer.joblib")
        _model = joblib.load(MODEL_DIR / "intent_model.joblib")
    return _vectorizer, _model


def predict_intent(text: str) -> dict:
    vectorizer, model = _load()
    vec = vectorizer.transform([clean_text(text)])
    pred = model.predict(vec)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        classes = model.classes_
        proba = float(max(probs))
    return {"intent": pred, "confidence": proba}


if __name__ == "__main__":
    tests = [
        "book a table for 4 at 8 pm",
        "what do you have on the menu",
        "cancel my reservation please",
        "the food was cold and service was rude",
        "what is the capital of india",
    ]
    for t in tests:
        print(t, "->", predict_intent(t))
