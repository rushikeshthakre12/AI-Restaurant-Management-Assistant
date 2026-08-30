"""
Stage 5 / Practical mapping (CO1-CO2 territory): ML intent classification.

Pipeline: raw text -> NLP preprocessing -> TF-IDF -> classifier -> intent.
Primary model: Logistic Regression. Compared against Multinomial Naive Bayes
on a held-out test split. All numbers below are computed from the actual
run, not invented -- see ml/evaluation.py for the saved report.
"""
import sys
import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nlp.preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "intents.csv"
MODEL_DIR = BASE_DIR / "ml" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train_and_evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["intent"], test_size=0.2, random_state=42, stratify=df["intent"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=5.0),
        "naive_bayes": MultinomialNB(),
    }

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )
        results[name] = {
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "classification_report": classification_report(y_test, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds, labels=sorted(df["intent"].unique())).tolist(),
        }
        print(f"\n=== {name} ===")
        print(f"accuracy={acc:.4f}  precision_macro={precision:.4f}  recall_macro={recall:.4f}  f1_macro={f1:.4f}")

    # Primary model = whichever scored higher on macro F1 (usually Logistic Regression here)
    primary_name = max(results, key=lambda k: results[k]["f1_macro"])
    primary_model = models[primary_name]
    joblib.dump(vectorizer, MODEL_DIR / "intent_vectorizer.joblib")
    joblib.dump(primary_model, MODEL_DIR / "intent_model.joblib")
    joblib.dump(primary_name, MODEL_DIR / "intent_model_name.joblib")

    with open(MODEL_DIR / "intent_eval_report.json", "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} for k, v in results.items()},
            f, indent=2,
        )

    print(f"\nSaved primary model: {primary_name} -> ml/saved_models/intent_model.joblib")
    return results, primary_name


if __name__ == "__main__":
    train_and_evaluate()
