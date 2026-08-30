"""Print the saved evaluation report from ml/train_intent.py (accuracy,
precision, recall, F1, per-class breakdown) — for viva / documentation use."""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = BASE_DIR / "ml" / "saved_models" / "intent_eval_report.json"


def show_report():
    if not REPORT_PATH.exists():
        print("No report found. Run: python -m ml.train_intent")
        return
    with open(REPORT_PATH) as f:
        report = json.load(f)
    for model_name, metrics in report.items():
        print(f"\n===== {model_name} =====")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
        print("\nPer-class report:")
        print(metrics["classification_report"])


if __name__ == "__main__":
    show_report()
