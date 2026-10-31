"""Evaluate saved models against a labelled CSV without fitting anything."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_FILE = ROOT / "data" / "test_data.csv"
METRICS_FILE = MODEL_DIR / "metrics.json"


def evaluate(file_path: Path = TEST_FILE) -> dict[str, dict]:
    metadata = json.loads(METRICS_FILE.read_text(encoding="utf-8"))["_metadata"]
    frame = pd.read_csv(file_path)
    expected = metadata["feature_names"] + [metadata["target_column"]]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    features = frame[metadata["feature_names"]]
    target = frame[metadata["target_column"]].map(metadata["target_mapping"])
    if target.isna().any():
        raise ValueError("Target values must be NRB or RB.")
    output = {}
    display_names = {
        "decision_tree": "Decision Tree",
        "knearest_neighbors": "K-Nearest Neighbors",
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Naive Bayes",
        "support_vector_machine": "Support Vector Machine",
    }
    for model_path in sorted(MODEL_DIR.glob("*.joblib")):
        model_name = display_names.get(model_path.stem, model_path.stem)
        model = joblib.load(model_path)
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]
        output[model_name] = {
            "accuracy": accuracy_score(target, predictions),
            "auc": roc_auc_score(target, probabilities),
            "precision": precision_score(target, predictions, zero_division=0),
            "recall": recall_score(target, predictions, zero_division=0),
            "f1": f1_score(target, predictions, zero_division=0),
            "mcc": matthews_corrcoef(target, predictions),
            "confusion_matrix": confusion_matrix(target, predictions).tolist(),
        }
    return output


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
