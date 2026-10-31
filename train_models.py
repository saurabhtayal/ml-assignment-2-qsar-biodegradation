"""Train and save the five classifiers required for ML Assignment 2."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
SOURCE_FILE = DATA_DIR / "qsar_biodegradation.csv"
TEST_FILE = DATA_DIR / "test_data.csv"
METRICS_FILE = MODEL_DIR / "metrics.json"
RANDOM_STATE = 2026

MODEL_FACTORIES = {
    "Logistic Regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=3, random_state=RANDOM_STATE
    ),
    "K-Nearest Neighbors": lambda: make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=11)
    ),
    "Naive Bayes": GaussianNB,
    "Support Vector Machine": lambda: make_pipeline(
        StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    ),
}


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(SOURCE_FILE)
    return frame.drop(columns="target"), frame["target"]


def score_model(model, features: pd.DataFrame, target: pd.Series) -> dict:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": accuracy_score(target, predictions),
        "auc": roc_auc_score(target, probabilities),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "mcc": matthews_corrcoef(target, predictions),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
    }


def main() -> None:
    features, raw_target = load_dataset()
    target_mapping = {"NRB": 0, "RB": 1}
    target = raw_target.map(target_mapping)
    if target.isna().any():
        raise ValueError("The dataset contains a target label outside NRB/RB.")
    x_train, x_test, y_train, y_test = train_test_split(
        features, target.astype(int), test_size=0.2, stratify=target,
        random_state=RANDOM_STATE
    )
    MODEL_DIR.mkdir(exist_ok=True)
    results: dict[str, dict] = {}
    for display_name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(x_train, y_train)
        file_name = display_name.lower().replace(" ", "_").replace("-", "") + ".joblib"
        joblib.dump(model, MODEL_DIR / file_name)
        results[display_name] = score_model(model, x_test, y_test)
    test_output = x_test.copy()
    test_output["target"] = y_test.map({0: "NRB", 1: "RB"})
    test_output.to_csv(TEST_FILE, index=False)
    results["_metadata"] = {
        "feature_names": list(features.columns),
        "target_column": "target",
        "target_mapping": target_mapping,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "sample_count": len(features),
        "feature_count": features.shape[1],
    }
    METRICS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
