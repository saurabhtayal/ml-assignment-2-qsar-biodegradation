"""Streamlit interface for the QSAR biodegradation classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
METRICS_FILE = MODEL_DIR / "metrics.json"

st.set_page_config(page_title="QSAR Biodegradation Lab", page_icon="🧪", layout="wide")


@st.cache_data
def load_metadata() -> dict:
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))["_metadata"]


@st.cache_resource
def load_models() -> dict:
    display_names = {
        "decision_tree": "Decision Tree",
        "knearest_neighbors": "K-Nearest Neighbors",
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Naive Bayes",
        "support_vector_machine": "Support Vector Machine",
    }
    return {
        display_names.get(path.stem, path.stem): joblib.load(path)
        for path in MODEL_DIR.glob("*.joblib")
    }


def evaluate(model, frame: pd.DataFrame, metadata: dict) -> tuple[dict, list[list[int]]]:
    features = frame[metadata["feature_names"]]
    target = frame[metadata["target_column"]].map(metadata["target_mapping"])
    if target.isna().any():
        raise ValueError("Target values must be NRB or RB.")
    predictions = model.predict(features)
    scores = model.predict_proba(features)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, scores),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1 Score": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }
    return metrics, confusion_matrix(target, predictions)


st.title("QSAR biodegradation classifier laboratory")
st.markdown(
    "Predict whether a compound is **readily biodegradable (RB)** or "
    "**not readily biodegradable (NRB)** from 41 molecular descriptors."
)
metadata = load_metadata()
models = load_models()
st.markdown(
    "Sample test data is available in the GitHub repository at "
    "`data/test_data.csv`: "
    "[download sample test data](https://github.com/saurabhtayal/"
    "ml-assignment-2-qsar-biodegradation/blob/main/data/test_data.csv)."
)
with st.form("evaluation_form"):
    uploaded = st.file_uploader(
        "Upload labelled test data as CSV", type="csv",
        help="Include all 41 descriptor columns and a target column containing NRB or RB."
    )
    selected = st.selectbox("Choose a classifier", list(models))
    submitted = st.form_submit_button("Run evaluation", type="primary")

if not submitted:
    st.info("Upload a labelled CSV file and click **Run evaluation** to see the results.")
elif uploaded is None:
    st.warning("Please upload a labelled CSV file before running the evaluation.")
else:
    try:
        frame = pd.read_csv(uploaded)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the uploaded CSV file: {error}")
    else:
        st.caption(
            f"Evaluating {len(frame):,} labelled compounds; models are loaded, not retrained."
        )
        try:
            metrics, matrix = evaluate(models[selected], frame, metadata)
        except (KeyError, ValueError) as error:
            st.error(str(error))
        else:
            columns = st.columns(6)
            for column, (name, value) in zip(columns, metrics.items()):
                column.metric(name, f"{value:.3f}")
            left, right = st.columns((1, 1))
            with left:
                st.subheader("Confusion matrix")
                chart = ConfusionMatrixDisplay(matrix, display_labels=["NRB", "RB"])
                st.pyplot(chart.plot(cmap="Blues", values_format="d").figure_, clear_figure=True)
            with right:
                st.subheader("Prediction summary")
                predictions = models[selected].predict(frame[metadata["feature_names"]])
                summary = pd.Series(predictions).map({0: "NRB", 1: "RB"}).value_counts()
                st.bar_chart(summary)
                st.dataframe(frame.head(10), use_container_width=True)
