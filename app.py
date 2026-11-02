"""Streamlit interface for the QSAR biodegradation classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
METRICS_FILE = MODEL_DIR / "metrics.json"
SAMPLE_FILE = ROOT / "data" / "test_data.csv"

st.set_page_config(
    page_title="BioDecode | QSAR Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #16231f;
        --muted: #62726c;
        --paper: #f5f7f2;
        --card: #ffffff;
        --mint: #c9f2d5;
        --green: #0e6b50;
        --orange: #e8794f;
        --line: #dce5dc;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245,247,242,0.86); }
    [data-testid="stSidebar"] {
        background: #19362f;
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #eaf5e9; }
    [data-testid="stSidebar"] .stCaption { color: #a9c7b5; }
    .block-container { max-width: 1280px; padding: 2.4rem 3.2rem 4rem; }
    h1, h2, h3, p, label, .stMarkdown { font-family: 'Space Grotesk', sans-serif; }
    h1 { letter-spacing: -0.055em; font-size: clamp(2.4rem, 5vw, 4.7rem); line-height: 0.96; }
    h2 { letter-spacing: -0.035em; }
    .mono, code { font-family: 'DM Mono', monospace !important; }
    .eyebrow {
        color: var(--orange); font-family: 'DM Mono', monospace; font-size: 0.76rem;
        font-weight: 500; letter-spacing: .16em; text-transform: uppercase; margin-bottom: .7rem;
    }
    .hero {
        background: linear-gradient(120deg, #dff7e3 0%, #f7f8ef 52%, #ffe1ce 100%);
        border: 1px solid #d5e7d5; border-radius: 24px; padding: 2.5rem 2.8rem;
        margin-bottom: 1.7rem; position: relative; overflow: hidden;
    }
    .hero:after {
        content: "01"; color: rgba(14,107,80,.11); font: 700 11rem/1 'DM Mono', monospace;
        position: absolute; right: 2rem; bottom: -3rem; transform: rotate(-8deg);
    }
    .hero p { color: #4a6257; font-size: 1.05rem; max-width: 650px; margin-bottom: 0; }
    .section-label {
        color: var(--green); font: 500 .72rem 'DM Mono', monospace; letter-spacing: .14em;
        text-transform: uppercase; margin: 2rem 0 .75rem;
    }
    .info-card, .result-card {
        background: var(--card); border: 1px solid var(--line); border-radius: 16px;
        padding: 1.1rem 1.25rem; height: 100%;
    }
    .info-card strong { display: block; color: var(--green); font-size: 1.55rem; margin-bottom: .2rem; }
    .info-card span { color: var(--muted); font-size: .86rem; }
    .result-card { border-top: 4px solid var(--green); }
    .result-card .value { font: 600 1.45rem 'DM Mono', monospace; color: var(--ink); }
    .result-card .label { color: var(--muted); font-size: .78rem; margin-top: .25rem; }
    .model-note {
        background: #edf7ee; border-left: 3px solid var(--green); border-radius: 4px;
        padding: .8rem 1rem; color: #416155; font-size: .9rem;
    }
    div[data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { color: var(--ink); font-family: 'DM Mono', monospace; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px; font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    }
    .stForm { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def load_benchmarks() -> dict:
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return {name: values for name, values in metrics.items() if name != "_metadata"}


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


metadata = load_metadata()
models = load_models()
benchmarks = load_benchmarks()

with st.sidebar:
    st.markdown("## 🧬 BioDecode")
    st.caption("QSAR biodegradation intelligence")
    st.divider()
    st.markdown("### Workflow")
    st.markdown("**01**  Load labelled descriptors\n\n**02**  Select a model\n\n**03**  Inspect the verdict")
    st.divider()
    st.markdown("### Dataset profile")
    st.markdown(
        f'<div class="mono" style="font-size:.82rem;line-height:1.9">'
        f'COMPOUNDS&nbsp;&nbsp; {metadata["sample_count"]:,}<br>'
        f'DESCRIPTORS&nbsp; {metadata["feature_count"]}<br>'
        f'CLASSES&nbsp;&nbsp;&nbsp;&nbsp; RB / NRB<br>'
        f'HOLDOUT&nbsp;&nbsp;&nbsp;&nbsp; {metadata["test_size"]:.0%}</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Models are pre-trained on the fixed 2026 holdout split. Uploads are evaluated, never retrained.")

st.markdown('<div class="eyebrow">Molecular intelligence / evaluation console</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>Read the molecule<br><em>before it breaks down.</em></h1>'
    '<p>A focused evaluation workspace for predicting ready biodegradation from 41 molecular descriptors.</p></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">At a glance</div>', unsafe_allow_html=True)
overview = st.columns(3)
overview[0].markdown('<div class="info-card"><strong>41</strong><span>molecular descriptors per compound</span></div>', unsafe_allow_html=True)
overview[1].markdown('<div class="info-card"><strong>5</strong><span>pre-trained classifiers available</span></div>', unsafe_allow_html=True)
overview[2].markdown('<div class="info-card"><strong>0.867</strong><span>best holdout accuracy · Logistic Regression</span></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Run an evaluation</div>', unsafe_allow_html=True)
with st.form("evaluation_form"):
    uploaded = st.file_uploader(
        "Drop a labelled descriptor CSV",
        type="csv",
        help="Include all 41 descriptor columns and a target column containing NRB or RB.",
    )
    form_left, form_right = st.columns((1.3, 1))
    with form_left:
        selected = st.selectbox("Classifier lens", list(models), label_visibility="visible")
    with form_right:
        submitted = st.form_submit_button("Run evaluation  →", type="primary", use_container_width=True)

sample_bytes = SAMPLE_FILE.read_bytes()
st.download_button(
    "↓ Download a correctly formatted sample CSV",
    data=sample_bytes,
    file_name="qsar_test_data.csv",
    mime="text/csv",
)

if not submitted:
    st.markdown(
        '<div class="model-note">Start with the sample CSV if you want to explore the expected schema. '
        'The target column should contain <strong>RB</strong> or <strong>NRB</strong>.</div>',
        unsafe_allow_html=True,
    )
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
            st.markdown('<div class="section-label">Evaluation readout</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="model-note">Analysed <strong>{len(frame):,} compounds</strong> with '
                f'<strong>{selected}</strong>. The metrics below are calculated on your uploaded labels.</div>',
                unsafe_allow_html=True,
            )
            columns = st.columns(6)
            for column, (name, value) in zip(columns, metrics.items()):
                column.metric(name, f"{value:.3f}")
            st.markdown(
                f'<div class="model-note">Reference holdout accuracy for {selected}: '
                f'<strong>{benchmarks[selected]["accuracy"]:.3f}</strong>. '
                "Use the comparison below to spot dataset shift.</div>",
                unsafe_allow_html=True,
            )
            left, right = st.columns((1, 1))
            with left:
                st.markdown('<div class="section-label">Where the model is right</div>', unsafe_allow_html=True)
                st.subheader("Confusion matrix")
                chart = ConfusionMatrixDisplay(matrix, display_labels=["NRB", "RB"])
                figure, axis = plt.subplots(figsize=(5.4, 4))
                chart.plot(ax=axis, cmap="GnBu", values_format="d", colorbar=False)
                axis.set_xlabel("Predicted class")
                axis.set_ylabel("Actual class")
                figure.tight_layout()
                st.pyplot(figure, clear_figure=True)
            with right:
                st.markdown('<div class="section-label">How predictions split</div>', unsafe_allow_html=True)
                st.subheader("Prediction summary")
                predictions = models[selected].predict(frame[metadata["feature_names"]])
                summary = pd.Series(predictions).map({0: "NRB", 1: "RB"}).value_counts()
                st.bar_chart(summary, color="#0e6b50")
                st.dataframe(frame.head(10), use_container_width=True)
