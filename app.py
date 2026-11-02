"""Streamlit interface for the QSAR biodegradation classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

    /* Page background lives on the app shell only. Putting it on p/label/markdown
       containers (as before) painted a visible box behind every text block. */
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245,247,242,0.86); }

    /* Main-area text colour is scoped to .block-container so it can't leak into the sidebar. */
    .block-container, .block-container p, .block-container label,
    .block-container h1, .block-container h2, .block-container h3,
    .block-container [data-testid="stMarkdownContainer"] { color: var(--ink); }
    .block-container { max-width: 1280px; padding: 2.4rem 3.2rem 4rem; }
    .block-container h1, .block-container h2, .block-container h3,
    .block-container p, .block-container label { font-family: 'Space Grotesk', sans-serif; }
    .block-container h1 { letter-spacing: -0.055em; font-size: clamp(2.2rem, 5vw, 4.7rem); line-height: 0.98; padding: 0; }
    .block-container h2 { letter-spacing: -0.035em; }
    .mono, code { font-family: 'DM Mono', monospace !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #19362f; border-right: 0; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] label { color: #eaf5e9; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #a9c7b5; }
    [data-testid="stSidebar"] hr { border-color: rgba(234,245,233,.18); }

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
        pointer-events: none;
    }
    .hero h1, .hero p { position: relative; z-index: 1; }
    .block-container .hero p { color: #4a6257; font-size: 1.05rem; max-width: 650px; margin-bottom: 0; }
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
        padding: .8rem 1rem; color: #416155; font-size: .9rem; margin: .6rem 0;
    }
    .block-container .model-note, .block-container .model-note strong { color: #416155; }

    div[data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem;
    }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p { color: var(--muted); }
    div[data-testid="stMetricValue"] {
        color: var(--ink); font-family: 'DM Mono', monospace; font-size: 1.5rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 9px; font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    }
    [data-testid="stForm"] {
        background: var(--card); border: 1px solid var(--line); border-radius: 18px;
        padding: 1.35rem 1.5rem 1.5rem;
    }
    [data-testid="stForm"] label, [data-testid="stForm"] [data-testid="stWidgetLabel"] p {
        color: var(--ink) !important; font-weight: 600;
    }
    [data-testid="stFileUploader"] section {
        background: #f7faf6; border: 1px dashed #9bb9a6; border-radius: 12px;
        min-height: 126px; padding: 1rem;
    }
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section span { color: var(--muted) !important; }
    [data-testid="stFileUploader"] button {
        background: var(--green) !important; border: 0 !important; color: #fff !important;
        min-height: 2.45rem; padding: 0 .95rem; font-weight: 600;
    }
    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p { color: #fff !important; }
    [data-testid="stSelectbox"] > div > div {
        background: #f7faf6; border-color: #9bb9a6; color: var(--ink) !important;
        min-height: 2.45rem;
    }
    [data-testid="stSelectbox"] input, [data-testid="stSelectbox"] span {
        color: var(--ink) !important;
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] { margin-top: 1rem; }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
        background: var(--green) !important; border: 1px solid var(--green) !important;
        color: #fff !important; min-height: 2.9rem; width: 100%;
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button p,
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button span { color: #fff !important; }
    .stDownloadButton { margin-top: .9rem; }
    .stDownloadButton button {
        background: transparent !important; border: 1px solid var(--green) !important;
        color: var(--green) !important; min-height: 2.5rem;
    }
    .stDownloadButton button p, .stDownloadButton button span { color: var(--green) !important; }

    /* Keyboard focus */
    .stApp button:focus-visible, .stApp input:focus-visible {
        outline: 2px solid var(--orange) !important; outline-offset: 2px;
    }

    /* Tablet / mobile */
    @media (max-width: 900px) {
        .block-container { padding: 1.4rem 1.1rem 3rem; }
        .hero { padding: 1.6rem 1.3rem; border-radius: 18px; }
        .hero:after { display: none; }
        [data-testid="stForm"] { padding: 1rem; }
        div[data-testid="stMetricValue"] { font-size: 1.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DISPLAY_NAMES = {
    "decision_tree": "Decision Tree",
    "knearest_neighbors": "K-Nearest Neighbors",
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "support_vector_machine": "Support Vector Machine",
}
CLASS_LABELS = ["NRB", "RB"]


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


@st.cache_resource
def load_models() -> dict:
    return {
        DISPLAY_NAMES.get(path.stem, path.stem): joblib.load(path)
        for path in sorted(MODEL_DIR.glob("*.joblib"))
    }


@st.cache_data
def load_sample() -> bytes:
    return SAMPLE_FILE.read_bytes()


def benchmark_for(benchmarks: dict, name: str) -> dict | None:
    """Look a model up by display name or by its file-stem key."""
    if name in benchmarks:
        return benchmarks[name]
    for stem, label in DISPLAY_NAMES.items():
        if label == name and stem in benchmarks:
            return benchmarks[stem]
    return None


def evaluate(model, frame: pd.DataFrame, metadata: dict):
    missing = [
        c
        for c in [*metadata["feature_names"], metadata["target_column"]]
        if c not in frame.columns
    ]
    if missing:
        shown = ", ".join(missing[:6]) + (
            f" and {len(missing) - 6} more" if len(missing) > 6 else ""
        )
        raise ValueError(f"Missing columns in the uploaded CSV: {shown}.")
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
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    return metrics, matrix, predictions, scores


try:
    all_metrics = load_metrics()
    metadata = all_metrics["_metadata"]
    benchmarks = {k: v for k, v in all_metrics.items() if k != "_metadata"}
    models = load_models()
except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
    st.error(f"Could not load model files from `{MODEL_DIR}`: {error}")
    st.stop()

if not models:
    st.error(f"No .joblib models found in `{MODEL_DIR}`.")
    st.stop()

# Best holdout model, derived instead of hard-coded.
best_name, best_acc = None, None
for model_name in models:
    entry = benchmark_for(benchmarks, model_name)
    if (
        entry
        and "accuracy" in entry
        and (best_acc is None or entry["accuracy"] > best_acc)
    ):
        best_name, best_acc = model_name, entry["accuracy"]

with st.sidebar:
    st.markdown("## 🧬 BioDecode")
    st.caption("QSAR biodegradation intelligence")
    st.divider()
    st.markdown("### Workflow")
    st.markdown(
        "**01**  Load labelled descriptors\n\n**02**  Select a model\n\n**03**  Inspect the verdict"
    )
    st.divider()
    st.markdown("### Dataset profile")
    st.markdown(
        f'<div class="mono" style="font-size:.82rem;line-height:1.9">'
        f'COMPOUNDS&nbsp;&nbsp; {metadata["sample_count"]:,}<br>'
        f'DESCRIPTORS&nbsp; {metadata["feature_count"]}<br>'
        f"CLASSES&nbsp;&nbsp;&nbsp;&nbsp; RB / NRB<br>"
        f'HOLDOUT&nbsp;&nbsp;&nbsp;&nbsp; {metadata["test_size"]:.0%}</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "Models are pre-trained on the fixed 2026 holdout split. Uploads are evaluated, never retrained."
    )

st.markdown(
    '<div class="hero"><div class="eyebrow">Molecular intelligence / evaluation console</div>'
    "<h1>Read the molecule<br><em>before it breaks down.</em></h1>"
    f'<p>A focused evaluation workspace for predicting ready biodegradation from {metadata["feature_count"]} molecular descriptors.</p></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">At a glance</div>', unsafe_allow_html=True)
overview = st.columns(3)
overview[0].markdown(
    f'<div class="info-card"><strong>{metadata["feature_count"]}</strong>'
    "<span>molecular descriptors per compound</span></div>",
    unsafe_allow_html=True,
)
overview[1].markdown(
    f'<div class="info-card"><strong>{len(models)}</strong>'
    "<span>pre-trained classifiers available</span></div>",
    unsafe_allow_html=True,
)
if best_acc is not None:
    overview[2].markdown(
        f'<div class="info-card"><strong>{best_acc:.3f}</strong>'
        f"<span>best holdout accuracy · {best_name}</span></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="section-label">Run an evaluation</div>', unsafe_allow_html=True
)
with st.form("evaluation_form"):
    form_left, form_right = st.columns((1.15, 0.85), gap="large")
    with form_left:
        uploaded = st.file_uploader(
            "Drop a labelled descriptor CSV",
            type="csv",
            help=f"Include all {metadata['feature_count']} descriptor columns and a target column containing NRB or RB.",
        )
    with form_right:
        selected = st.selectbox("Classifier lens", list(models))
        submitted = st.form_submit_button("Run evaluation", type="primary")

st.download_button(
    "Download a correctly formatted sample CSV",
    data=load_sample(),
    file_name="qsar_test_data.csv",
    mime="text/csv",
)

# Keep the last run in session state so unrelated reruns (e.g. clicking the
# download button) don't wipe the results off the screen.
if submitted:
    if uploaded is None:
        st.session_state.pop("run", None)
        st.warning("Please upload a labelled CSV file before running the evaluation.")
    else:
        try:
            st.session_state["run"] = {
                "frame": pd.read_csv(uploaded),
                "selected": selected,
            }
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            st.session_state.pop("run", None)
            st.error(f"Could not read the uploaded CSV file: {error}")

run = st.session_state.get("run")

if run is None:
    if not submitted:
        st.markdown(
            '<div class="model-note">Start with the sample CSV if you want to explore the expected schema. '
            "The target column should contain <strong>RB</strong> or <strong>NRB</strong>.</div>",
            unsafe_allow_html=True,
        )
else:
    frame, selected = run["frame"], run["selected"]
    st.caption(
        f"Evaluating {len(frame):,} labelled compounds; models are loaded, not retrained."
    )
    try:
        metrics, matrix, predictions, scores = evaluate(
            models[selected], frame, metadata
        )
    except ValueError as error:
        st.error(str(error))
    else:
        st.markdown(
            '<div class="section-label">Evaluation readout</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="model-note">Analysed <strong>{len(frame):,} compounds</strong> with '
            f"<strong>{selected}</strong>. The metrics below are calculated on your uploaded labels.</div>",
            unsafe_allow_html=True,
        )
        for column, (name, value) in zip(st.columns(6), metrics.items()):
            column.metric(name, f"{value:.3f}")

        reference = benchmark_for(benchmarks, selected)
        if reference and "accuracy" in reference:
            delta = metrics["Accuracy"] - reference["accuracy"]
            st.markdown(
                f'<div class="model-note">Reference holdout accuracy for {selected}: '
                f'<strong>{reference["accuracy"]:.3f}</strong> '
                f"(your upload: {delta:+.3f}). Use the comparison to spot dataset shift.</div>",
                unsafe_allow_html=True,
            )

        left, right = st.columns((1, 1), gap="large")
        with left:
            st.markdown(
                '<div class="section-label">Where the model is right</div>',
                unsafe_allow_html=True,
            )
            st.subheader("Confusion matrix")
            figure, axis = plt.subplots(figsize=(5.4, 4))
            figure.patch.set_alpha(0)
            ConfusionMatrixDisplay(matrix, display_labels=CLASS_LABELS).plot(
                ax=axis, cmap="GnBu", values_format="d", colorbar=False
            )
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("Actual class")
            figure.tight_layout()
            st.pyplot(figure, clear_figure=True)
        with right:
            st.markdown(
                '<div class="section-label">How predictions split</div>',
                unsafe_allow_html=True,
            )
            st.subheader("Prediction summary")
            summary = (
                pd.Series(predictions)
                .map({0: "NRB", 1: "RB"})
                .value_counts()
                .reindex(CLASS_LABELS, fill_value=0)
            )
            st.bar_chart(summary, color="#0e6b50")

        st.markdown(
            '<div class="section-label">Compound-level output</div>',
            unsafe_allow_html=True,
        )
        preview = frame.head(10).copy()
        preview.insert(
            0, "Predicted", pd.Series(predictions[:10]).map({0: "NRB", 1: "RB"}).values
        )
        preview.insert(1, "P(RB)", scores[:10].round(3))
        st.dataframe(preview, width="stretch")
