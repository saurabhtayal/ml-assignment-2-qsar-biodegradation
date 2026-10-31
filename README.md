# QSAR Biodegradation Classifier

Machine Learning Assignment 2 for the BITS WILP M.Tech AIML programme. This
project compares five classical binary classifiers and exposes the saved
models through a Streamlit application.

## Problem statement

Given molecular descriptors for an organic compound, predict whether it is
**readily biodegradable (RB)** or **not readily biodegradable (NRB)**. The
application accepts a labelled CSV test set, loads one of the pre-trained
models, and reports its performance without retraining on uploaded data.

## Dataset description

This project uses the **QSAR biodegradation** dataset
from the UCI Machine Learning Repository:

- Source: https://archive.ics.uci.edu/dataset/254/qsar+biodegradation
- 1,055 compounds and 41 numerical molecular-descriptor features
- Binary target: `RB` (readily biodegradable) or `NRB` (not readily
  biodegradable)
- The supplied data is stored as `data/qsar_biodegradation.csv` with
  descriptive feature headers.
- An 80/20 stratified holdout split uses `random_state=2026`. The exact
  labelled holdout used for the results is `data/test_data.csv`.

The descriptor names are intentionally neutral (`descriptor_01` through
`descriptor_41`) because the original UCI file provides descriptor values
without a header row. The original source remains the authoritative definition
of each descriptor.

## GitHub repository link

**Repository:** https://github.com/saurabhtayal/ml-assignment-2-qsar-biodegradation

**Live Streamlit application:** https://ml-assignment-2-qsar-biodegradation-onsuyyappeeggmgshgvpift.streamlit.app/

## Models used and evaluation

All models were trained on the same training partition and evaluated on the
same untouched 211-row test partition. Standard scaling is included in the
Logistic Regression, KNN, and SVM pipelines. The Decision Tree and Gaussian
Naive Bayes models use the original numerical values.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8673 | 0.8925 | 0.8308 | 0.7606 | 0.7941 | 0.6980 |
| Decision Tree | 0.7820 | 0.7501 | 0.6923 | 0.6338 | 0.6618 | 0.5025 |
| KNN | 0.8152 | 0.8570 | 0.7162 | 0.7465 | 0.7310 | 0.5906 |
| Naive Bayes | 0.6919 | 0.8335 | 0.5242 | 0.9155 | 0.6667 | 0.4742 |
| Support Vector Machine | 0.8578 | 0.8913 | 0.8361 | 0.7183 | 0.7727 | 0.6742 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall accuracy, AUC, F1, and MCC. Scaling helps it combine the many correlated descriptors into a stable linear boundary. |
| Decision Tree | Lowest AUC among the tree-based and distance-based alternatives; its single, bounded-depth tree cannot represent all descriptor interactions as effectively. |
| KNN | Balanced performance, but weaker than the linear and kernel models. Distance-based decisions are sensitive to the high-dimensional descriptor space even after scaling. |
| Naive Bayes | Highest recall, so it identifies most RB compounds, but its low precision and accuracy indicate many false positives. The conditional-independence assumption is restrictive for molecular descriptors. |
| Support Vector Machine | Very strong AUC and precision, close to Logistic Regression. Its slightly lower recall leads to a marginally lower F1 and accuracy on this split. |
| **Overall winner** | **Logistic Regression**, because it leads on the principal aggregate measures while remaining interpretable and efficient. SVM is a close second. |

## Streamlit application

The app includes:

1. A CSV uploader for labelled test data.
2. A classifier dropdown for all five saved models.
3. A **Run evaluation** button that evaluates the uploaded file only after
   explicit submission.
4. Accuracy, AUC, precision, recall, F1, and MCC metric cards.
5. A confusion matrix and prediction-summary chart.
6. Validation for missing columns and unsupported target labels.

Uploaded data must contain all 41 `descriptor_01` ... `descriptor_41` columns
and a `target` column containing `NRB` or `RB`. The application never calls
`.fit()`; it only loads the committed `.joblib` artifacts and evaluates the
uploaded rows.
