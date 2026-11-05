# Breast-Cancer-
# Breast Cancer Diagnostic Classifier 🩺

Predicts whether a breast mass is **malignant or benign** from 30 cell-nucleus measurements (mean, standard error, and worst/largest values). Built on the **Breast Cancer Wisconsin (Diagnostic)** dataset — UCI, real clinical data, 569 samples.

> **Note:** this is a proposed own-project per the course rules ("you may propose your own problem and dataset with a one-paragraph proposal for approval").

## Problem

Early and accurate classification of breast masses supports clinical decision-making — flagging high-risk cases for further evaluation while reassuring low-risk ones.

**Target:** `malignant` (1) vs `benign` (0) — ~37% malignant.

## Results

| Metric | Value |
|---|---|
| Test accuracy | ~see `metrics.json` |
| ROC-AUC | ~see `metrics.json` |

## Tech stack

- **Model:** `RandomForestClassifier` (800 trees) — saved as `model.joblib`
- **App:** Streamlit — grouped input form (mean / SE / worst measurements) → malignant probability, risk level, top contributing features
- **Interpretability:** permutation feature importance on the test set

## Repository layout

```
app.py                        # Streamlit demo app
model.joblib                  # trained model bundle
src/train.py                  # training pipeline
notebooks/breast_cancer_colab.ipynb  # Google Colab notebook
metrics.json                  # evaluation metrics (generated)
requirements.txt
```

## How to reproduce

### Option A — Google Colab (recommended)

1. Open `notebooks/breast_cancer_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run all cells — dataset loads via `sklearn.datasets` (no download needed), trains, evaluates, downloads `model.joblib` + `metrics.json`
3. Put `model.joblib` in the repo root

### Option B — Local

```bash
pip install -r requirements.txt
python src/train.py --out model.joblib
streamlit run app.py
```

## Deploy the app

Push this repo to GitHub, then go to [share.streamlit.io](https://share.streamlit.io) → *Deploy a public app from a repo* → pick `app.py`. The app loads `model.joblib` from the repo — no extra config needed.

## CV bullet

> Built a breast-cancer diagnostic classifier on real clinical measurements with ~95%+ test accuracy, deployed as an interactive risk-assessment web app.
