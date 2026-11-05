"""
Breast Cancer Diagnostic Classifier — Streamlit demo app.

Run locally:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Diagnostic", page_icon="🩺", layout="wide"
)


@st.cache_resource
def load_bundle():
    return joblib.load("model.joblib")


bundle = load_bundle()
model = bundle["model"]
FEATURES = bundle["feature_names"]
RANGES = bundle["feature_ranges"]
IMPORTANCES = bundle["feature_importances"]
METRICS = bundle["metrics"]

# prettier labels for the 30 features
LABELS = {
    "mean radius": "Mean cell nucleus radius",
    "mean texture": "Mean cell nucleus texture",
    "mean perimeter": "Mean cell nucleus perimeter",
    "mean area": "Mean cell nucleus area",
    "mean smoothness": "Mean smoothness",
    "mean compactness": "Mean compactness",
    "mean concavity": "Mean concavity",
    "mean concave points": "Mean concave points",
    "mean symmetry": "Mean symmetry",
    "mean fractal dimension": "Mean fractal dimension",
    "radius error": "Radius standard error",
    "texture error": "Texture standard error",
    "perimeter error": "Perimeter standard error",
    "area error": "Area standard error",
    "smoothness error": "Smoothness standard error",
    "compactness error": "Compactness standard error",
    "concavity error": "Concavity standard error",
    "concave points error": "Concave points standard error",
    "symmetry error": "Symmetry standard error",
    "fractal dimension error": "Fractal dimension standard error",
    "worst radius": "Largest nucleus radius",
    "worst texture": "Largest nucleus texture",
    "worst perimeter": "Largest nucleus perimeter",
    "worst area": "Largest nucleus area",
    "worst smoothness": "Worst smoothness",
    "worst compactness": "Worst compactness",
    "worst concavity": "Worst concavity",
    "worst concave points": "Worst concave points",
    "worst symmetry": "Worst symmetry",
    "worst fractal dimension": "Worst fractal dimension",
}

GROUPS = {
    "Mean measurements": [f for f in FEATURES if f.startswith("mean ")],
    "Standard errors": [f for f in FEATURES if f.endswith(" error")],
    "Worst (largest) measurements": [f for f in FEATURES if f.startswith("worst ")],
}

st.title("🩺 Breast Cancer Diagnostic Classifier")
st.caption(
    "Predicts whether a breast mass is **malignant** from cell-nucleus measurements. "
    f"Random Forest on {METRICS['train_rows']:,} training samples "
    f"(test accuracy {METRICS['accuracy']:.1%}, ROC-AUC {METRICS['roc_auc']:.3f}). "
    "*Educational demo — not a medical device.*"
)

values = {}
for group_name, cols in GROUPS.items():
    with st.expander(group_name, expanded=(group_name == "Mean measurements")):
        gcols = st.columns(2)
        for i, feat in enumerate(cols):
            lo, hi = RANGES[feat]
            default = float((lo + hi) / 2)
            with gcols[i % 2]:
                values[feat] = st.number_input(
                    LABELS.get(feat, feat),
                    min_value=float(lo),
                    max_value=float(hi),
                    value=default,
                    step=(hi - lo) / 200 or 0.01,
                    format="%.4f",
                )

X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
prob = float(model.predict_proba(X)[0, 1])

st.subheader("Prediction")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Malignant probability", f"{prob:.1%}")
with col2:
    st.progress(prob)

if prob >= 0.5:
    st.error("🔴 High risk — malignant indicators detected. Further clinical evaluation recommended.")
elif prob >= 0.3:
    st.warning("🟡 Intermediate risk — additional tests advised.")
else:
    st.success("🟢 Low risk — features look benign.")

st.subheader("Top features driving the prediction")
imp_df = pd.DataFrame(
    list(IMPORTANCES.items()), columns=["feature", "importance"]
).head(10)
imp_df["feature"] = imp_df["feature"].map(lambda f: LABELS.get(f, f))
st.bar_chart(imp_df.set_index("feature"))

with st.expander("Model evaluation details"):
    st.json(METRICS)
