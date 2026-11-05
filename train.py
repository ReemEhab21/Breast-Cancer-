"""
Breast Cancer Diagnostic Classifier — training pipeline.
Dataset: Breast Cancer Wisconsin (Diagnostic) — UCI / sklearn built-in.
569 real patient samples, 30 numeric cell-nucleus features.
Target: malignant (1) vs benign (0).

NOTE: this is a proposed own-project per the course rules —
needs instructor approval (one-paragraph proposal).

Usage:
    python src/train.py --out model.joblib
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

TARGET_NAME = "malignant"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="model.joblib")
    parser.add_argument("--metrics-out", default="metrics.json")
    args = parser.parse_args()

    data = load_breast_cancer()
    feature_names = list(data.feature_names)
    # sklearn target: 0 = malignant, 1 = benign -> relabel so 1 = malignant
    X = data.data
    y = 1 - data.target

    print(f"Loaded {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Malignant rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nAccuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=pred):")
    print(np.array(cm))

    perm = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=42,
        n_jobs=-1, scoring="roc_auc",
    )
    order = np.argsort(perm.importances_mean)[::-1][:15]
    importances = {
        feature_names[i]: round(float(perm.importances_mean[i]), 4)
        for i in order
    }
    print("\nTop-15 features:")
    for k, v in importances.items():
        print(f"  {k}: {v}")

    bundle = {
        "model": clf,
        "feature_names": feature_names,
        "feature_ranges": {
            name: [float(X[:, i].min()), float(X[:, i].max())]
            for i, name in enumerate(feature_names)
        },
        "feature_importances": importances,
        "metrics": {
            "accuracy": round(float(acc), 4),
            "roc_auc": round(float(auc), 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
    }
    joblib.dump(bundle, args.out, compress=3)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, indent=2)

    print(f"\nSaved bundle -> {args.out}")
    print(f"Metrics      -> {args.metrics_out}")


if __name__ == "__main__":
    main()
