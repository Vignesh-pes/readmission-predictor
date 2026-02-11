# src/models/train_baseline.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessing_pipeline


def main():
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train = train_df.drop(columns=["readmitted", "readmitted_binary", "patient_nbr"])
    y_train = train_df["readmitted_binary"]

    X_test = test_df.drop(columns=["readmitted", "readmitted_binary", "patient_nbr"])
    y_test = test_df["readmitted_binary"]

    # Class-weighted Logistic Regression
    model = LogisticRegression(
        max_iter=2000,
        class_weight={0: 1, 1: 5},  # penalize missing readmissions
        n_jobs=-1,
    )

    pipeline = Pipeline(
        [("preprocessing", build_preprocessing_pipeline()), ("model", model)]
    )

    pipeline.fit(X_train, y_train)

    # Evaluation
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds))

    roc = roc_auc_score(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    main()
