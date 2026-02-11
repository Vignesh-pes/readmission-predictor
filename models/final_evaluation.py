import os

import joblib
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def main():

    print("\n===== FINAL MODEL EVALUATION =====\n")

    # -----------------------------
    # Resolve base directory
    # -----------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_path = os.path.join(BASE_DIR, "data", "raw", "diabetic_data.csv")
    artifact_path = os.path.join(BASE_DIR, "artifacts", "final_model.joblib")

    print("Loading raw dataset from:", data_path)
    print("Loading model from:", artifact_path)

    # -----------------------------
    # Load raw dataset
    # -----------------------------
    df = pd.read_csv(data_path)

    # -----------------------------
    # Create binary target
    # -----------------------------
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

    # -----------------------------
    # Drop leakage / unused columns
    # -----------------------------
    df = df.drop(columns=["encounter_id"])

    # -----------------------------
    # Recreate same split as training
    # -----------------------------
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["readmitted_binary"]
    )

    X_test = test_df.drop(columns=["readmitted", "readmitted_binary"])
    y_test = test_df["readmitted_binary"]

    # -----------------------------
    # Load trained pipeline
    # -----------------------------
    artifact = joblib.load(artifact_path)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    # -----------------------------
    # Predict
    # -----------------------------
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)

    # -----------------------------
    # Metrics
    # -----------------------------
    roc_auc = roc_auc_score(y_test, y_probs)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall_curve, precision_curve)

    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)

    cm = confusion_matrix(y_test, y_preds)

    # -----------------------------
    # Print results
    # -----------------------------
    print(f"Threshold used: {threshold}\n")

    print("--- Threshold-independent ---")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}\n")

    print("--- Threshold-dependent ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print("--- Confusion Matrix ---")
    print(cm)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_preds))


if __name__ == "__main__":
    main()
