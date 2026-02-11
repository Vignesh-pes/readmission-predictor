# src/models/threshold_tuning.py

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocessing import build_preprocessing_pipeline


def main():
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train = train_df.drop(columns=["readmitted", "readmitted_binary", "patient_nbr"])
    y_train = train_df["readmitted_binary"]

    X_test = test_df.drop(columns=["readmitted", "readmitted_binary", "patient_nbr"])
    y_test = test_df["readmitted_binary"]

    # 🔐 Best params from Optuna
    best_params = {
        "n_estimators": 571,
        "max_depth": 6,
        "learning_rate": 0.017747030784529255,
        "subsample": 0.6779109723647712,
        "colsample_bytree": 0.8392168209257673,
        "min_child_weight": 3,
        "reg_alpha": 3.1353690038478406,
        "reg_lambda": 4.981120484534052,
        "scale_pos_weight": 7.87,
        "eval_metric": "auc",
        "tree_method": "hist",
    }

    model = XGBClassifier(**best_params)

    pipeline = Pipeline(
        [("preprocessing", build_preprocessing_pipeline()), ("model", model)]
    )

    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:, 1]

    print("\nThreshold tuning results:\n")
    print("Threshold | Precision | Recall | F1")
    print("----------------------------------")

    for t in np.arange(0.10, 0.61, 0.05):
        preds = (probs >= t).astype(int)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"{t:0.2f}      | {precision:0.3f}     | {recall:0.3f} | {f1:0.3f}")


if __name__ == "__main__":
    main()
