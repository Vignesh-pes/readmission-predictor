# src/models/train_final_model.py

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocessing import build_preprocessing_pipeline

FINAL_THRESHOLD = 0.45


def main():
    train_df = pd.read_parquet("data/processed/train.parquet")

    X_train = train_df.drop(columns=["readmitted", "readmitted_binary", "patient_nbr"])
    y_train = train_df["readmitted_binary"]

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

    joblib.dump(
        {"pipeline": pipeline, "threshold": FINAL_THRESHOLD},
        "artifacts/final_model.joblib",
    )

    print("Final model saved with threshold =", FINAL_THRESHOLD)


if __name__ == "__main__":
    main()
