# src/models/train_xgb_optuna.py

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
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

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print("scale_pos_weight:", round(scale_pos_weight, 2))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "tree_method": "hist",
        }

        model = XGBClassifier(**params)

        pipeline = Pipeline(
            [("preprocessing", build_preprocessing_pipeline()), ("model", model)]
        )

        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    print("\nBest ROC-AUC:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
