from pathlib import Path

import joblib

MODEL_PATH = Path("artifacts/final_model.joblib")


def load_model():
    return joblib.load(MODEL_PATH)


def predict(model_dict, df):
    pipeline = model_dict["pipeline"]
    prob = pipeline.predict_proba(df)[0, 1]
    return prob
