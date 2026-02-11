import joblib
from pathlib import Path

MODEL_PATH = Path("artifacts/final_model.joblib")


def load_model():
    return joblib.load(MODEL_PATH)


def predict(model_dict, df):
    pipeline = model_dict["pipeline"]
    prob = pipeline.predict_proba(df)[0, 1]
    return prob
