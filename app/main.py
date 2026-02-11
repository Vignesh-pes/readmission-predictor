# app/main.py
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Readmission Prediction API")


# -----------------------------
# Request Schema
# -----------------------------
class PatientData(BaseModel):
    age: str
    gender: str
    race: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    insulin: str
    diabetesMed: str
    change: str
    diag_1: str
    diag_2: str
    diag_3: str


# -----------------------------
# Startup: Load Model
# -----------------------------
@app.on_event("startup")
def load_artifact():
    model_path = Path("artifacts/final_model.joblib")

    if not model_path.exists():
        print("⚠ Model file not found:", model_path)
        app.state.pipeline = None
        app.state.threshold = 0.5
        return

    try:
        artifact = joblib.load(model_path)
        app.state.pipeline = artifact["pipeline"]
        app.state.threshold = artifact.get("threshold", 0.5)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print("❌ Model loading failed:", str(e))
        app.state.pipeline = None
        app.state.threshold = 0.5


# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_readmission(data: PatientData):

    record = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    df = pd.DataFrame([record])

    pipeline = getattr(app.state, "pipeline", None)

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prob = pipeline.predict_proba(df)[0, 1]
    prediction = int(prob >= app.state.threshold)

    return {
        "readmission_probability": round(float(prob), 4),
        "prediction": prediction,
    }
