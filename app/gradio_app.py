# src/app/gradio_app.py

import os

import gradio as gr
import joblib
import pandas as pd

# ----------------------------
# Load model artifact safely
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(BASE_DIR, "..", "..", "artifacts", "final_model.joblib")

artifact = joblib.load(ARTIFACT_PATH)
pipeline = artifact["pipeline"]
THRESHOLD = artifact["threshold"]


# ----------------------------
# Prediction function
# ----------------------------
def predict_readmission(
    age,
    gender,
    race,
    admission_type_id,
    discharge_disposition_id,
    admission_source_id,
    time_in_hospital,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    number_diagnoses,
    insulin,
    diabetesMed,
    change,
    diag_1,
    diag_2,
    diag_3,
):
    record = {
        "age": age,
        "gender": gender,
        "race": race,
        "admission_type_id": int(admission_type_id),
        "discharge_disposition_id": int(discharge_disposition_id),
        "admission_source_id": int(admission_source_id),
        "time_in_hospital": int(time_in_hospital),
        "num_lab_procedures": int(num_lab_procedures),
        "num_procedures": int(num_procedures),
        "num_medications": int(num_medications),
        "number_outpatient": int(number_outpatient),
        "number_emergency": int(number_emergency),
        "number_inpatient": int(number_inpatient),
        "number_diagnoses": int(number_diagnoses),
        "insulin": insulin,
        "diabetesMed": diabetesMed,
        "change": change,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
    }

    df = pd.DataFrame([record])

    prob = pipeline.predict_proba(df)[0, 1]
    if prob >= THRESHOLD:
        label = "⚠️ High Readmission Risk"
    else:
        label = "✅ Low Readmission Risk"

    return label, round(float(prob), 3), THRESHOLD


# ----------------------------
# Gradio UI
# ----------------------------
age_bins = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]

diag_examples = [
    "250.83",
    "428",
    "410",
    "401",
    "414",
    "486",
    "786",
    "403",
    "V45",
    "250.7",
]

with gr.Blocks(title="30-Day Hospital Readmission Risk Predictor") as demo:
    gr.Markdown(
        """
        # 🏥 30-Day Hospital Readmission Risk Predictor
        Predict whether a patient is likely to be **readmitted within 30 days**
        so hospitals can intervene early.
        """
    )

    with gr.Row():
        with gr.Column():
            age = gr.Dropdown(age_bins, label="Age")
            gender = gr.Dropdown(["Male", "Female", "Unknown/Invalid"], label="Gender")
            race = gr.Dropdown(
                ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "?"],
                label="Race",
            )

            admission_type_id = gr.Dropdown(
                [1, 2, 3, 4, 5, 6, 7, 8], value=1, label="Admission Type ID"
            )

            discharge_disposition_id = gr.Dropdown(
                [1, 2, 3, 4, 5, 6, 7, 11, 18, 19, 20, 25],
                value=1,
                label="Discharge Disposition ID",
            )

            admission_source_id = gr.Dropdown(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22],
                value=1,
                label="Admission Source ID",
            )

            time_in_hospital = gr.Number(value=3, label="Time in Hospital (days)")
            num_medications = gr.Number(value=10, label="Number of Medications")
            num_lab_procedures = gr.Number(value=40, label="Lab Procedures")
            num_procedures = gr.Number(value=0, label="Procedures")

        with gr.Column():
            number_outpatient = gr.Number(value=0, label="Outpatient Visits")
            number_emergency = gr.Number(value=0, label="Emergency Visits")
            number_inpatient = gr.Number(value=0, label="Inpatient Visits")
            number_diagnoses = gr.Number(value=2, label="Number of Diagnoses")

            insulin = gr.Dropdown(["No", "Up", "Down", "Steady"], label="Insulin")
            diabetesMed = gr.Dropdown(["Yes", "No"], label="Diabetes Medication")
            change = gr.Dropdown(["No", "Ch"], label="Medication Change")

            diag_1 = gr.Dropdown(diag_examples, label="Primary Diagnosis (diag_1)")
            diag_2 = gr.Dropdown(diag_examples, label="Secondary Diagnosis (diag_2)")
            diag_3 = gr.Dropdown(diag_examples, label="Tertiary Diagnosis (diag_3)")

    with gr.Row():
        predict_btn = gr.Button("Predict Readmission Risk")
        risk_label = gr.Textbox(label="Risk Assessment")
        prob_out = gr.Textbox(label="Predicted Probability")
        thresh_out = gr.Textbox(label="Decision Threshold")

    predict_btn.click(
        predict_readmission,
        inputs=[
            age,
            gender,
            race,
            admission_type_id,
            discharge_disposition_id,
            admission_source_id,
            time_in_hospital,
            num_lab_procedures,
            num_procedures,
            num_medications,
            number_outpatient,
            number_emergency,
            number_inpatient,
            number_diagnoses,
            insulin,
            diabetesMed,
            change,
            diag_1,
            diag_2,
            diag_3,
        ],
        outputs=[risk_label, prob_out, thresh_out],
    )


if __name__ == "__main__":
    server_name = os.getenv("SERVER_NAME", "127.0.0.1")
    demo.launch(server_name=server_name, server_port=7860, share=False)
