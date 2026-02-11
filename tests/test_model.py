# tests/test_model.py
import joblib
import pandas as pd


def test_model_prediction():
    # joblib.load is mocked in conftest, so we get the fake artifact
    artifact = joblib.load("artifacts/final_model.joblib")
    assert isinstance(artifact, dict)
    assert "pipeline" in artifact and "threshold" in artifact

    pipeline = artifact["pipeline"]

    sample = pd.DataFrame(
        [
            {
                "age": "[60-70)",
                "gender": "Male",
                "race": "Caucasian",
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "time_in_hospital": 3,
                "num_lab_procedures": 45,
                "num_procedures": 1,
                "num_medications": 13,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 0,
                "number_diagnoses": 5,
                "insulin": "No",
                "diabetesMed": "Yes",
                "change": "No",
                "diag_1": "250.83",
                "diag_2": "401.9",
                "diag_3": "276",
            }
        ]
    )

    prob = pipeline.predict_proba(sample)[0, 1]
    assert 0 <= prob <= 1
