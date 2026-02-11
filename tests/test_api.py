# tests/test_api.py
def test_predict_endpoint(client):
    payload = {
        "age": "[60-70)",  # use any string; mock pipeline accepts any df
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "readmission_probability" in body
    assert "prediction" in body
    assert 0 <= body["readmission_probability"] <= 1
    assert body["prediction"] in (0, 1)
