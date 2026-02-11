# 🏥 Hospital Readmission Predictor

A production-grade Machine Learning system that predicts hospital
readmission risk using an XGBoost model, deployed via FastAPI,
containerized with Docker, and automated using GitHub Actions CI/CD.

------------------------------------------------------------------------

## 🚀 Project Overview

This project predicts the probability of hospital readmission based on
patient data such as:

-   Demographics
-   Admission details
-   Diagnosis codes
-   Medication information

The model is trained using XGBoost and deployed as a REST API.

------------------------------------------------------------------------

## 🏗️ Tech Stack

-   Python 3.11
-   FastAPI
-   XGBoost
-   Scikit-learn
-   Pandas
-   Docker
-   GitHub Actions (CI/CD)
-   Ruff (Linting)
-   Bandit (Security Scanning)
-   Pytest (Testing + Coverage)

------------------------------------------------------------------------

## 📂 Project Structure

    readmission-predictor/
    │
    ├── app/                    # FastAPI application
    │   └── main.py
    │
    ├── src/                    # Training & preprocessing logic
    │   ├── features/
    │   ├── inference/
    │   └── data/
    │
    ├── artifacts/              # Trained model artifact
    │   └── final_model.joblib
    │
    ├── tests/                  # Unit & API tests
    │
    ├── Dockerfile
    ├── requirements.txt
    ├── .github/workflows/      # CI/CD pipelines
    │   ├── ci.yml
    │   └── docker.yml
    │
    └── README.md

------------------------------------------------------------------------

## ⚙️ Running Locally

### 1️⃣ Create Virtual Environment

``` bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 2️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### 3️⃣ Start API

``` bash
uvicorn app.main:app --reload
```

Open:

    http://localhost:8000/docs

------------------------------------------------------------------------

## 🐳 Run with Docker

### Pull Image

``` bash
docker pull <your-dockerhub-username>/readmission-predictor:latest
```

### Run Container

``` bash
docker run -p 8000:8000 <your-dockerhub-username>/readmission-predictor:latest
```

Open:

    http://localhost:8000/docs

------------------------------------------------------------------------

## 🔄 CI/CD Pipeline

On every push to `main`:

1.  Ruff lint check\
2.  Bandit security scan\
3.  Pytest with coverage enforcement\
4.  Docker image build\
5.  Automatic push to DockerHub

Manual Docker builds are not required.

------------------------------------------------------------------------

## 📊 API Endpoint

### POST `/predict`

Request Body:

``` json
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
  "diag_3": "276"
}
```

Response:

``` json
{
  "readmission_probability": 0.7321,
  "prediction": 1
}
```

------------------------------------------------------------------------

## 🛡 Security & Code Quality

-   Static analysis via Ruff\
-   Security scanning via Bandit\
-   80% minimum test coverage enforced\
-   Deterministic dependency versions

------------------------------------------------------------------------

## 📦 Model Details

-   Algorithm: XGBoost Classifier\
-   Threshold tuning applied\
-   Saved using Joblib\
-   Loaded at application startup

------------------------------------------------------------------------

## 👨‍💻 Author

Vignesh , Likitha , Aaryan , Harshitha 

------------------------------------------------------------------------

## 📌 Future Improvements

-   Model registry integration\
-   Cloud deployment (AWS / Azure)\
-   Monitoring & logging\
-   Automated retraining pipeline
