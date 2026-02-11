# src/features/preprocessing.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


# -------------------------
# Diagnosis code grouping
# -------------------------
def map_diag(code):
    """
    Map ICD-9 codes into high-level clinical groups
    """
    try:
        code = str(code)
        if code.startswith("250"):
            return "diabetes"

        val = float(code.split(".")[0])

        if 390 <= val <= 459:
            return "circulatory"
        if 460 <= val <= 519:
            return "respiratory"
        if 520 <= val <= 579:
            return "digestive"
        if 800 <= val <= 999:
            return "injury"
        return "other"

    except Exception:
        return "other"


def add_diag_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_group"] = df[col].apply(map_diag)
    return df


# -------------------------
# Preprocessing pipeline
# -------------------------
def build_preprocessing_pipeline():

    numeric_features = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    ordinal_features = ["age"]

    categorical_features = [
        "race",
        "gender",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "insulin",
        "diabetesMed",
        "change",
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("ord", ordinal_pipeline, ordinal_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline(
        steps=[
            ("diag_mapper", FunctionTransformer(add_diag_groups)),
            ("preprocessor", preprocessor),
        ]
    )

    return full_pipeline
