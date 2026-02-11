# src/data/load_and_split.py

import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw diabetic hospital data
    """
    df = pd.read_csv(path)
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target:
    1 -> readmitted within 30 days
    0 -> otherwise
    """
    df = df.copy()
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
    return df


def patient_level_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data by patient_nbr to avoid data leakage
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    groups = df["patient_nbr"]
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df


def save_processed_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    RAW_PATH = "data/raw/diabetic_data.csv"
    TRAIN_PATH = "data/processed/train.parquet"
    TEST_PATH = "data/processed/test.parquet"

    df = load_raw_data(RAW_PATH)
    print("Raw shape:", df.shape)

    df = create_target(df)
    print("Target distribution:")
    print(df["readmitted_binary"].value_counts(normalize=True))

    train_df, test_df = patient_level_split(df)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # safety check
    overlap = set(train_df["patient_nbr"]).intersection(
        set(test_df["patient_nbr"])
    )
    print("Patient overlap:", len(overlap))

    save_processed_data(train_df, TRAIN_PATH)
    save_processed_data(test_df, TEST_PATH)

    print("Processed data saved.")
