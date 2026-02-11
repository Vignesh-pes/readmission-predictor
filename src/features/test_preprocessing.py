import pandas as pd

from src.features.preprocessing import build_preprocessing_pipeline

df = pd.read_parquet("data/processed/train.parquet")

X = df.drop(columns=["readmitted", "readmitted_binary"])

pipeline = build_preprocessing_pipeline()
Xt = pipeline.fit_transform(X)

print("Transformed shape:", Xt.shape)
