"""
Standalone CLI inference: score a patient/sample already present in one of
the processed task tables, without needing Streamlit.

Usage:
    python src/inference/predict.py --task task2_stage --id TCGA-CR-7392
    python src/inference/predict.py --task task1_tumor_vs_normal --id TCGA-CR-7392-01A-11R-1873-07
"""
from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASK_FILES = {
    "task1_tumor_vs_normal": ("task1_tumor_vs_normal.csv", "sample_barcode", "sample_type"),
    "task2_stage": ("task2_stage.csv", "patient_id", "stage_group"),
    "task3_risk_factors": ("task3_risk_factors.csv", "patient_id", "smoking_status"),
}


def main():
    parser = argparse.ArgumentParser(description="Score a patient/sample with a trained model.")
    parser.add_argument("--task", required=True, choices=list(TASK_FILES.keys()))
    parser.add_argument("--id", required=True, help="patient_id or sample_barcode, depending on task")
    args = parser.parse_args()

    data_file, id_col, label_col = TASK_FILES[args.task]
    df = pd.read_csv(f"{ROOT}/data/processed/{data_file}")
    bundle = joblib.load(f"{ROOT}/models/{args.task}_best_model.joblib")

    row = df[df[id_col] == args.id]
    if row.empty:
        raise SystemExit(f"No row found with {id_col} == {args.id!r} in {data_file}")

    pipe, le, feature_cols = bundle["pipeline"], bundle["label_encoder"], bundle["feature_cols"]
    X = row[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pred = le.inverse_transform(pipe.predict(X))[0]
    print(f"{id_col} = {args.id}")
    print(f"Predicted {label_col}: {pred}")
    print(f"Actual {label_col}:    {row[label_col].iloc[0]}")
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        for cls, p in zip(le.classes_, proba):
            print(f"  P({cls}) = {p:.3f}")


if __name__ == "__main__":
    main()
