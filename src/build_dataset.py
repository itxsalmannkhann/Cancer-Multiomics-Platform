"""
Runs the full preprocessing + feature engineering pipeline and writes
processed, model-ready CSVs to data/processed/.

Outputs:
  data/processed/clinical_master.csv          patient-level clinical/exposure/followup/pathology
  data/processed/genome_features.csv          patient-level mutation features
  data/processed/transcriptome_matrix.csv     sample-level top-variable-gene expression (tumor+normal)
  data/processed/methylation_matrix.csv       sample-level top-variable-probe methylation (tumor+normal)
  data/processed/task1_tumor_vs_normal.csv    modeling table for Task 1
  data/processed/task2_stage.csv              modeling table for Task 2
  data/processed/task3_risk_factors.csv       modeling table for Task 3
  data/processed/task4_survival.csv           modeling table for Task 4
"""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.clinical_loader import build_clinical_master
from preprocessing.genome_features import build_genome_features
from preprocessing.omics_loader import load_omics_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
TOP_N_EXPR = 2000
TOP_N_METH = 2000


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("=== Step 1/6: clinical/exposure/follow-up/pathology ===")
    clinical = build_clinical_master(RAW_DIR)
    clinical.to_csv(f"{OUT_DIR}/clinical_master.csv", index=False)
    logger.info("clinical_master: %s", clinical.shape)

    logger.info("=== Step 2/6: genome (mutations) ===")
    genome = build_genome_features(f"{RAW_DIR}/mutations.txt")
    genome.to_csv(f"{OUT_DIR}/genome_features.csv", index=False)
    logger.info("genome_features: %s", genome.shape)

    logger.info("=== Step 3/6: transcriptome ===")
    expr = load_omics_matrix(f"{RAW_DIR}/transcriptomics.txt", top_n_features=TOP_N_EXPR,
                              log_transform=True, prefix="expr")
    expr.to_csv(f"{OUT_DIR}/transcriptome_matrix.csv", index=False)
    logger.info("transcriptome_matrix: %s", expr.shape)

    logger.info("=== Step 4/6: methylation ===")
    meth = load_omics_matrix(f"{RAW_DIR}/methylation.txt", top_n_features=TOP_N_METH,
                              log_transform=False, prefix="meth")
    meth.to_csv(f"{OUT_DIR}/methylation_matrix.csv", index=False)
    logger.info("methylation_matrix: %s", meth.shape)

    logger.info("=== Step 5/6: building task tables ===")

    # ---- Task 1: Tumor vs Normal --------------------------------------
    # One row per sample (not per patient), using expression + methylation
    # features for samples present in BOTH matrices (inner join on barcode
    # would be too strict across platforms, so we join on patient_id +
    # sample_type instead, since a patient can have one tumor + one normal
    # sample sequenced on both platforms).
    expr_t1 = expr.drop(columns=["sample_barcode_dup"], errors="ignore").copy()
    meth_t1 = meth.drop(columns=["sample_barcode"], errors="ignore").copy()
    task1 = expr_t1.merge(
        meth_t1, on=["patient_id", "sample_type"], how="inner", suffixes=("", "_dupmeth")
    )
    task1 = task1[task1["sample_type"].isin(["Tumor", "Normal"])].reset_index(drop=True)
    task1.to_csv(f"{OUT_DIR}/task1_tumor_vs_normal.csv", index=False)
    logger.info("task1_tumor_vs_normal: %s  label counts:\n%s",
                task1.shape, task1["sample_type"].value_counts().to_string())

    # ---- Shared tumor-only multi-omics table for tasks 2/3/4 ----------
    expr_tumor = expr[expr["sample_type"] == "Tumor"].drop(columns=["sample_barcode", "sample_type"])
    meth_tumor = meth[meth["sample_type"] == "Tumor"].drop(columns=["sample_barcode", "sample_type"])
    expr_tumor = expr_tumor.groupby("patient_id").mean(numeric_only=True).reset_index()
    meth_tumor = meth_tumor.groupby("patient_id").mean(numeric_only=True).reset_index()

    multiomics_tumor = expr_tumor.merge(meth_tumor, on="patient_id", how="outer")
    multiomics_tumor = multiomics_tumor.merge(genome, on="patient_id", how="left")
    multiomics_tumor = multiomics_tumor.merge(clinical, on="patient_id", how="left")

    # ---- Task 2: Stage -------------------------------------------------
    task2 = multiomics_tumor[multiomics_tumor["stage_group"].notna()].reset_index(drop=True)
    task2.to_csv(f"{OUT_DIR}/task2_stage.csv", index=False)
    logger.info("task2_stage: %s  label counts:\n%s",
                task2.shape, task2["stage_group"].value_counts().to_string())

    # ---- Task 3: Risk factors (smoking status) -------------------------
    task3 = multiomics_tumor[multiomics_tumor["smoking_status"].notna()].reset_index(drop=True)
    task3.to_csv(f"{OUT_DIR}/task3_risk_factors.csv", index=False)
    logger.info("task3_risk_factors: %s  label counts:\n%s",
                task3.shape, task3["smoking_status"].value_counts().to_string())

    # ---- Task 4: Survival -----------------------------------------------
    task4 = multiomics_tumor[
        multiomics_tumor["survival_days"].notna() & (multiomics_tumor["survival_days"] > 0)
    ].reset_index(drop=True)
    task4.to_csv(f"{OUT_DIR}/task4_survival.csv", index=False)
    logger.info("task4_survival: %s  event rate: %.2f",
                task4.shape, task4["event_death"].mean())

    logger.info("=== Step 6/6: done. Processed files written to %s ===", OUT_DIR)


if __name__ == "__main__":
    main()
