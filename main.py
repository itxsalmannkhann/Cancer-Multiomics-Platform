"""
Orchestrates model training + evaluation + explainability for all four tasks,
after data/build_dataset.py has produced the processed CSVs.

Usage:
    python main.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from src.training.classification import train_and_evaluate
from src.training.survival import run_survival_and_insights
from src.explainability.shap_utils import explain_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = f"{BASE}/data/processed"
MODELS = f"{BASE}/models"
REPORTS = f"{BASE}/reports"


def main():
    all_reports = {}

    logger.info("############ TASK 1: Tumor vs Normal ############")
    task1 = pd.read_csv(f"{PROCESSED}/task1_tumor_vs_normal.csv")
    all_reports["task1"] = train_and_evaluate(
        task1, label_col="sample_type", task_name="task1_tumor_vs_normal",
        models_dir=MODELS, reports_dir=REPORTS, k_best=50,
    )
    explain_task("task1_tumor_vs_normal", task1, "sample_type", MODELS, REPORTS)

    logger.info("############ TASK 2: Cancer Stage ############")
    task2 = pd.read_csv(f"{PROCESSED}/task2_stage.csv")
    all_reports["task2"] = train_and_evaluate(
        task2, label_col="stage_group", task_name="task2_stage",
        models_dir=MODELS, reports_dir=REPORTS, k_best=50,
        leakage_prefixes=("diagnoses.ajcc", "diagnoses.morphology", "diagnoses.primary_diagnosis"),
    )
    explain_task("task2_stage", task2, "stage_group", MODELS, REPORTS)

    logger.info("############ TASK 3: Risk factors (Smoking status) ############")
    task3 = pd.read_csv(f"{PROCESSED}/task3_risk_factors.csv")
    all_reports["task3"] = train_and_evaluate(
        task3, label_col="smoking_status", task_name="task3_risk_factors",
        models_dir=MODELS, reports_dir=REPORTS, k_best=50,
        leakage_prefixes=("exposures.",),
    )
    explain_task("task3_risk_factors", task3, "smoking_status", MODELS, REPORTS)

    logger.info("############ TASK 4: Advanced insights (survival, biomarkers, similarity, subtypes) ############")
    task4 = pd.read_csv(f"{PROCESSED}/task4_survival.csv")
    all_reports["task4"] = run_survival_and_insights(task4, MODELS, REPORTS)

    with open(f"{REPORTS}/all_tasks_summary.json", "w") as f:
        json.dump(all_reports, f, indent=2, default=str)

    logger.info("############ ALL TASKS COMPLETE ############")
    for k, v in all_reports.items():
        if "best_model" in v:
            logger.info("%s -> best_model=%s macro_f1=%.3f",
                        k, v["best_model"], v["all_model_results"][v["best_model"]]["macro_f1"])
        elif "cox_concordance_index" in v:
            logger.info("%s -> cox c-index=%.3f, best subtype k=%d",
                        k, v["cox_concordance_index"], v["best_subtype_k"])


if __name__ == "__main__":
    main()
