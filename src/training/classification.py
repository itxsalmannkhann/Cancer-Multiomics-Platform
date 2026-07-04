"""
Generic small-sample classification trainer used for Task 1 (Tumor vs
Normal), Task 2 (Stage), and Task 3 (Risk factors).

Given the sample sizes here (70-88 rows) and thousands of omics features,
every pipeline does: StandardScaler -> SelectKBest(f_classif) -> model,
evaluated with Stratified K-Fold cross-validation (not a single train/test
split, which would be unstable at this n). The best model (by mean CV
macro-F1) is refit on all data and persisted.
"""
from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb

logger = logging.getLogger(__name__)

MODEL_ZOO = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "lightgbm": lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    class_weight="balanced", verbose=-1, random_state=42),
}


def _id_and_meta_cols(df: pd.DataFrame, label_col: str, leakage_prefixes: tuple[str, ...] = ()) -> list[str]:
    """Non-feature columns to drop before modeling (ids, raw clinical text, other targets,
    and any columns that would leak the label -- e.g. don't let the model 'predict' smoking
    status using other smoking-exposure fields, or predict stage using the raw AJCC T/N/M
    fields that stage is literally derived from)."""
    known_meta = {
        "patient_id", "sample_barcode", "sample_type",
        "stage_group", "ajcc_stage", "smoking_status", "alcohol_history",
        "event_death", "survival_days", "days_to_death", "days_to_last_follow_up",
        label_col,
    }
    obj_cols = [c for c in df.columns if df[c].dtype == object and c not in known_meta]
    leaky_cols = [c for c in df.columns if any(c.startswith(p) for p in leakage_prefixes)]
    return list(known_meta.union(obj_cols).union(leaky_cols))


def prepare_xy(df: pd.DataFrame, label_col: str, leakage_prefixes: tuple[str, ...] = ()):
    meta_cols = _id_and_meta_cols(df, label_col, leakage_prefixes)
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_raw = df[label_col].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le, feature_cols


def train_and_evaluate(
    df: pd.DataFrame,
    label_col: str,
    task_name: str,
    models_dir: str,
    reports_dir: str,
    k_best: int = 50,
    n_splits: int = 5,
    leakage_prefixes: tuple[str, ...] = (),
) -> dict:
    X, y, le, feature_cols = prepare_xy(df, label_col, leakage_prefixes)
    n_splits = min(n_splits, pd.Series(y).value_counts().min())
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    k_best = min(k_best, X.shape[1])

    results = {}
    fitted_pipelines = {}
    for name, model in MODEL_ZOO.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k_best)),
            ("clf", model),
        ])
        try:
            y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "macro_f1": f1_score(y, y_pred, average="macro"),
                "weighted_f1": f1_score(y, y_pred, average="weighted"),
                "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
                "cohen_kappa": cohen_kappa_score(y, y_pred),
            }
            if len(np.unique(y)) == 2:
                try:
                    y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y, y_proba)
                except Exception:
                    pass
            metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()
            results[name] = metrics
            pipe.fit(X, y)
            fitted_pipelines[name] = pipe
            logger.info("[%s] %s -> macro_f1=%.3f acc=%.3f", task_name, name, metrics["macro_f1"], metrics["accuracy"])
        except Exception as e:
            logger.warning("[%s] %s failed: %s", task_name, name, e)

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_pipe = fitted_pipelines[best_name]

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "label_encoder": le, "feature_cols": feature_cols},
                f"{models_dir}/{task_name}_best_model.joblib")

    report = {
        "task": task_name,
        "label_col": label_col,
        "classes": le.classes_.tolist(),
        "n_samples": int(len(y)),
        "n_features_total": int(X.shape[1]),
        "k_best_selected": int(k_best),
        "cv_folds": int(n_splits),
        "best_model": best_name,
        "all_model_results": results,
    }
    with open(f"{reports_dir}/{task_name}_evaluation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("[%s] BEST MODEL: %s (macro_f1=%.3f)", task_name, best_name, results[best_name]["macro_f1"])
    return report
