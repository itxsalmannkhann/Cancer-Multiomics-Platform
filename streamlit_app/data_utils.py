"""Shared data/model loading helpers for the Streamlit app. Cached so the
~25MB of processed CSVs and joblib models are only read once per session."""
from __future__ import annotations

import json
import os

import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = f"{ROOT}/data/processed"
MODELS = f"{ROOT}/models"
REPORTS = f"{ROOT}/reports"


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(f"{PROCESSED}/{name}")


@st.cache_data
def load_report(name: str) -> dict:
    path = f"{REPORTS}/{name}"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _iter_estimators(obj):
    """Yield the object and any nested sub-estimators (pipeline steps,
    calibrated classifiers, meta-estimators) so we can patch each one."""
    if obj is None:
        return
    yield obj
    # sklearn Pipeline
    for attr in ("steps",):
        steps = getattr(obj, attr, None)
        if steps:
            for _, est in steps:
                yield from _iter_estimators(est)
    # common wrapper attributes
    for attr in ("estimator", "base_estimator", "final_estimator", "best_estimator_"):
        sub = getattr(obj, attr, None)
        if sub is not None and sub is not obj:
            yield from _iter_estimators(sub)
    # calibrated classifiers hold a list of fitted sub-classifiers
    for cc in getattr(obj, "calibrated_classifiers_", []) or []:
        inner = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if inner is not None:
            yield from _iter_estimators(inner)


def _patch_sklearn_compat(obj):
    """Backfill attributes removed/renamed across scikit-learn versions so
    models pickled with an older version still run under the installed one.
    Currently restores ``multi_class`` on LogisticRegression (removed in 1.7)."""
    for est in _iter_estimators(obj):
        cls_name = type(est).__name__
        if cls_name in ("LogisticRegression", "LogisticRegressionCV"):
            if not hasattr(est, "multi_class"):
                # "auto" replicates the historical default: multinomial unless
                # the solver is liblinear, in which case one-vs-rest.
                est.multi_class = "auto"
    return obj


@st.cache_resource
def load_model_bundle(task_name: str):
    path = f"{MODELS}/{task_name}_best_model.joblib"
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    _patch_sklearn_compat(bundle.get("pipeline") if isinstance(bundle, dict) else bundle)
    return bundle


@st.cache_resource
def load_survival_bundle():
    path = f"{MODELS}/task4_survival_pipeline.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


TASK_META = {
    "task1_tumor_vs_normal": {
        "title": "Tumor vs Normal Tissue",
        "data_file": "task1_tumor_vs_normal.csv",
        "label_col": "sample_type",
    },
    "task2_stage": {
        "title": "Cancer Stage (Early I/II vs Late III/IV)",
        "data_file": "task2_stage.csv",
        "label_col": "stage_group",
    },
    "task3_risk_factors": {
        "title": "Smoking Status (from molecular signature)",
        "data_file": "task3_risk_factors.csv",
        "label_col": "smoking_status",
    },
}
