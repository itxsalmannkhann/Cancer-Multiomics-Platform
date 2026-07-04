"""
SHAP explainability for the tree-based models saved by training/classification.py.
Produces a summary bar plot (mean |SHAP value| per feature) and a JSON of the
top-N most important features per task, saved to reports/figures and reports/.
"""
from __future__ import annotations

import json
import logging
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def explain_task(task_name: str, df: pd.DataFrame, label_col: str,
                  models_dir: str, reports_dir: str, top_n: int = 20) -> dict | None:
    bundle_path = f"{models_dir}/{task_name}_best_model.joblib"
    if not os.path.exists(bundle_path):
        logger.warning("No saved model for %s, skipping SHAP", task_name)
        return None

    bundle = joblib.load(bundle_path)
    pipe = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    clf = pipe.named_steps["clf"]

    X_scaled = pipe.named_steps["scaler"].transform(X)
    X_selected = pipe.named_steps["select"].transform(X_scaled)
    selected_mask = pipe.named_steps["select"].get_support()
    selected_features = [f for f, keep in zip(feature_cols, selected_mask) if keep]

    clf_name = clf.__class__.__name__
    try:
        if clf_name in ("RandomForestClassifier", "LGBMClassifier"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_selected)
        elif clf_name == "LogisticRegression":
            explainer = shap.LinearExplainer(clf, X_selected)
            shap_values = explainer.shap_values(X_selected)
        else:
            logger.info("[%s] unsupported model type %s for SHAP, skipping", task_name, clf_name)
            return None
    except Exception as e:
        logger.warning("[%s] SHAP explainer failed: %s", task_name, e)
        return None

    if isinstance(shap_values, list):  # multi-class -> average abs across classes
        abs_vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        abs_vals = np.abs(shap_values)
        if abs_vals.ndim == 3:  # (n_samples, n_features, n_classes)
            abs_vals = abs_vals.mean(axis=2)

    mean_abs = abs_vals.mean(axis=0)
    importance = pd.Series(mean_abs, index=selected_features).sort_values(ascending=False)

    os.makedirs(f"{reports_dir}/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.head(top_n).iloc[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} features - {task_name}")
    fig.tight_layout()
    fig.savefig(f"{reports_dir}/figures/{task_name}_shap_importance.png", dpi=150)
    plt.close(fig)

    top_features = importance.head(top_n).to_dict()
    with open(f"{reports_dir}/{task_name}_shap_top_features.json", "w") as f:
        json.dump({k: float(v) for k, v in top_features.items()}, f, indent=2)

    logger.info("[%s] SHAP explainability written (top feature: %s)",
                task_name, importance.index[0])
    return top_features
