"""
Task 4 - Advanced insights:
  - Survival risk score via Cox Proportional Hazards (lifelines), on a
    dimensionality-reduced feature set (PCA over omics + raw clinical/genome
    features) since n_patients (82) << n_raw_features (~4000).
  - Biomarker importance: |Cox coefficient| per covariate, plus which raw
    genes/probes load most heavily onto the top PCA components (proxy for
    "most important biomarkers" without needing n>>p).
  - Patient similarity: cosine-similarity nearest neighbours in the
    multi-omics PCA space.
  - Cancer subtype discovery: KMeans clustering over the same PCA space,
    model selection via silhouette score.
"""
from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _omics_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("expr_") or c.startswith("meth_")]


def _clinical_numeric_cols(df: pd.DataFrame) -> list[str]:
    candidates = ["age_at_index", "mutation_burden", "deleterious_mutation_burden"]
    return [c for c in candidates if c in df.columns]


def run_survival_and_insights(
    df: pd.DataFrame,
    models_dir: str,
    reports_dir: str,
    n_pca_components: int = 10,
    n_neighbors: int = 5,
) -> dict:
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    df = df.reset_index(drop=True).copy()
    omics_cols = _omics_feature_cols(df)
    clin_cols = _clinical_numeric_cols(df)

    X_omics = df[omics_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_clin = df[clin_cols].apply(pd.to_numeric, errors="coerce").fillna(df[clin_cols].median(numeric_only=True))

    scaler = StandardScaler()
    X_omics_scaled = scaler.fit_transform(X_omics)
    n_comp = min(n_pca_components, X_omics_scaled.shape[0] - 1, X_omics_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pcs = pca.fit_transform(X_omics_scaled)
    pc_cols = [f"PC{i+1}" for i in range(n_comp)]
    pc_df = pd.DataFrame(pcs, columns=pc_cols)

    # ---------------- Cox Proportional Hazards ----------------------------
    cox_df = pd.concat([pc_df, X_clin.reset_index(drop=True)], axis=1)
    cox_df["survival_days"] = df["survival_days"].values
    cox_df["event_death"] = df["event_death"].values
    cox_df = cox_df.dropna()

    cph = CoxPHFitter(penalizer=0.3)
    cph.fit(cox_df, duration_col="survival_days", event_col="event_death")
    c_index = cph.concordance_index_

    risk_scores = cph.predict_partial_hazard(cox_df)
    risk_out = df.loc[cox_df.index, ["patient_id"]].copy()
    risk_out["survival_risk_score"] = risk_scores.values
    risk_out["risk_group"] = pd.qcut(
        risk_out["survival_risk_score"], q=2, labels=["Low Risk", "High Risk"]
    )
    risk_out.to_csv(f"{reports_dir}/task4_survival_risk_scores.csv", index=False)

    hazard_ratios = cph.hazard_ratios_.sort_values(ascending=False)

    # Which raw genes/probes drive the top PCA components (proxy biomarker importance)
    loadings = pd.DataFrame(pca.components_, columns=omics_cols, index=pc_cols)
    top_biomarkers = {}
    for pc in pc_cols[:3]:
        top_genes = loadings.loc[pc].abs().sort_values(ascending=False).head(10)
        top_biomarkers[pc] = top_genes.to_dict()

    # ---------------- Patient similarity (cosine, PCA space) --------------
    sim_matrix = cosine_similarity(pcs)
    patient_ids = df["patient_id"].values
    similarity_records = []
    for i, pid in enumerate(patient_ids):
        order = np.argsort(-sim_matrix[i])
        order = [o for o in order if o != i][:n_neighbors]
        similarity_records.append({
            "patient_id": pid,
            "most_similar_patients": [patient_ids[o] for o in order],
            "similarity_scores": [round(float(sim_matrix[i, o]), 4) for o in order],
        })
    with open(f"{reports_dir}/task4_patient_similarity.json", "w") as f:
        json.dump(similarity_records, f, indent=2, default=str)

    # ---------------- Subtype discovery (KMeans + silhouette) -------------
    best_k, best_score, best_labels = None, -1, None
    for k in range(2, min(6, len(df) - 1)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(pcs)
        score = silhouette_score(pcs, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    subtype_out = df[["patient_id"]].copy()
    subtype_out["discovered_subtype"] = [f"Subtype {l+1}" for l in best_labels]
    subtype_out.to_csv(f"{reports_dir}/task4_subtype_discovery.csv", index=False)

    joblib.dump(
        {"scaler": scaler, "pca": pca, "cox_model": cph, "omics_cols": omics_cols, "clin_cols": clin_cols},
        f"{models_dir}/task4_survival_pipeline.joblib",
    )

    report = {
        "task": "task4_advanced_insights",
        "n_patients_used": int(len(cox_df)),
        "cox_concordance_index": float(c_index),
        "top_hazard_ratios": {k: float(v) for k, v in hazard_ratios.head(10).items()},
        "pca_components_used": n_comp,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "top_biomarkers_by_pc": top_biomarkers,
        "best_subtype_k": int(best_k),
        "best_subtype_silhouette": float(best_score),
        "subtype_counts": pd.Series(subtype_out["discovered_subtype"]).value_counts().to_dict(),
    }
    with open(f"{reports_dir}/task4_evaluation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("[task4] Cox c-index=%.3f | best subtype k=%d (silhouette=%.3f)",
                c_index, best_k, best_score)
    return report
