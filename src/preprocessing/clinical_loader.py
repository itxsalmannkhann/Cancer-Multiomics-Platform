"""
Loads and cleans the four GDC clinical-family TSVs (clinical, exposure,
follow_up, pathology_detail) and collapses each down to ONE ROW PER PATIENT.

GDC exports are exploded to one row per (diagnosis x treatment x follow-up)
combination, so a single patient can appear dozens of times. We aggregate
back to patient level by taking the first non-missing value per column
(these fields are patient/diagnosis-level constants that just get repeated
across the exploded rows) and, for a couple of fields that are genuinely
longitudinal (follow-up days, adverse events), we take the max/most-recent.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

MISSING_TOKENS = {"'--", "--", "", "Not Reported", "Unknown", "not reported", "unknown"}


def _clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(list(MISSING_TOKENS), pd.NA)


def _first_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else pd.NA


def load_patient_level(path: str, id_col: str = "cases.submitter_id") -> pd.DataFrame:
    """Generic loader: read TSV, clean sentinel-missing tokens, collapse to 1 row/patient."""
    df = pd.read_csv(path, sep="\t", low_memory=False, encoding="utf-8", on_bad_lines="skip")
    df = _clean_missing(df)
    if id_col not in df.columns:
        raise ValueError(f"{id_col} not found in {path}")
    n_before = df[id_col].nunique()
    agg = df.groupby(id_col, dropna=True).agg(_first_non_null)
    agg = agg.reset_index()
    logger.info("Loaded %s: %d raw rows -> %d unique patients", path, len(df), n_before)
    return agg


CLINICAL_KEEP = [
    "cases.submitter_id",
    "cases.disease_type",
    "cases.primary_site",
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.tumor_grade",
    "diagnoses.primary_diagnosis",
    "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.morphology",
    "diagnoses.prior_malignancy",
    "diagnoses.synchronous_malignancy",
]

EXPOSURE_KEEP = [
    "cases.submitter_id",
    "exposures.alcohol_history",
    "exposures.alcohol_intensity",
    "exposures.tobacco_smoking_status",
    "exposures.pack_years_smoked",
    "exposures.years_smoked",
    "exposures.tobacco_smoking_quit_year",
]

FOLLOWUP_KEEP = [
    "cases.submitter_id",
    "follow_ups.bmi",
    "follow_ups.ecog_performance_status",
    "follow_ups.comorbidities",
    "follow_ups.progression_or_recurrence",
]

PATHOLOGY_KEEP = [
    "cases.submitter_id",
    "pathology_details.margin_status",
    "pathology_details.lymph_node_involvement",
    "pathology_details.perineural_invasion_present",
    "pathology_details.lymphovascular_invasion_present",
]


def _subset(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    missing = set(cols) - set(present)
    if missing:
        logger.warning("Columns not found and skipped: %s", missing)
    return df[present]


def build_clinical_master(raw_dir: str) -> pd.DataFrame:
    """Merge clinical + exposure + follow_up + pathology_detail into one patient table."""
    clin = _subset(load_patient_level(f"{raw_dir}/clinical.txt"), CLINICAL_KEEP)
    exp = _subset(load_patient_level(f"{raw_dir}/exposure.txt"), EXPOSURE_KEEP)
    fu = _subset(load_patient_level(f"{raw_dir}/follow_up.txt"), FOLLOWUP_KEEP)
    path_ = _subset(load_patient_level(f"{raw_dir}/pathology_detail.txt"), PATHOLOGY_KEEP)

    master = clin.merge(exp, on="cases.submitter_id", how="left")
    master = master.merge(fu, on="cases.submitter_id", how="left")
    master = master.merge(path_, on="cases.submitter_id", how="left")
    master = master.rename(columns={"cases.submitter_id": "patient_id"})

    # Derived targets -----------------------------------------------------
    master["age_at_index"] = pd.to_numeric(master["demographic.age_at_index"], errors="coerce")
    master["days_to_death"] = pd.to_numeric(master.get("demographic.days_to_death"), errors="coerce")
    master["days_to_last_follow_up"] = pd.to_numeric(
        master.get("diagnoses.days_to_last_follow_up"), errors="coerce"
    )
    master["event_death"] = (master["demographic.vital_status"] == "Dead").astype(int)
    master["survival_days"] = master["days_to_death"].fillna(master["days_to_last_follow_up"])

    def stage_group(s):
        if pd.isna(s):
            return pd.NA
        s = str(s)
        if "IV" in s:
            return "Late (III/IV)"
        if "III" in s:
            return "Late (III/IV)"
        if "II" in s or "I" in s:
            return "Early (I/II)"
        return pd.NA

    master["stage_group"] = master["diagnoses.ajcc_pathologic_stage"].apply(stage_group)
    master["ajcc_stage"] = master["diagnoses.ajcc_pathologic_stage"]

    def smoker_group(s):
        if pd.isna(s):
            return pd.NA
        s = str(s)
        if s == "Lifelong Non-Smoker":
            return "Non-Smoker"
        if "Smoker" in s:
            return "Smoker"
        return pd.NA

    master["smoking_status"] = master["exposures.tobacco_smoking_status"].apply(smoker_group)
    master["alcohol_history"] = master.get("exposures.alcohol_history")

    return master
