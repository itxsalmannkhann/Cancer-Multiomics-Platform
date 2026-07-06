# Architecture

This document describes how data flows through the platform, from raw TCGA
files to the interactive dashboard.

## High-level pipeline

```
data/raw/*.txt
      │  src/build_dataset.py
      ▼
data/processed/*.csv        (patient/sample-level modeling tables)
      │  main.py
      ├── src/training/classification.py   → tasks 1-3
      ├── src/training/survival.py          → task 4 (Cox)
      └── src/explainability/shap_utils.py  → SHAP importances
      ▼
models/*.joblib  +  reports/*.json  +  reports/figures/*.png
      │  streamlit_app/app.py
      ▼
Multi-page Streamlit dashboard
```

## Layers

### 1. Preprocessing (`src/preprocessing/`)
Loaders for clinical, exposure, follow-up, pathology, genome (MAF-style), and
omics (transcriptome / methylation) inputs. TCGA barcodes are parsed in
`src/utils/barcode.py` to distinguish tumor (`-01`) from normal (`-11`) samples.

### 2. Feature assembly (`src/build_dataset.py`)
Joins the modalities on patient/sample identifiers and writes the per-task
modeling tables to `data/processed/`.

### 3. Modeling (`src/training/`)
- `classification.py` — Tasks 1-3, each using a
  `StandardScaler → SelectKBest → model` pipeline evaluated with stratified
  K-fold cross-validation.
- `survival.py` — Task 4 Cox proportional-hazards model on PCA-reduced omics
  plus clinical features.

### 4. Explainability (`src/explainability/`)
SHAP values for tree and linear models, plus Cox hazard ratios and PCA-loading
biomarker tracing.

### 5. Serving (`streamlit_app/`)
The dashboard reads pre-computed artifacts (`models/`, `reports/`) rather than
retraining on page load. Shared visual language lives in `streamlit_app/ui/`.

## Design principles

- **Reproducible:** re-running `build_dataset.py` + `main.py` retrains from
  scratch; nothing is hardcoded to the current cohort.
- **Honest evaluation:** out-of-fold cross-validation, not in-sample metrics.
- **Explicit leakage guards:** documented in `main.py`.
