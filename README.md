# Cancer Multi-Omics Intelligence Platform

Built for **NeoHacks GIKI Edition 2025**. An end-to-end, real (not templated)
multi-omics ML pipeline + Streamlit app trained on a **TCGA Head & Neck
Squamous Cell Carcinoma (HNSC) cohort**.

## ⚠️ Read this first: what's real vs. what's scoped

The brief describes 4 data modalities (images, genome, transcriptome,
epigenome). **Only 3 were actually provided** — no `.svs` histopathology
slides were in the uploaded dataset, only 7 tabular files:

| File | Modality | Rows |
|---|---|---|
| `clinical.txt` | Clinical | 309 rows → 82 patients |
| `exposure.txt` | Risk factors (smoking/alcohol) | 114 rows → 82 patients |
| `follow_up.txt` | Longitudinal follow-up | 279 rows → 82 patients |
| `pathology_detail.txt` | Pathology detail | 175 rows → 82 patients |
| `mutations.txt` | Genome (MAF-style) | 13,180 mutation calls, 82 tumor samples |
| `transcriptomics.txt` | Transcriptome (RNA-seq) | 20,502 genes x 92 samples |
| `methylation.txt` | Epigenome (methylation) | 20,114 probes x 96 samples |

This is also **one cancer type, not a cancer-type-agnostic dataset**
(`cases.disease_type` = "Squamous Cell Neoplasms", primary sites = tongue,
tonsil, floor of mouth, gum, palate, lip → classic HNSC). That reshapes the
4 tasks into their honest, data-supported form:

| Brief's task | What we actually built | Why |
|---|---|---|
| Cancer vs Non-Cancer | **Tumor vs Normal tissue** classification | Every sample is HNSC; but ~10-14 patients have a matched *normal* tissue sample (barcode suffix `-11` vs `-01`), so tumor-vs-normal is the real binary distinction in the data. |
| Stage I-IV multiclass | **Early (I/II) vs Late (III/IV)** binary | Only 15/27/25/108/2 patients per stage — 4-way classification on ~80 patients would be unstable/meaningless. Binarizing gives usable class sizes. |
| Risk factor prediction | **Smoking status from molecular signature** (omics only — other exposure fields are explicitly excluded as features, see leakage note below) | `exposure.txt` has real `tobacco_smoking_status` / `alcohol_history` fields; used as labels, not inputs. |
| Advanced insights | Cox survival risk score, SHAP-driven biomarkers, patient similarity, subtype discovery | All real, computed on the same cohort. |
| Histopathology imaging | **Not trained — stubbed** | No `.svs` files were uploaded. See `src/image_pipeline_stub/README.md` for exactly how to plug it in once slide data exists. |

**A data-leakage bug was caught and fixed during development:** predicting
smoking status using *other* exposure-table fields (e.g. `pack_years_smoked`)
would trivially "solve" the task without touching biology. `main.py` now
explicitly excludes all `exposures.*` fields from Task 3's inputs, and all
`diagnoses.ajcc*` (T/N/M) fields from Task 2's inputs (since stage is
literally derived from those). Metrics below are honest, post-fix numbers.

## Results (5-fold stratified cross-validation, not in-sample)

| Task | Best model | Accuracy | Macro F1 | Cohen's κ | ROC-AUC | n |
|---|---|---|---|---|---|---|
| 1. Tumor vs Normal | Logistic Regression | 0.977 | 0.894 | 0.788 | 1.000 | 88 samples |
| 2. Stage (Early vs Late) | LightGBM | 0.686 | 0.626 | 0.252 | 0.627 | 70 patients |
| 3. Smoking status (molecular) | Logistic Regression | 0.835 | 0.751 | 0.503 | 0.765 | 79 patients |
| 4. Survival (Cox) | Cox PH (PCA + clinical) | c-index = **0.721** | — | — | — | 82 patients |

Task 2's weaker score (κ=0.25, ROC-AUC=0.63, close to random) is the honest
result of only 70 patients spread across two imbalanced classes with heavy
omics dimensionality — flagged here rather than hidden. Task 1 and Task 3
are both strong and clinically plausible: tumor-vs-normal has a huge,
well-known expression signature, and there is real published literature on
smoking leaving a DNA-methylation signature (predicting it purely from
omics, with all direct exposure fields excluded, at ROC-AUC 0.77 is a
genuinely interesting result for a hackathon).

Task 4 discovered **2 molecular subtypes** (silhouette = 0.246, modest but
real cluster separation) — see the Analytics page for the subtype-vs-stage
cross-tab.

## Project structure

```
project/
├── data/
│   ├── raw/                  # your 7 input files
│   └── processed/            # patient/sample-level modeling tables (generated)
├── src/
│   ├── preprocessing/        # clinical/exposure/follow-up/pathology loader, genome + omics loaders
│   ├── training/              # classification.py (tasks 1-3), survival.py (task 4)
│   ├── explainability/       # SHAP (tree + linear models)
│   ├── inference/            # predict.py — CLI scoring
│   ├── utils/                # TCGA barcode parsing
│   ├── image_pipeline_stub/  # documented, inactive — see its README
│   └── build_dataset.py      # step 1: raw -> data/processed/*.csv
├── models/                   # saved best model per task (.joblib, generated)
├── reports/                  # evaluation JSON + SHAP figures (generated)
├── streamlit_app/            # multi-page dashboard
├── main.py                   # step 2: train + evaluate + explain all 4 tasks
├── config.yaml
├── requirements.txt
└── Dockerfile
```

## Running it

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv

# Step 1: raw TSVs -> processed modeling tables
python src/build_dataset.py

# Step 2: train, evaluate, explain all 4 tasks
python main.py

# Step 3: launch the app
streamlit run streamlit_app/app.py
```

Re-running `build_dataset.py` + `main.py` after dropping new files into
`data/raw/` (matching the same schema) will re-train everything from scratch
— nothing is hardcoded to this specific 82-patient cohort.

CLI single-patient scoring, without the app:
```bash
python src/inference/predict.py --task task2_stage --id TCGA-CR-7392
```

## Streamlit app pages

- **Dashboard** — cohort overview, KPIs, stage/smoking/vital-status breakdowns.
- **Upload Data** — schema-validating upload for MAF / transcriptome /
  methylation files (histopathology tab is disabled — see note above).
- **Prediction** — pick a patient/sample, see the model's prediction vs.
  ground truth, class probabilities, and a full-cohort batch prediction table.
- **Explainability** — SHAP importance plots per task; Cox hazard ratios and
  PCA-loading biomarkers for the survival model.
- **Analytics** — survival risk distribution, patient-similarity lookup,
  discovered-subtype breakdown.
- **Leaderboard** — every model tried on every task, cross-validated, sortable.

## Deployment

### Streamlit Community Cloud (simplest)
1. Push this repo to GitHub.
2. On https://share.streamlit.io, point at `streamlit_app/app.py`, and set
   the requirements file to `requirements.txt`.
3. Run `python src/build_dataset.py && python main.py` once locally first
   and commit `data/processed/`, `models/`, and `reports/` — the app reads
   pre-computed artifacts rather than training on every page load.

### Docker (Render / Railway / any container host)
```bash
docker build -t cancer-omics-platform .
docker run -p 8501:8501 cancer-omics-platform
```

### Local machine
```bash
streamlit run streamlit_app/app.py
```

## Methodology notes (for the judges)

- **Small-n, high-p handling:** every classification pipeline is
  `StandardScaler → SelectKBest(f_classif, k=50) → model`, evaluated with
  Stratified K-Fold (not a single holdout, which would be unstable at
  n=70-88) so the reported numbers are honest out-of-fold performance.
- **Survival modeling:** with 82 patients and ~4,000 raw omics features, a
  Cox model can't be fit directly (p >> n). We PCA-reduce the omics block to
  10 components, combine with age + mutation burden, and fit a penalized Cox
  model — then trace the top PCA loadings back to raw genes/probes for
  interpretable "biomarker" reporting.
- **Patient similarity / subtypes** are computed in the same PCA space via
  cosine similarity and KMeans (silhouette-selected k), respectively.
- **Leakage guards** are explicit and documented in `main.py` /
  `config.yaml` (`leakage_prefixes`) rather than implicit — this is the kind
  of thing that's easy to accidentally get "too good" results on.
