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
