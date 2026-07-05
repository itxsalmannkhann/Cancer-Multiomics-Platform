# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

_No unreleased changes yet._

## [1.1.0] — 2026-07-05

### Added
- **Centralized UI theme** (`streamlit_app/ui/theme.py`, `streamlit_app/ui/__init__.py`)
  as the single source of truth for the dashboard's visual language: clinical
  color palette, shared Plotly template, global CSS, and reusable components
  (`apply_base`, `hero`, `section`, `pill`, `footer`, `style_fig`).
- **Modern, professional healthcare UI** applied consistently across all pages
  (Dashboard, Upload Data, Prediction, Explainability, Analytics, Leaderboard)
  with hero headers, section titles, metric cards, styled tabs, and bordered
  card containers.
- **Sidebar credits** component pinned to the bottom of the sidebar on every
  page: "Built by Salman Khan & Team — NeoHack GIKI Edition 2025, in
  collaboration with Precision Medicine Lab."
- **scikit-learn version-compatibility shim** in `streamlit_app/data_utils.py`
  (`_patch_sklearn_compat`) that backfills the `multi_class` attribute on
  pickled `LogisticRegression` estimators, walking nested pipeline/meta
  estimators.
- Project governance and documentation files: `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, and `.gitignore`.

### Changed
- Rewrote all `streamlit_app/pages/*.py` to consume the shared theme instead of
  raw `st.title` / default Streamlit styling, while keeping all data-loading and
  model logic unchanged.
- Prediction page now renders class probabilities as a themed Plotly bar chart
  and surfaces correctness/risk-group status pills.
- Explainability page adds a colored hazard-ratio chart (risk-increasing vs
  protective) with a reference line at HR = 1.
- Resolved leftover git merge-conflict markers in `README.md` and expanded it
  with a summary of UI and platform changes.

### Fixed
- **`AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`**
  on the Prediction page. Models were pickled with scikit-learn 1.8.0 but the
  runtime had 1.7.2, whose `predict_proba` still reads `self.multi_class`. The
  loader now restores the attribute (`"auto"`) at load time so predictions work
  across affected tasks.

## [1.0.0] — NeoHack GIKI Edition 2025

### Added
- End-to-end multi-omics ML pipeline for a TCGA Head & Neck Squamous Cell
  Carcinoma (HNSC) cohort.
- Four tasks: Tumor vs Normal classification, Stage (Early vs Late)
  classification, Smoking status from molecular signature, and Cox survival
  risk scoring with subtype discovery and patient similarity.
- SHAP-based explainability, cross-validated model leaderboard, and a
  multi-page Streamlit dashboard.
- Documented, inactive histopathology image pipeline stub
  (`src/image_pipeline_stub/`).
- Explicit data-leakage guards for Task 2 (AJCC T/N/M fields) and Task 3
  (exposure fields).
