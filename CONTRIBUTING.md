# Contributing Guide

Thanks for your interest in the **Cancer Multi-Omics Intelligence Platform**.
This project was built for **NeoHack — GIKI Edition 2025** in collaboration
with the **Precision Medicine Lab**. Contributions that improve the pipeline,
the Streamlit dashboard, documentation, or reproducibility are welcome.

> ⚠️ **Research preview — not for clinical use.** This project handles
> de-identified TCGA research data. Do not commit any real patient-identifiable
> information.

---

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Project setup](#project-setup)
- [Project structure](#project-structure)
- [Development workflow](#development-workflow)
- [Coding standards](#coding-standards)
- [UI / theming conventions](#ui--theming-conventions)
- [Data & model conventions](#data--model-conventions)
- [Commit & pull request guidelines](#commit--pull-request-guidelines)
- [Reporting bugs](#reporting-bugs)
- [Requesting features](#requesting-features)

---

## Code of conduct

By participating you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).
Be respectful, constructive, and collaborative.

---

## Ways to contribute

- **Bug fixes** — especially data-leakage guards, reproducibility, or version
  compatibility issues.
- **New analyses** — additional tasks, models, or explainability views.
- **UI/UX** — improvements to the Streamlit dashboard within the shared theme.
- **Documentation** — clarifying setup, methodology, or results.
- **Image pipeline** — activating `src/image_pipeline_stub/` once `.svs`
  histopathology slides are available.

---

## Project setup

Requires **Python 3.12+**.

```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/cancer-multiomics-platform.git
cd cancer-multiomics-platform

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate processed data + train models (one-time)
python src/build_dataset.py
python main.py

# 5. Launch the dashboard
streamlit run streamlit_app/app.py
```

---

## Project structure

```
project/
├── data/
│   ├── raw/                  # input .txt files (not committed if large)
│   └── processed/            # generated modeling tables
├── src/
│   ├── preprocessing/        # clinical/exposure/omics loaders
│   ├── training/             # classification.py, survival.py
│   ├── explainability/       # SHAP utilities
│   ├── inference/            # predict.py CLI
│   ├── utils/                # TCGA barcode parsing
│   ├── image_pipeline_stub/  # inactive, documented
│   └── build_dataset.py
├── models/                   # saved .joblib models (generated)
├── reports/                  # evaluation JSON + SHAP figures (generated)
├── streamlit_app/
│   ├── app.py                # dashboard entry point
│   ├── data_utils.py         # cached data/model loaders
│   ├── ui/                   # shared theme + components (single source of truth)
│   └── pages/                # multi-page dashboard
├── main.py                   # train + evaluate + explain all tasks
├── requirements.txt
└── Dockerfile
```

---

## Development workflow

1. **Create a branch** off `main`:
   ```bash
   git checkout -b feat/short-description
   ```
2. Make focused changes. Keep pull requests small and single-purpose.
3. Verify the app still launches and pages render without errors.
4. If you touch training or preprocessing, re-run:
   ```bash
   python src/build_dataset.py && python main.py
   ```
5. Commit using the [conventions below](#commit--pull-request-guidelines).
6. Push and open a pull request against `main`.

---

## Coding standards

- **Language:** Python 3.12+.
- **Style:** follow [PEP 8](https://peps.python.org/pep-0008/); 4-space indent,
  `snake_case` for functions/variables, `PascalCase` for classes.
- **Imports:** standard library, third-party, then local — separated by blank
  lines.
- **Docstrings:** module- and function-level docstrings for non-trivial logic.
- **No secrets** in code or commits.
- Prefer small, pure, testable functions. Keep data-loading logic in
  `streamlit_app/data_utils.py` and visual logic in `streamlit_app/ui/`.

---

## UI / theming conventions

The dashboard's visual language lives in **one place**: `streamlit_app/ui/theme.py`.

- Always call `apply_base(page_title, page_icon)` at the top of every page.
- Use the shared components: `hero()`, `section()`, `pill()`, `footer()`,
  `style_fig()`, and the `PALETTE` / `CHART_*` constants.
- Do **not** hardcode colors or inject page-specific CSS — extend `theme.py`
  instead so the look stays consistent across pages.
- Style Plotly charts with `style_fig(fig)` and the shared color scales.

---

## Data & model conventions

- **Never commit raw patient-identifiable data.** Only de-identified TCGA
  research tables belong in `data/raw/`.
- Generated artifacts (`data/processed/`, `models/`, `reports/`) are outputs of
  `build_dataset.py` + `main.py`. Commit them only when needed for deployment
  (the app reads pre-computed artifacts).
- **Guard against leakage.** Any field derived from the label must be excluded
  from features (see `leakage_prefixes` handling in `main.py`). If you add a
  task, document its leakage guards.
- **Reproducibility:** pin library versions when a model artifact depends on
  them. Models are pickled with a specific scikit-learn version — keep training
  and serving environments aligned to avoid `InconsistentVersionWarning`.

---

## Commit & pull request guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add molecular subtype export to Analytics page
fix: restore multi_class attribute for pickled LogisticRegression
docs: expand methodology notes in README
style: align sidebar credits with theme palette
refactor: centralize chart styling in theme.py
```

**Pull requests should include:**
- A clear description of *what* changed and *why*.
- Screenshots for any UI change.
- Confirmation that the app launches and affected pages render.
- Notes on any new dependency or data/model regeneration required.

---

## Reporting bugs

Open an issue with:
- What you expected vs. what happened.
- Steps to reproduce.
- Full traceback / error message.
- Environment (OS, Python version, `scikit-learn` version).

---

## Requesting features

Open an issue describing the use case, the proposed behavior, and how it fits
the platform's scope. For substantial changes, discuss in an issue before
opening a large pull request.

---

Thank you for helping improve the platform. 🧬
