import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_csv, load_report
from ui import apply_base, hero, section, footer, style_fig, PALETTE, CHART_CONTINUOUS

apply_base("Cancer Multi-Omics Platform", "🧬")

hero(
    title="Cancer Multi-Omics Intelligence Platform",
    subtitle="Head & Neck Squamous Cell Carcinoma (TCGA) · integrating genome, "
    "transcriptome, epigenome and clinical evidence into one decision surface.",
    eyebrow="Clinical Decision Support · Research Preview",
    icon="🧬",
)

# ----------------------------------------------------------------- load data
clinical = load_csv("clinical_master.csv")
genome = load_csv("genome_features.csv")
expr = load_csv("transcriptome_matrix.csv")
meth = load_csv("methylation_matrix.csv")
summary = load_report("all_tasks_summary.json")

# ----------------------------------------------------------------- KPI row
section("Cohort at a glance", "Key figures across the integrated multi-omics dataset", "📌")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Patients", clinical["patient_id"].nunique())
c2.metric("Tumor samples", int((expr["sample_type"] == "Tumor").sum()))
c3.metric("Normal samples", int((expr["sample_type"] == "Normal").sum()))
c4.metric("Genes profiled", expr.filter(like="expr_").shape[1])
c5.metric("Methylation probes", meth.filter(like="meth_").shape[1])

st.write("")
st.info(
    "**Data scope:** this cohort has no histopathology (`.svs`) images — only genome "
    "(mutations), transcriptome (RNA-seq), epigenome (methylation) and clinical / exposure / "
    "follow-up / pathology tables. The image pipeline is scaffolded in "
    "`src/image_pipeline_stub/` and activates automatically once `.svs` files are added.",
    icon="ℹ️",
)

# ----------------------------------------------------------------- cohort charts
st.write("")
section("Cohort composition", "Where the tumors originate and how they are staged", "🧫")
col1, col2 = st.columns([1.1, 1])

with col1:
    with st.container(border=True):
        st.markdown("**Primary tumor site**")
        site_counts = clinical["cases.primary_site"].value_counts().reset_index()
        site_counts.columns = ["Primary site", "Patients"]
        fig = px.bar(
            site_counts, x="Patients", y="Primary site", orientation="h",
            color="Patients", color_continuous_scale=CHART_CONTINUOUS,
        )
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown("**AJCC pathologic stage**")
        stage_counts = clinical["ajcc_stage"].fillna("Unknown/Not staged").value_counts().reset_index()
        stage_counts.columns = ["Stage", "Patients"]
        fig = px.pie(stage_counts, names="Stage", values="Patients", hole=0.55)
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    with st.container(border=True):
        st.markdown("**Vital status**")
        vs = clinical["demographic.vital_status"].fillna("Unknown").value_counts().reset_index()
        vs.columns = ["Status", "Patients"]
        fig = px.pie(
            vs, names="Status", values="Patients", hole=0.55,
            color="Status",
            color_discrete_map={"Alive": PALETTE["healthy"], "Dead": PALETTE["danger"]},
        )
        fig.update_traces(textposition="outside", textinfo="label+value")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

with col4:
    with st.container(border=True):
        st.markdown("**Smoking status**")
        sm = clinical["smoking_status"].fillna("Unknown").value_counts().reset_index()
        sm.columns = ["Status", "Patients"]
        fig = px.bar(sm, x="Status", y="Patients", color="Status")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ----------------------------------------------------------------- model summary
st.write("")
section("Model performance at a glance", "Cross-validated headline metrics per task", "🎯")

rows = []
for tkey in ["task1", "task2", "task3"]:
    rep = summary.get(tkey, {})
    if not rep or "best_model" not in rep:
        continue
    best = rep["best_model"]
    m = rep["all_model_results"][best]
    rows.append({
        "Task": rep["task"],
        "Best model": best,
        "Macro F1": round(m["macro_f1"], 3),
        "Accuracy": round(m["accuracy"], 3),
        "Cohen's Kappa": round(m["cohen_kappa"], 3),
        "N samples": rep["n_samples"],
    })
if rows:
    with st.container(border=True):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

t4 = summary.get("task4", {})
if t4:
    a, b, c = st.columns(3)
    a.metric("Survival c-index (Cox)", f"{t4.get('cox_concordance_index', 0):.3f}")
    b.metric("Molecular subtypes", t4.get("best_subtype_k", "?"))
    c.metric("Subtype silhouette", f"{t4.get('best_subtype_silhouette', 0):.3f}")

footer(
    "Navigate via the sidebar → Upload Data · Prediction · Explainability · Analytics · "
    "Leaderboard &nbsp;|&nbsp; Research preview — not for clinical use."
)
