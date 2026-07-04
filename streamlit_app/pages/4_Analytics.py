import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import REPORTS, load_csv, load_report
from ui import apply_base, hero, section, footer, style_fig, PALETTE, CHART_CONTINUOUS

apply_base("Analytics", "📊")

hero(
    title="Cohort Analytics",
    subtitle="Explore survival risk distributions, patient-to-patient similarity in multi-omics "
    "space, and unsupervised molecular subtypes discovered across the cohort.",
    eyebrow="Population Insights · Unsupervised Discovery",
    icon="📊",
)

section("Explore", "Three lenses on the cohort", "🔎")
tab1, tab2, tab3 = st.tabs(["❤️‍🩹 Survival Risk", "🕸️ Patient Similarity", "🧫 Cancer Subtypes"])

with tab1:
    st.write("")
    section("Survival risk score distribution", "How relative hazard spreads across risk groups", "📈")
    scores = pd.read_csv(f"{REPORTS}/task4_survival_risk_scores.csv") \
        if os.path.exists(f"{REPORTS}/task4_survival_risk_scores.csv") else None
    if scores is None:
        st.info("Run `python main.py` to generate survival analytics.")
    else:
        with st.container(border=True):
            fig = px.histogram(
                scores, x="survival_risk_score", color="risk_group", nbins=25,
                barmode="overlay", opacity=0.75,
            )
            fig.update_layout(height=360)
            st.plotly_chart(style_fig(fig), use_container_width=True)

        clinical = load_csv("clinical_master.csv")
        merged = scores.merge(
            clinical[["patient_id", "event_death", "survival_days", "age_at_index"]],
            on="patient_id", how="left",
        )
        st.write("")
        section("Risk vs survival time", "Each point is a patient; symbol marks death events", "🎯")
        with st.container(border=True):
            fig2 = px.scatter(
                merged, x="survival_days", y="survival_risk_score", color="risk_group",
                hover_data=["patient_id", "age_at_index"], symbol="event_death",
                labels={"event_death": "Event (1=death)"},
            )
            fig2.update_layout(height=420)
            st.plotly_chart(style_fig(fig2), use_container_width=True)

with tab2:
    st.write("")
    section("Patient similarity network", "Nearest neighbours in multi-omics PCA space", "🕸️")
    sim_path = f"{REPORTS}/task4_patient_similarity.json"
    if not os.path.exists(sim_path):
        st.info("Run `python main.py` to generate similarity analytics.")
    else:
        sim = load_report("task4_patient_similarity.json")
        patient_ids = [r["patient_id"] for r in sim]
        chosen = st.selectbox("Select a patient", patient_ids)
        record = next(r for r in sim if r["patient_id"] == chosen)
        neigh_df = pd.DataFrame({
            "Similar patient": record["most_similar_patients"],
            "Cosine similarity": record["similarity_scores"],
        })
        with st.container(border=True):
            fig = px.bar(
                neigh_df, x="Cosine similarity", y="Similar patient", orientation="h",
                color="Cosine similarity", color_continuous_scale=CHART_CONTINUOUS,
            )
            fig.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig), use_container_width=True)
            st.dataframe(neigh_df, use_container_width=True, hide_index=True)
            st.caption(
                "Similarity computed in multi-omics PCA space (transcriptome + methylation), "
                "cosine similarity between patients."
            )

with tab3:
    st.write("")
    section("Discovered molecular subtypes", "Unsupervised clustering of the cohort", "🧫")
    subtype_path = f"{REPORTS}/task4_subtype_discovery.csv"
    if not os.path.exists(subtype_path):
        st.info("Run `python main.py` to generate subtype analytics.")
    else:
        subtypes = pd.read_csv(subtype_path)
        t4 = load_report("task4_evaluation.json")

        with st.container(border=True):
            a, b = st.columns(2)
            a.metric("Optimal subtypes (silhouette)", t4.get("best_subtype_k", "?"))
            b.metric("Silhouette score", f"{t4.get('best_subtype_silhouette', 0):.3f}")

        counts = subtypes["discovered_subtype"].value_counts().reset_index()
        counts.columns = ["Subtype", "Patients"]
        st.write("")
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.container(border=True):
                st.markdown("**Patients per subtype**")
                fig = px.bar(counts, x="Subtype", y="Patients", color="Subtype")
                fig.update_layout(height=340, showlegend=False)
                st.plotly_chart(style_fig(fig), use_container_width=True)

        clinical = load_csv("clinical_master.csv")
        merged = subtypes.merge(clinical, on="patient_id", how="left")
        cross = pd.crosstab(merged["discovered_subtype"], merged["stage_group"])
        with col2:
            with st.container(border=True):
                st.markdown("**Subtype vs stage cross-tabulation**")
                st.dataframe(cross, use_container_width=True)

        st.write("")
        with st.container(border=True):
            st.markdown("**Full subtype assignments**")
            st.dataframe(subtypes, use_container_width=True, hide_index=True)

footer("Population Insights &nbsp;|&nbsp; Research preview — not for clinical use.")
