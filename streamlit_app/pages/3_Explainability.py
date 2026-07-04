import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import REPORTS, load_report, TASK_META
from ui import apply_base, hero, section, footer, pill, style_fig, CHART_CONTINUOUS

apply_base("Explainability", "🔍")

hero(
    title="Model Explainability",
    subtitle="SHAP feature importance for the classification tasks, plus Cox hazard ratios and "
    "PCA-loading biomarkers for survival — so every prediction can be traced back to biology.",
    eyebrow="Interpretability · SHAP · Hazard Ratios",
    icon="🔍",
)

section("By task", "Switch between tasks to inspect what drives each model", "🧩")

tab_labels = [TASK_META[k]["title"] for k in TASK_META] + ["Survival (Task 4)"]
tabs = st.tabs(tab_labels)

for tab, task_key in zip(tabs[:-1], TASK_META.keys()):
    with tab:
        img_path = f"{REPORTS}/figures/{task_key}_shap_importance.png"
        eval_report = load_report(f"{task_key}_evaluation.json")

        if eval_report:
            with st.container(border=True):
                a, b, c, d = st.columns(4)
                a.metric("Best model", str(eval_report.get("best_model")))
                b.metric("Samples", eval_report.get("n_samples"))
                c.metric("Features considered", eval_report.get("n_features_total"))
                d.metric("Top selected", eval_report.get("k_best_selected"))

        st.write("")
        section("SHAP feature importance", "Mean absolute SHAP value per feature", "📊")
        with st.container(border=True):
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info("SHAP plot not available for this task's winning model.")

        top_feats = load_report(f"{task_key}_shap_top_features.json")
        if top_feats:
            feat_df = pd.DataFrame(
                {"Feature": list(top_feats.keys()), "Mean |SHAP value|": list(top_feats.values())}
            )
            with st.expander("Show top features as a table"):
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

with tabs[-1]:
    t4 = load_report("task4_evaluation.json")
    if not t4:
        st.info("Run `python main.py` to generate Task 4 explainability outputs.")
    else:
        with st.container(border=True):
            a, b = st.columns(2)
            a.metric("Cox concordance index", f"{t4.get('cox_concordance_index', 0):.3f}")
            b.metric("PCA components used", t4.get("pca_components_used"))

        st.write("")
        section("Top hazard ratios", "Covariate risk contribution from the Cox model", "⚠️")
        hr = t4.get("top_hazard_ratios", {})
        hr_df = pd.DataFrame({"Covariate": list(hr.keys()), "Hazard ratio": list(hr.values())})
        hr_df["Effect"] = hr_df["Hazard ratio"].apply(
            lambda x: "Increases risk" if x > 1 else "Decreases risk"
        )
        with st.container(border=True):
            fig = px.bar(
                hr_df.sort_values("Hazard ratio"), x="Hazard ratio", y="Covariate",
                orientation="h", color="Effect",
                color_discrete_map={"Increases risk": "#E5484D", "Decreases risk": "#1FA971"},
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="#5B7189")
            fig.update_layout(height=380)
            st.plotly_chart(style_fig(fig), use_container_width=True)
            st.dataframe(hr_df, use_container_width=True, hide_index=True)
            st.caption(
                "Hazard ratio > 1 means higher values of that covariate/PC are associated "
                "with higher risk of death; < 1 means protective."
            )

        st.write("")
        section(
            "Candidate biomarkers",
            "Genes/probes driving the top principal components",
            "🧬",
        )
        st.caption(
            "With ~4,000 omics features and only 82 patients, we reduce to PCA components for "
            "the Cox model, then trace back which raw genes/probes load most heavily onto each "
            "component — these are the candidate biomarkers."
        )
        for pc, genes in t4.get("top_biomarkers_by_pc", {}).items():
            with st.expander(f"{pc} — top contributing features"):
                gdf = pd.DataFrame(
                    {"Feature": list(genes.keys()), "Loading (abs)": list(genes.values())}
                )
                st.dataframe(gdf, use_container_width=True, hide_index=True)

footer("Interpretability &nbsp;|&nbsp; Research preview — not for clinical use.")
