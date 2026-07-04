import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_report
from ui import apply_base, hero, section, footer, pill, style_fig

apply_base("Leaderboard", "🏆")

hero(
    title="Model Leaderboard",
    subtitle="Cross-validated performance (Stratified K-Fold) for every model tried on every "
    "task. These are honest out-of-fold scores — not in-sample fit.",
    eyebrow="Benchmarks · Cross-Validated Metrics",
    icon="🏆",
)

summary = load_report("all_tasks_summary.json")

for tkey in ["task1", "task2", "task3"]:
    rep = summary.get(tkey)
    if not rep or "all_model_results" not in rep:
        continue

    section(
        rep["task"],
        f"{rep['n_samples']} samples · {rep['cv_folds']}-fold CV · classes: {', '.join(rep['classes'])}",
        "🎯",
    )

    rows = []
    for model_name, m in rep["all_model_results"].items():
        rows.append({
            "Model": model_name,
            "Accuracy": m["accuracy"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
            "Precision (macro)": m["precision_macro"],
            "Recall (macro)": m["recall_macro"],
            "Cohen's Kappa": m["cohen_kappa"],
            "ROC-AUC": m.get("roc_auc"),
        })
    df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)

    with st.container(border=True):
        st.markdown(pill(f"🥇 Best model: {rep['best_model']}", "ok"), unsafe_allow_html=True)
        st.write("")
        st.dataframe(
            df.style.highlight_max(
                subset=["Accuracy", "Macro F1", "Weighted F1"], color="#1FA971"
            ),
            use_container_width=True, hide_index=True,
        )
        fig = px.bar(df, x="Model", y=["Accuracy", "Macro F1", "Weighted F1"], barmode="group")
        fig.update_layout(height=360)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.write("")

t4 = summary.get("task4")
if t4:
    section(t4["task"], "Survival ranking and subtype separation quality", "❤️‍🩹")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Cox concordance index", f"{t4['cox_concordance_index']:.3f}")
        c2.metric("Discovered subtypes (k)", t4["best_subtype_k"])
        c3.metric("Subtype silhouette", f"{t4['best_subtype_silhouette']:.3f}")
        st.caption(
            "Concordance index (c-index) measures how well the model ranks patients by risk — "
            "0.5 = random, 1.0 = perfect ranking. Silhouette score measures how well-separated "
            "the discovered subtypes are (-1 to 1, higher is better)."
        )

footer("Benchmarks &nbsp;|&nbsp; Research preview — not for clinical use.")
