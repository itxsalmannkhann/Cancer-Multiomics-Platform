import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_csv, load_model_bundle, load_survival_bundle, TASK_META
from ui import apply_base, hero, section, footer, pill, style_fig, PALETTE, CHART_CONTINUOUS

apply_base("Prediction", "🔮")

hero(
    title="Patient-Level Prediction",
    subtitle="Score any patient in the cohort against the trained multi-omics models — tumor "
    "classification, staging, risk factors and survival hazard — with transparent probabilities.",
    eyebrow="Inference · Clinical Decision Support",
    icon="🔮",
)

section("Select a task", "Pick which trained model to run", "🎯")
task_key = st.selectbox(
    "Choose a prediction task",
    options=list(TASK_META.keys()) + ["task4_survival"],
    format_func=lambda k: "Survival Risk Score (Task 4)" if k == "task4_survival"
    else TASK_META[k]["title"],
    label_visibility="collapsed",
)

if task_key == "task4_survival":
    bundle = load_survival_bundle()
    st.write("")
    section("Survival risk score", "Relative hazard from the Cox Proportional Hazards model", "❤️‍🩹")
    if bundle is None:
        st.error("Survival model not found. Run `python main.py` first.")
    else:
        from data_utils import REPORTS
        scores = pd.read_csv(f"{REPORTS}/task4_survival_risk_scores.csv")
        patient = st.selectbox("Select patient", scores["patient_id"].unique())
        row = scores[scores["patient_id"] == patient].iloc[0]

        risk_group = str(row["risk_group"])
        pill_kind = "bad" if "high" in risk_group.lower() else ("warn" if "med" in risk_group.lower() else "ok")

        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("Survival risk score (relative hazard)", f"{row['survival_risk_score']:.3f}")
            c2.metric("Risk group", risk_group)
            st.markdown(pill(f"Risk group: {risk_group}", pill_kind), unsafe_allow_html=True)
            st.caption(
                "Risk score = predicted partial hazard from the Cox Proportional Hazards model "
                "fitted on PCA-reduced multi-omics features + age + mutation burden. Higher = "
                "higher relative risk of the event (death) within this cohort."
            )

        st.write("")
        section("Cohort ranking", "All patients sorted by descending relative hazard", "📋")
        with st.container(border=True):
            st.dataframe(
                scores.sort_values("survival_risk_score", ascending=False),
                use_container_width=True, hide_index=True,
            )
else:
    meta = TASK_META[task_key]
    bundle = load_model_bundle(task_key)
    df = load_csv(meta["data_file"])

    if bundle is None:
        st.error(f"Model for {task_key} not found. Run `python main.py` first.")
    else:
        pipe = bundle["pipeline"]
        le = bundle["label_encoder"]
        feature_cols = bundle["feature_cols"]

        st.write("")
        section(f"Predict: {meta['title']}", "Choose a sample and inspect the model's decision", "🔬")
        id_col = "sample_barcode" if "sample_barcode" in df.columns else "patient_id"
        selection = st.selectbox("Select a sample/patient from the cohort", df[id_col].unique())
        row = df[df[id_col] == selection].iloc[[0]]

        X = row[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        pred_idx = pipe.predict(X)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        proba = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)[0]

        actual = row[meta["label_col"]].iloc[0]
        is_correct = str(pred_label) == str(actual)

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted", pred_label)
            c2.metric("Actual (ground truth)", actual)
            c3.metric("Correct?", "✅ Yes" if is_correct else "❌ No")
            st.markdown(
                pill("Prediction matches ground truth", "ok") if is_correct
                else pill("Prediction differs from ground truth", "bad"),
                unsafe_allow_html=True,
            )

        if proba is not None:
            st.write("")
            section("Class probabilities", "Model confidence across every possible class", "📊")
            with st.container(border=True):
                proba_df = pd.DataFrame({"Class": le.classes_, "Probability": proba})
                fig = px.bar(
                    proba_df, x="Probability", y="Class", orientation="h",
                    color="Probability", color_continuous_scale=CHART_CONTINUOUS,
                )
                fig.update_layout(height=320, coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig), use_container_width=True)

        with st.expander("Show raw feature values used for this prediction"):
            st.dataframe(X.T.rename(columns={X.index[0]: "value"}), use_container_width=True)

        st.write("")
        section("Batch view", "Predictions across the full cohort", "🗂️")
        X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        preds_all = le.inverse_transform(pipe.predict(X_all))
        out = df[[id_col, meta["label_col"]]].copy()
        out["predicted"] = preds_all
        out["correct"] = out[meta["label_col"]].astype(str) == out["predicted"].astype(str)

        acc = out["correct"].mean()
        with st.container(border=True):
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.caption(
                f"Overall accuracy on full (in-sample) cohort: {acc:.1%} "
                "— see the Leaderboard page for honest cross-validated performance."
            )

footer("Inference &nbsp;|&nbsp; Research preview — not for clinical use.")
