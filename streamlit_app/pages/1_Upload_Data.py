import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui import apply_base, hero, section, footer, pill

apply_base("Upload Data", "📤")

hero(
    title="Upload & Validate Omics Data",
    subtitle="Bring new mutation, transcriptome or methylation files into the platform to "
    "re-score patients or extend the cohort. Files are validated against the training schema "
    "before they touch any model.",
    eyebrow="Data Ingestion · Schema Validation",
    icon="📤",
)

section(
    "Accepted inputs",
    "The trained models expect the same schema produced by src/build_dataset.py",
    "📁",
)

with st.container(border=True):
    st.markdown(
        """
        - **MAF** (`.maf` / `.tsv`) — mutation calls, same columns as `mutations.txt`
        - **Transcriptome matrix** (`.tsv`) — genes × samples, same layout as `transcriptomics.txt`
        - **Methylation matrix** (`.tsv`) — probes × samples, same layout as `methylation.txt`
        - **Histopathology slide** (`.svs`) — *pipeline not active in this deployment (no image
          training data was provided); wiring lives in `src/image_pipeline_stub/`.*
        """
    )

st.write("")
section("Upload a file", "Pick the modality below — each upload is parsed and previewed instantly", "⬆️")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🧬 Mutations (MAF)", "📈 Transcriptome", "🔬 Methylation", "🖼️ Histopathology (.svs)"]
)

with tab1:
    maf_file = st.file_uploader("Upload mutation file", type=["maf", "tsv", "txt"], key="maf")
    if maf_file is not None:
        try:
            df = pd.read_csv(maf_file, sep="\t", nrows=200, low_memory=False, encoding="latin-1")
            st.success(f"Parsed {df.shape[0]} preview rows, {df.shape[1]} columns.")
            st.dataframe(df.head(20), use_container_width=True)
            required = {"Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"}
            missing = required - set(df.columns)
            if missing:
                st.markdown(pill(f"Missing expected columns: {missing}", "warn"), unsafe_allow_html=True)
            else:
                st.markdown(
                    pill("Schema compatible with genome_features.py", "ok"),
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Could not parse file: {e}")

with tab2:
    expr_file = st.file_uploader("Upload transcriptome matrix", type=["tsv", "txt"], key="expr")
    if expr_file is not None:
        try:
            df = pd.read_csv(expr_file, sep="\t", nrows=20)
            st.success(f"Parsed preview: {df.shape[0]} genes (rows shown) × {df.shape[1]} columns.")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not parse file: {e}")

with tab3:
    meth_file = st.file_uploader("Upload methylation matrix", type=["tsv", "txt"], key="meth")
    if meth_file is not None:
        try:
            df = pd.read_csv(meth_file, sep="\t", nrows=20)
            st.success(f"Parsed preview: {df.shape[0]} probes (rows shown) × {df.shape[1]} columns.")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not parse file: {e}")

with tab4:
    st.warning(
        "No `.svs` histopathology files were included in the uploaded dataset, so no image "
        "model was trained. Uploading a slide here will not produce a prediction yet — this "
        "tab is a placeholder showing where the image pipeline plugs in once slide data and "
        "labels are available."
    )
    st.file_uploader("Upload histopathology slide (inactive)", type=["svs"], key="svs", disabled=True)

st.write("")
st.info(
    "Uploaded files are held only in this browser session and are not written back into the "
    "trained models automatically. To retrain on new data, add files to `data/raw/` and re-run "
    "`python src/build_dataset.py && python main.py`.",
    icon="ℹ️",
)

footer("Data Ingestion &nbsp;|&nbsp; Research preview — not for clinical use.")
