"""
Genome pipeline: turns the per-mutation MAF-style table into one row of
numerical features per patient (mutation burden, driver-gene mutation flags,
variant-classification counts).
"""
from __future__ import annotations

import logging

import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.barcode import patient_id_from_barcode

logger = logging.getLogger(__name__)

# Well-known driver / frequently mutated genes in HNSC (and cancer generally).
# Used to build interpretable binary "is this gene mutated in this patient" features.
DRIVER_GENES = [
    "TP53", "CDKN2A", "PIK3CA", "NOTCH1", "FAT1", "CASP8", "HRAS",
    "TTN", "CSMD3", "MUC16", "FLG", "LRP1B", "PCLO", "SYNE1", "MUC17",
]

DELETERIOUS_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}


def build_genome_features(maf_path: str) -> pd.DataFrame:
    usecols = [
        "Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "Variant_Type",
    ]
    df = pd.read_csv(maf_path, sep="\t", usecols=usecols, low_memory=False, encoding="latin-1")
    df["patient_id"] = df["Tumor_Sample_Barcode"].apply(patient_id_from_barcode)
    df["is_deleterious"] = df["Variant_Classification"].isin(DELETERIOUS_CLASSES).astype(int)

    patients = df["patient_id"].unique()
    out = pd.DataFrame({"patient_id": patients}).set_index("patient_id")

    # Overall tumor mutation burden
    burden = df.groupby("patient_id").size().rename("mutation_burden")
    del_burden = df.groupby("patient_id")["is_deleterious"].sum().rename("deleterious_mutation_burden")
    out = out.join(burden).join(del_burden)

    # Variant classification composition
    vc_counts = df.pivot_table(
        index="patient_id", columns="Variant_Classification", values="Hugo_Symbol",
        aggfunc="count", fill_value=0,
    )
    vc_counts.columns = [f"vc_{c.lower()}" for c in vc_counts.columns]
    out = out.join(vc_counts)

    # Driver-gene binary flags + mutation counts
    for gene in DRIVER_GENES:
        gene_df = df[df["Hugo_Symbol"] == gene]
        flag = gene_df.groupby("patient_id").size()
        out[f"gene_{gene}_mutated"] = out.index.map(lambda p: 1 if p in flag.index else 0)
        out[f"gene_{gene}_count"] = out.index.map(lambda p: flag.get(p, 0))

    out = out.fillna(0).reset_index()
    logger.info("Genome features built for %d patients, %d features", out.shape[0], out.shape[1] - 1)
    return out
