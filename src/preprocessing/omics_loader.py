"""
Shared loader for the two wide omics matrices (transcriptomics.txt,
methylation.txt). Both files share the same shape: first column is the
gene / probe id, every other column is a sample barcode, and the file is
gene-by-sample. We transpose to sample-by-gene, tag each sample with its
patient_id + sample_type (Tumor/Normal), and keep only the top-N most
variable features to keep things tractable (this is standard practice for
these matrix sizes: ~20k genes / 20k probes x <100 samples).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.barcode import patient_id_from_barcode, sample_type_from_barcode

logger = logging.getLogger(__name__)


def load_omics_matrix(
    path: str,
    top_n_features: int = 2000,
    log_transform: bool = False,
    prefix: str = "feat",
) -> pd.DataFrame:
    """Returns a DataFrame indexed by sample_barcode with columns:
    patient_id, sample_type, <prefix>_<gene1>, <prefix>_<gene2>, ...
    restricted to the top_n_features most variable rows in the raw matrix.
    """
    raw = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    raw = raw.apply(pd.to_numeric, errors="coerce")

    if log_transform:
        raw = np.log2(raw.clip(lower=0) + 1)

    variances = raw.var(axis=1, skipna=True)
    top_features = variances.sort_values(ascending=False).head(top_n_features).index
    filtered = raw.loc[top_features]

    mat = filtered.T  # samples as rows now
    mat.columns = [f"{prefix}_{c}" for c in mat.columns]
    mat = mat.fillna(mat.mean(numeric_only=True))

    mat["sample_barcode"] = mat.index
    mat["patient_id"] = mat["sample_barcode"].apply(patient_id_from_barcode)
    mat["sample_type"] = mat["sample_barcode"].apply(sample_type_from_barcode)

    front = ["sample_barcode", "patient_id", "sample_type"]
    mat = mat[front + [c for c in mat.columns if c not in front]]
    logger.info(
        "Loaded omics matrix %s: %d samples x %d features (top variable)",
        path, mat.shape[0], top_n_features,
    )
    return mat.reset_index(drop=True)
