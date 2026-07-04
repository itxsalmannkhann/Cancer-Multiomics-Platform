"""
TCGA barcode parsing utilities.

A full TCGA aliquot barcode looks like:
    TCGA-BA-4074-01A-01R-1436-07
     |    |   |    |  |   |    |
     |    |   |    |  |   |    plate
     |    |   |    |  |   center
     |    |   |    |  portion-analyte
     |    |   |    sample-vial  (01 = primary solid tumor, 11 = normal solid tissue)
     |    |   patient number
     |    tissue source site
     project

We only ever need two derived keys:
    patient_id   -> "TCGA-BA-4074"                (join key across all modalities)
    sample_type  -> "Tumor" / "Normal" / "Other"   (derived from the 4th barcode field)
"""
from __future__ import annotations

NORMAL_CODES = {"10", "11", "12", "13", "14"}  # blood/solid tissue normal codes
TUMOR_CODES = {"01", "02", "03", "04", "05", "06", "07", "08", "09"}


def patient_id_from_barcode(barcode: str) -> str:
    """TCGA-BA-4074-01A-01R-1436-07 -> TCGA-BA-4074"""
    parts = str(barcode).split("-")
    if len(parts) < 3:
        return str(barcode)
    return "-".join(parts[:3])


def sample_type_from_barcode(barcode: str) -> str:
    """Return 'Tumor', 'Normal', or 'Other' based on the sample-vial code."""
    parts = str(barcode).split("-")
    if len(parts) < 4:
        return "Other"
    code = "".join(ch for ch in parts[3] if ch.isdigit())[:2]
    if code in TUMOR_CODES:
        return "Tumor"
    if code in NORMAL_CODES:
        return "Normal"
    return "Other"
