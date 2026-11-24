"""The six cluster-level health indicators that the pipeline predicts.

In the competition the labels come pre-computed in ``training_label.csv``:
one row per DHS cluster (a village or urban block of ~20-30 surveyed
households). This module records what each target means, using the standard
DHS definitions, and gives a quick summary of the label table.

Reference: Guide to DHS Statistics, https://dhsprogram.com/data/Guide-to-DHS-Statistics/
"""
import pandas as pd

from . import config

TARGET_INFO = {
    "Mean_BMI": "Mean body-mass index (kg/m^2) of surveyed women in the cluster.",
    "Median_BMI": "Median BMI of the same women; less sensitive to extreme values.",
    "Unmet_Need_Rate": "Share of women who want to stop or delay childbearing but use no contraception.",
    "Under5_Mortality_Rate": "Probability of dying before age 5 (child deaths in the cluster's birth histories).",
    "Skilled_Birth_Attendant_Rate": "Share of recent births attended by a doctor, nurse or midwife.",
    "Stunted_Rate": "Share of under-5 children with height-for-age z-score below -2 SD (WHO standard).",
}

assert set(TARGET_INFO) == set(config.TARGET_COLUMNS)


def describe_targets(labels: pd.DataFrame) -> pd.DataFrame:
    """Count, missing share and spread of each target. Look at this before modelling."""
    rows = []
    for t in config.TARGET_COLUMNS:
        s = labels[t]
        rows.append({"target": t, "labelled": int(s.notna().sum()),
                     "missing_%": round(100 * s.isna().mean(), 1),
                     "mean": round(s.mean(), 3), "std": round(s.std(), 3),
                     "min": round(s.min(), 3), "max": round(s.max(), 3)})
    return pd.DataFrame(rows).set_index("target")
