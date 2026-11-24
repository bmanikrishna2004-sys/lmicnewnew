"""Synthetic data with exactly the competition's file layout.

The real files (an 8.15 GB GEE table + DHS labels) are only available inside
the Kaggle competition. This module writes small stand-ins with the same
names, columns and quirks, so the real pipeline can be run, tested and
learned from on a laptop:

* ``training_label.csv``   - DHSID + 6 targets, with per-target missing labels
                              and one duplicated DHSID.
* ``gee_features.csv``     - DHSID + metadata + numeric features, including
                              near-empty and constant columns the screen must drop,
                              one duplicated DHSID, and unrelated extra rows.
* ``gee_features_10pct.csv``- a 10% row sample of the above.
* ``sample submission.csv`` - DHSIDs to predict; a few have no GEE row.

Only a handful of features carry signal, mixed with many noise features,
which is the situation feature selection is meant to handle. The numbers are
not real health statistics.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

COUNTRIES = ["KE", "NG", "ET", "BD", "IA", "TZ", "UG", "GH", "MW", "NP", "BO", "PE"]
FAMILIES = ["viirs_ntl", "ndvi", "lst_day", "lst_night", "chirps_precip", "worldpop",
            "srtm_elev", "ghsl_built", "dist_road", "dist_city", "evi", "pdsi",
            "surface_water", "landcover_crop", "landcover_urban", "travel_time"]
STATS = ["mean", "median", "std", "min", "max", "p25", "p75"]

# (mean, sd, low, high, label missing-rate) - plausible scales, not real values.
TARGET_SPEC = {
    "Mean_BMI": (23.0, 2.0, 15, 40, 0.15),
    "Median_BMI": (22.5, 1.9, 15, 40, 0.15),
    "Unmet_Need_Rate": (18.0, 9.0, 0, 100, 0.10),
    "Under5_Mortality_Rate": (8.0, 5.0, 0, 100, 0.05),
    "Skilled_Birth_Attendant_Rate": (60.0, 25.0, 0, 100, 0.05),
    "Stunted_Rate": (30.0, 14.0, 0, 100, 0.30),
}


def generate(n_train=1500, n_test=400, n_extra=300, n_features=600, n_signal=15,
             seed=config.RANDOM_STATE):
    rng = np.random.RandomState(seed)
    n_all = n_train + n_test + n_extra

    country = rng.choice(COUNTRIES, n_all)
    year = rng.choice([2014, 2016, 2018, 2021], n_all)
    ids = [f"{c}{y}{i:08d}" for i, (c, y) in enumerate(zip(country, year))]

    names = [f"{FAMILIES[i % len(FAMILIES)]}_{STATS[(i // len(FAMILIES)) % len(STATS)]}_{i:03d}"
             for i in range(n_features)]
    # Country-level offsets make clusters in the same country alike (spatial correlation).
    country_effect = {c: rng.normal(scale=0.7, size=n_features) for c in COUNTRIES}
    X = rng.normal(size=(n_all, n_features)) + np.stack([country_effect[c] for c in country])

    signal_idx = rng.choice(n_features, n_signal, replace=False)
    signal = X[:, signal_idx]

    feats = pd.DataFrame(X.astype("float32"), columns=names)
    # Columns the quality screen should drop.
    feats.iloc[:, -5:] = np.nan                              # entirely missing
    feats.iloc[:, -10:-5] = 1.0                              # constant
    feats.iloc[:, :n_features - 10] = feats.iloc[:, :n_features - 10].mask(
        rng.rand(n_all, n_features - 10) < 0.03)             # 3% scattered missing
    meta = pd.DataFrame({
        config.ID_COLUMN: ids,
        config.YEAR_COLUMN: year,
        config.CLUSTER_COLUMN: np.arange(n_all),
        config.LAT_COLUMN: rng.uniform(-15, 28, n_all).round(5),
        config.LON_COLUMN: rng.uniform(-75, 90, n_all).round(5),
        "ADM1NAME": [f"Region_{c}_{k}" for c, k in zip(country, rng.randint(0, 8, n_all))],
    })
    gee = pd.concat([meta, feats], axis=1)

    labels = pd.DataFrame({config.ID_COLUMN: ids[:n_train]})
    for target, (mu, sd, lo, hi, miss) in TARGET_SPEC.items():
        w = rng.uniform(-1, 1, n_signal)
        values = mu + sd * (0.8 * np.tanh(signal[:n_train] @ w / 3) + 0.4 * rng.normal(size=n_train))
        values = np.clip(values, lo, hi)
        values[rng.rand(n_train) < miss] = np.nan
        labels[target] = values.round(4)
    labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)  # duplicate label row

    test_ids = ids[n_train:n_train + n_test]
    submission = pd.DataFrame({config.ID_COLUMN: test_ids})
    for target in config.TARGET_COLUMNS:
        submission[target] = 0.0

    # 5 submission clusters with no GEE row (mirrors the 25 missing in the real data),
    # and one duplicated GEE DHSID (mirrors the conflicting Bolivia row).
    gee = gee[~gee[config.ID_COLUMN].isin(test_ids[:5])]
    gee = pd.concat([gee, gee.iloc[[1]]], ignore_index=True).sample(frac=1, random_state=seed)
    return gee, labels, submission


def write_competition_files(out_dir=config.DATA_SYNTHETIC_DIR, **kwargs) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gee, labels, submission = generate(**kwargs)
    gee.to_csv(out_dir / config.GEE_FULL_FILE, index=False)
    gee.sample(frac=0.10, random_state=config.RANDOM_STATE).to_csv(
        out_dir / config.GEE_SAMPLE_FILE, index=False)
    labels.to_csv(out_dir / config.LABELS_FILE, index=False)
    submission.to_csv(out_dir / config.SUBMISSION_FILE, index=False)
    return out_dir


if __name__ == "__main__":
    print(f"Synthetic competition files written to {write_competition_files()}")
