"""Memory-conscious loaders for the competition files.

The full GEE feature file is ~8.15 GB, which does not fit comfortably in a
Kaggle notebook's RAM as a pandas DataFrame. The strategy is:

1. Decide *which columns* to keep using the small 10% sample file.
2. Stream the full file in chunks, reading only those columns and only the
   rows whose DHSID appears in the labels or the submission.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_labels(data_dir):
    """Load training labels, one row per DHSID (first duplicate wins)."""
    labels = pd.read_csv(Path(data_dir) / config.LABELS_FILE)
    labels[config.ID_COLUMN] = labels[config.ID_COLUMN].astype(str)
    return labels.drop_duplicates(config.ID_COLUMN, keep="first").reset_index(drop=True)


def load_submission_ids(data_dir):
    """Load the DHSIDs that must be predicted, in submission order."""
    sub = pd.read_csv(Path(data_dir) / config.SUBMISSION_FILE, usecols=[config.ID_COLUMN])
    sub[config.ID_COLUMN] = sub[config.ID_COLUMN].astype(str)
    if sub[config.ID_COLUMN].duplicated().any():
        raise ValueError("sample submission contains duplicate DHSIDs")
    return sub


def candidate_columns(data_dir, sample_file=config.GEE_SAMPLE_FILE):
    """Numeric, non-identifier, non-target columns of the GEE file."""
    sample = pd.read_csv(Path(data_dir) / sample_file, nrows=1000)
    excluded = config.DROP_FEATURE_COLUMNS | set(config.TARGET_COLUMNS)
    return [c for c in sample.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(sample[c])]


def score_features(data_dir, candidates, sample_file=config.GEE_SAMPLE_FILE):
    """Target-independent quality statistics for each candidate feature.

    Streams the 10% file and accumulates count / sum / sum-of-squares so the
    whole file never has to be in memory. Features that are almost always
    missing or constant are removed; the rest are ranked by
    ``log1p(variance) * (1 - missing_rate)``.

    The labels are not used here, so this step cannot leak target information.
    """
    path = Path(data_dir) / sample_file
    sums = np.zeros(len(candidates))
    sums2 = np.zeros(len(candidates))
    counts = np.zeros(len(candidates), dtype=np.int64)
    n_rows = 0
    for chunk in pd.read_csv(path, usecols=candidates, chunksize=1000):
        arr = chunk[candidates].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
        ok = np.isfinite(arr)
        safe = np.where(ok, arr, 0.0)
        counts += ok.sum(axis=0)
        sums += safe.sum(axis=0)
        sums2 += (safe * safe).sum(axis=0)
        n_rows += len(arr)

    mean = sums / np.maximum(counts, 1)
    var = np.maximum(sums2 / np.maximum(counts, 1) - mean * mean, 0.0)
    miss_rate = 1.0 - counts / max(n_rows, 1)
    score = pd.DataFrame({"missing_rate": miss_rate, "variance": var}, index=candidates)
    score = score[(score["missing_rate"] <= config.MAX_MISSING_RATE) & (score["variance"] > 0)]
    score["rank_score"] = np.log1p(score["variance"]) * (1.0 - score["missing_rate"])
    return score.sort_values("rank_score", ascending=False)


def screen_features(data_dir, n_features=None):
    """Stage 1 of feature selection: return the top-N screened feature names."""
    ranked = score_features(data_dir, candidate_columns(data_dir))
    return ranked.index[: (n_features or config.N_SCREEN_FEATURES)].tolist()


def load_matched_feature_matrix(data_dir, feature_columns, wanted_ids, chunksize=10000):
    """Read the full GEE file once, keeping only wanted DHSIDs and columns."""
    path = Path(data_dir) / config.GEE_FULL_FILE
    wanted = set(wanted_ids)
    usecols = [config.ID_COLUMN] + list(feature_columns)
    pieces = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk[config.ID_COLUMN] = chunk[config.ID_COLUMN].astype(str)
        hit = chunk[chunk[config.ID_COLUMN].isin(wanted)]
        if not hit.empty:
            pieces.append(hit)
    if not pieces:
        raise ValueError(f"No requested DHSIDs were found in {config.GEE_FULL_FILE}")
    gee = pd.concat(pieces, ignore_index=True)
    # A DHSID can appear more than once in the GEE file (one known conflicting
    # Bolivia row in the real data). Keep the first row so the result is
    # deterministic and every DHSID maps to exactly one feature vector.
    return gee.drop_duplicates(config.ID_COLUMN, keep="first")


def build_train_test(data_dir, feature_columns):
    """Join features to labels (train) and to submission IDs (test).

    Returns ``train, test, labels, submission, n_test_missing_gee``.
    """
    labels = load_labels(data_dir)
    submission = load_submission_ids(data_dir)
    wanted = set(labels[config.ID_COLUMN]) | set(submission[config.ID_COLUMN])
    gee = load_matched_feature_matrix(data_dir, feature_columns, wanted)

    train = gee.merge(labels, on=config.ID_COLUMN, how="inner", validate="one_to_one")
    # Left join keeps every submission row, even ones with no GEE features.
    test = submission.merge(gee, on=config.ID_COLUMN, how="left", validate="one_to_one")

    for c in feature_columns:
        # float32 halves the memory of float64 with no practical accuracy loss.
        train[c] = pd.to_numeric(train[c], errors="coerce").astype("float32")
        test[c] = pd.to_numeric(test[c], errors="coerce").astype("float32")
    n_missing = int(test[feature_columns].isna().all(axis=1).sum())
    return train, test, labels, submission, n_missing


def country_code(dhsid: pd.Series) -> pd.Series:
    """First two characters of a DHSID are the DHS country code (e.g. 'KE')."""
    return dhsid.astype(str).str[:2]
