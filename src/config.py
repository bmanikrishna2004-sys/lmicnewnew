"""Central configuration for the LMIC maternal & child health pipeline.

Every tunable number in the project lives here, so experiments are changed in
one place and the rest of the code stays the same.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
REPORTS_DIR = PROJECT_ROOT / "reports"

# File names exactly as shipped by the competition.
LABELS_FILE = "training_label.csv"
GEE_FULL_FILE = "gee_features.csv"            # ~8.15 GB, one row per DHS cluster
GEE_SAMPLE_FILE = "gee_features_10pct.csv"    # 10% row sample, same columns
SUBMISSION_FILE = "sample submission.csv"     # note the space in the name

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
ID_COLUMN = "DHSID"          # e.g. "KE201400001234": country + survey year + cluster
LAT_COLUMN = "LATNUM"
LON_COLUMN = "LONGNUM"
YEAR_COLUMN = "DHSYEAR"
CLUSTER_COLUMN = "DHSCLUST"

TARGET_COLUMNS = [
    "Mean_BMI",
    "Median_BMI",
    "Unmet_Need_Rate",
    "Under5_Mortality_Rate",
    "Skilled_Birth_Attendant_Rate",
    "Stunted_Rate",
]

# Identifiers / text metadata are never passed to a model: they either leak
# the row identity or are not meaningful as numbers.
DROP_FEATURE_COLUMNS = {
    "DHSID", "new_ind", "index", "DHSYEAR", "DHSCLUST",
    "key1", "key2", "key3", "ADM1DHS", "ADM1FIPS", "ADM1FIPSNA",
    "ADM1NAME", "ADM1SALBCO", "ADM1SALBNA", "ADM1SALNA",
}

# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
N_SCREEN_FEATURES = 250   # stage 1: quality screen -> 250-feature superset
N_SHAP_FEATURES = 100     # stage 2: NN + SHAP ranking -> top 100
MAX_MISSING_RATE = 0.98   # drop a feature if more than 98% of rows are missing

RANDOM_STATE = 42
NN_HIDDEN_LAYERS = (64, 32)
NN_MAX_ITER = 150
SHAP_MAX_ROWS = 5000      # NN/SHAP is fit on at most this many rows per target

# Kernel SHAP budget. Kernel SHAP is model-agnostic but expensive: cost grows
# with background rows x explained rows x coalition samples.
# Coalition samples are set to 2 * n_features + EXTRA (the same rule SHAP uses
# by default). With fewer samples than features the internal least-squares
# solve is under-determined and can return values around 1e15 that cancel
# each other out, which wrecks the ranking.
SHAP_BACKGROUND_FAST = 10
SHAP_EXPLAIN_FAST = 12
SHAP_EXTRA_NSAMPLES_FAST = 50
SHAP_BACKGROUND_FULL = 50
SHAP_EXPLAIN_FULL = 200
SHAP_EXTRA_NSAMPLES_FULL = 2048

# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------
XGB_FIXED_PARAMS = {
    "objective": "reg:pseudohubererror",   # Huber-style loss: robust to outlier clusters
    "n_estimators": 600,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,    # one thread per model; parallelism comes from the CV search
}

XGB_SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "huber_slope": [0.5, 1.0, 1.5, 2.0],
    "min_child_weight": [1, 3, 5],
}

N_SEARCH_ITER = 25
CV_FOLDS = 5
SEARCH_N_JOBS = -1   # run CV fits in parallel on all cores

# Set by --gpu: XGBoost trains on an NVIDIA GPU, one fit at a time.
USE_GPU = False

# --fast mode: same code path, smaller budget. Use it for the first run.
FAST_XGB_FIXED_PARAMS = {**XGB_FIXED_PARAMS, "n_estimators": 150}
FAST_N_SEARCH_ITER = 3
FAST_CV_FOLDS = 3

VALIDATION_FRACTION = 0.20
