"""Fast tests on synthetic data. Run with:  python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src import config
from src.evaluate import rmse
from src.geo_features import build_osm_features, haversine_km
from src.model import tune_and_fit
from src.pipeline import run
from src.real_data import build_train_test, screen_features
from src.shap_selection import select_top_features
from src.synthetic_data import write_competition_files


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return write_competition_files(tmp_path_factory.mktemp("data"),
                                   n_train=300, n_test=60, n_extra=40, n_features=120)


def test_screen_drops_empty_and_constant_columns(data_dir):
    kept = screen_features(data_dir, n_features=1000)
    names = pd.read_csv(data_dir / config.GEE_SAMPLE_FILE, nrows=1).columns
    feature_names = [c for c in names if c.startswith(("viirs", "ndvi", "lst", "chirps", "worldpop",
                                                       "srtm", "ghsl", "dist", "evi", "pdsi",
                                                       "surface", "landcover", "travel"))]
    assert not set(feature_names[-10:]) & set(kept)          # 5 empty + 5 constant
    assert not set(kept) & config.DROP_FEATURE_COLUMNS
    assert not set(kept) & set(config.TARGET_COLUMNS)


def test_join_handles_duplicates_and_missing_rows(data_dir):
    feats = screen_features(data_dir, n_features=30)
    train, test, labels, submission, n_missing = build_train_test(data_dir, feats)
    assert train[config.ID_COLUMN].is_unique
    assert len(test) == len(submission) == 60
    assert n_missing == 5
    assert (test[config.ID_COLUMN].values == submission[config.ID_COLUMN].values).all()


def test_shap_selection_returns_requested_count(data_dir):
    feats = screen_features(data_dir, n_features=40)
    train, *_ = build_train_test(data_dir, feats)
    top = select_top_features(train[feats], train[config.TARGET_COLUMNS],
                              config.TARGET_COLUMNS, fast=True, n_features=10)
    assert len(top) == 10 and set(top) <= set(feats)


def test_base_score_anchor_beats_default_start():
    """Pseudo-Huber needs base_score near the target mean (the 'base_score fix')."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 5))
    y = 23 + 2 * X[:, 0] + rng.normal(scale=0.5, size=400)
    common = dict(objective="reg:pseudohubererror", n_estimators=100, huber_slope=1.0,
                  learning_rate=0.1, max_depth=3)
    bad = XGBRegressor(**common, base_score=0.5).fit(X[:300], y[:300])
    good = XGBRegressor(**common, base_score=float(y[:300].mean())).fit(X[:300], y[:300])
    assert rmse(y[300:], good.predict(X[300:])) < rmse(y[300:], bad.predict(X[300:]))


def test_tune_and_fit_predicts():
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.normal(size=(200, 8)))
    y = pd.Series(50 + 10 * X[0] + rng.normal(size=200))
    m, params = tune_and_fit(X.iloc[:150], y.iloc[:150], fast=True)
    assert set(params) <= set(config.XGB_SEARCH_SPACE)
    assert rmse(y.iloc[150:], m.predict(X.iloc[150:])) < y.std()


def test_osm_features():
    clusters = pd.DataFrame({config.ID_COLUMN: ["A"], config.LAT_COLUMN: [0.0],
                             config.LON_COLUMN: [0.0]})
    osm = pd.DataFrame({"lat": [0.0, 0.03, 0.3], "lon": [0.01, 0.0, 0.0],
                        "category": ["school", "school", "market"]})
    out = build_osm_features(clusters, osm)
    assert out.loc[0, "osm_school_count_5km"] == 2
    assert out.loc[0, "osm_market_count_10km"] == 0
    assert out.loc[0, "osm_market_nearest_km"] == pytest.approx(haversine_km(0, 0, 0.3, 0), rel=1e-6)


def test_end_to_end(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "N_SHAP_FEATURES", 15)
    metrics = run(data_dir, fast=True, n_screen=40, out_dir=tmp_path)
    sub = pd.read_csv(tmp_path / "submission.csv")
    assert list(sub.columns) == [config.ID_COLUMN] + config.TARGET_COLUMNS
    assert len(sub) == 60 and sub[config.TARGET_COLUMNS].notna().all().all()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert set(saved["per_target_rmse"]) == set(config.TARGET_COLUMNS)
    assert metrics["mean_rmse"] < metrics["baseline_mean_rmse"]
