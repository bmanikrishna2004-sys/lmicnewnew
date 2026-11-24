# Predicting Maternal & Child Health in LMICs from Geospatial Data

A geospatial machine-learning pipeline that predicts **six maternal and child health indicators** for every **DHS survey cluster** in low- and middle-income countries (LMICs), using **Google Earth Engine (GEE)** satellite-derived features. The goal is to fill gaps in **localized health monitoring**: household surveys run only every few years in sampled places, while satellite data covers everywhere.

**Highlights**
- Streams an **~8 GB GEE feature table** in chunks and joins it to **DHS** cluster labels on `DHSID`, without loading the file into memory
- **Two-stage feature selection:** a label-free quality screen builds a **250-feature superset**, then **neural-network + Kernel SHAP** rankings keep the **top 100** environmental and population features
- **One tuned XGBoost regressor per target** with **pseudo-Huber loss**, including a fix for **`base_score` divergence**
- Reports **per-target and mean RMSE across all 6 targets**, always next to a predict-the-mean baseline
- Runs end-to-end offline on **synthetic data with the competition's exact file layout**, with a pytest suite and a step-by-step learning notebook

> **New to the project?** Start with [`notebooks/LMIC_Pipeline_Learning_Guide.ipynb`](notebooks/LMIC_Pipeline_Learning_Guide.ipynb). It explains every concept and step, runs the real code, and documents every problem we hit and how we fixed it.

---

## Targets

| Target | Meaning (DHS definition) |
|---|---|
| `Mean_BMI` | Mean body-mass index of surveyed women in the cluster |
| `Median_BMI` | Median BMI of the same women |
| `Unmet_Need_Rate` | Women who want to stop or delay childbearing but use no contraception |
| `Under5_Mortality_Rate` | Probability of dying before age 5 |
| `Skilled_Birth_Attendant_Rate` | Recent births attended by a doctor, nurse or midwife |
| `Stunted_Rate` | Under-5 children with height-for-age below −2 SD (WHO standard) |

## Pipeline

```
gee_features_10pct.csv ─► [1] label-free quality screen (missingness, variance) ─► 250 features
gee_features.csv (~8 GB) ─► [2] stream in chunks, keep 250 cols + needed DHSIDs, join labels on DHSID
                            [3] NN (MLP 64-32) + Kernel SHAP per target, train split only ─► top 100
                            [4] XGBoost per target: pseudo-Huber loss, randomised search, 5-fold CV,
                                base_score anchored to the target mean
                            [5] validation RMSE per target + mean RMSE, vs. mean baseline
                            [6] refit on all labelled rows ─► submission.csv
```

| Module | Role |
|---|---|
| `src/config.py` | every tunable number: paths, 250/100 sizes, search space, SHAP budget |
| `src/real_data.py` | chunked loading, feature screening, `DHSID` joins |
| `src/shap_selection.py` | NN + Kernel SHAP ranking, per-target normalisation, top-100 selection |
| `src/model.py` | per-target XGBoost, tuning, `base_score` fix |
| `src/evaluate.py` | RMSE per target, mean RMSE, baseline |
| `src/pipeline.py` | end-to-end CLI |
| `src/targets.py` | target definitions and label summary |
| `src/geo_features.py` | optional OpenStreetMap context features (counts within 2/5/10 km, nearest distance) |
| `src/synthetic_data.py` | synthetic data in the competition's file format, for offline runs and tests |

## Quick start

```bash
git clone https://github.com/<your-username>/lmic-health-pipeline.git
cd lmic-health-pipeline
pip install -r requirements.txt

python -m pytest -q                              # tests (synthetic data)
python -m src.pipeline --synthetic --fast        # full pipeline on synthetic data, ~1 min
```

**On the real competition data** (see [`docs/KAGGLE_RUN.md`](docs/KAGGLE_RUN.md)):

```bash
# first run: small budget, checks the whole real-data path end to end
python -m src.pipeline --data-dir /kaggle/input/<competition-folder> --fast
# full run
python -m src.pipeline --data-dir /kaggle/input/<competition-folder>
# optional: hold out whole countries for an "unseen country" estimate
python -m src.pipeline --data-dir /kaggle/input/<competition-folder> --split country
```

Expected files in `--data-dir`: `training_label.csv`, `gee_features.csv`, `gee_features_10pct.csv`, `sample submission.csv`.

Outputs in `reports/`: `metrics.json` (per-target RMSE, mean RMSE, baseline, best params, selected features), `submission.csv`, `shap_rankings.csv`, `models/xgb_<target>.json`.

## Results

Validation set: random 20% of labelled clusters, full (non-`--fast`) budget.

| Target | RMSE | Baseline RMSE |
|---|---|---|
| Mean_BMI | _fill in_ | _fill in_ |
| Median_BMI | _fill in_ | _fill in_ |
| Unmet_Need_Rate | _fill in_ | _fill in_ |
| Under5_Mortality_Rate | _fill in_ | _fill in_ |
| Skilled_Birth_Attendant_Rate | _fill in_ | _fill in_ |
| Stunted_Rate | _fill in_ | _fill in_ |
| **Mean across 6 targets** | **_fill in_** | _fill in_ |

_Copy these from `reports/metrics.json` after the Kaggle run (`per_target_rmse`, `mean_rmse`, `baseline_*`)._

## Key engineering decisions

- **`base_score` fix.** With `reg:pseudohubererror`, the pseudo-Huber hessian vanishes for large residuals. If boosting starts far from the targets, Newton steps blow up and predictions diverge. In a test on BMI-scale data, XGBoost's default start gave a validation RMSE of ~23 and the anchored model ~1.1. Setting `base_score = mean(y)` per target fixes it (`src/model.py`).
- **Pseudo-Huber loss.** Cluster rates come from ~20–30 households, so some labels are very noisy. Huber limits their pull, and `huber_slope` is tuned per target.
- **One model per target.** Targets differ in scale, noise and *which rows are labelled*. Each model trains on its own labelled rows instead of dropping rows that miss any target.
- **No leakage in selection.** The screen ignores labels, and SHAP ranking uses only the training split.
- **Kernel SHAP budget.** Coalition samples are `2 × n_features + extra`. Fewer samples than features makes the solve under-determined and produced occasional ~1e15 importances. An additivity check guards against this.
- **Data quirks.** Duplicate `DHSID`s keep the first row, and joins use `validate="one_to_one"`. Submission clusters with no GEE row (25 in the real data) get the training mean as an explicit, reported fallback.

## Limitations and next steps

- A random split measures interpolation among nearby clusters. `--split country` gives a stricter estimate for unseen countries.
- DHS GPS points are displaced by up to 2–10 km, so fine-scale features are noisy.
- The variance-based screen favours features with large raw units. A unit-free score is a possible improvement.
- Possible next steps: spatial CV, LightGBM/CatBoost ensembles, logit-transformed rate targets, and prediction intervals.

## Data

The competition data (DHS-derived labels and GEE features) is **not included** in this repository. It is subject to the competition's terms and DHS Program data-use conditions. `data/` holds only placeholders, and `python -m src.synthetic_data` writes stand-in files to `data/synthetic/`.
