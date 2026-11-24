# Running on Kaggle

1. Create a new notebook inside the competition (so the data is attached under `/kaggle/input/`).
2. Settings → **Internet: On** (needed for `git clone`; or upload this repo as a Kaggle Dataset instead).
3. Run:

```bash
!git clone https://github.com/<your-username>/lmic-health-pipeline.git
%cd lmic-health-pipeline
!pip install -q -r requirements.txt
!ls /kaggle/input/                       # find the competition folder name
```

4. First, a fast run, which checks the whole real-data path with a small budget:

```bash
!python -m src.pipeline --data-dir /kaggle/input/<competition-folder> --fast
```

5. Then the full run:

```bash
!python -m src.pipeline --data-dir /kaggle/input/<competition-folder>
```

6. Outputs are in `reports/`: `metrics.json`, `submission.csv`, `shap_rankings.csv`, `models/`.
   Copy `reports/submission.csv` to `/kaggle/working/` if you want to submit it from the notebook output.

## Runtime
For reference, measured on 4 CPU cores with **1,500 synthetic clusters**:
* `--fast`: about 1 minute.
* Full budget: about 30 minutes, mostly the 25-iteration × 5-fold search for each of the 6 targets.

Tuning time grows roughly linearly with the number of labelled rows, so the real data will take longer. Always do the `--fast` run first. If the full run would exceed Kaggle's session limit, lower `N_SEARCH_ITER` (e.g. 10) or `n_estimators` in `src/config.py`.

## Tips
* Streaming the full 8.15 GB GEE file takes the longest. It is read once per run.
* If memory is tight, lower `chunksize` in `real_data.load_matched_feature_matrix`.
* XGBoost runs single-threaded per model while the CV search runs fits in parallel (`SEARCH_N_JOBS = -1`). On small per-target tables this is much faster than multi-threading inside each model.
* Note which split (`random` / `country`) and mode (`--fast` or full) produced each RMSE you report.
