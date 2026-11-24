"""End-to-end pipeline: screen -> join -> NN+SHAP select -> XGBoost -> evaluate -> submit.

Usage
-----
    # offline smoke test on generated data with the competition's file layout
    python -m src.pipeline --synthetic --fast

    # Kaggle, first run (small budget, validates the whole real-data path)
    python -m src.pipeline --data-dir /kaggle/input/<competition-folder> --fast

    # Kaggle, full tuning budget
    python -m src.pipeline --data-dir /kaggle/input/<competition-folder>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config, evaluate, model
from .real_data import build_train_test, country_code, screen_features
from .shap_selection import select_top_features
from .targets import describe_targets


def split_indices(train: pd.DataFrame, how: str = "random"):
    """Train/validation split.

    ``random``  - rows assigned at random (the default; matches the original runs).
    ``country`` - whole countries held out. Nearby clusters look alike, so a random
                  split can be optimistic; this shows how well the model transfers
                  to unseen countries.
    """
    idx = np.arange(len(train))
    if how == "random":
        return train_test_split(idx, test_size=config.VALIDATION_FRACTION,
                                random_state=config.RANDOM_STATE)
    if how == "country":
        groups = country_code(train[config.ID_COLUMN])
        gss = GroupShuffleSplit(n_splits=1, test_size=config.VALIDATION_FRACTION,
                                random_state=config.RANDOM_STATE)
        return next(gss.split(idx, groups=groups))
    raise ValueError(f"unknown split: {how}")


def run(data_dir, fast=False, split="random", n_screen=None, out_dir=None):
    t0 = time.time()
    data_dir = Path(data_dir)
    out_dir = Path(out_dir or config.REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_screen = n_screen or config.N_SCREEN_FEATURES

    print(f"[1/6] Screening GEE features on the 10% file (keep {n_screen})...")
    screened = screen_features(data_dir, n_screen)
    print(f"      retained {len(screened)} candidate features")

    print("[2/6] Streaming the full GEE file and joining on DHSID...")
    train, test, labels, submission, n_missing = build_train_test(data_dir, screened)
    print(f"      labelled clusters with features: {len(train):,}")
    print(f"      submission rows: {len(test):,}  (no GEE row: {n_missing})")
    print(describe_targets(train).to_string())

    X, y = train[screened], train[config.TARGET_COLUMNS]
    tr_idx, va_idx = split_indices(train, split)
    Xtr, Xva, ytr, yva = X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]
    print(f"      split={split}: train={len(tr_idx):,} validation={len(va_idx):,}")

    print(f"[3/6] NN + Kernel SHAP ranking on {len(screened)} features (train split only)...")
    top, rankings = select_top_features(Xtr, ytr, config.TARGET_COLUMNS, fast=fast,
                                        return_rankings=True)
    rankings.to_csv(out_dir / "shap_rankings.csv", index_label="feature")
    print(f"      selected {len(top)} features")

    print("[4/6] Tuning one XGBoost (pseudo-Huber) model per target...")
    fitted = model.fit_all_targets(Xtr[top], ytr, config.TARGET_COLUMNS, fast=fast)

    print("[5/6] Validation RMSE (model vs. predict-the-mean baseline):")
    metrics = evaluate.evaluate_all(fitted, Xva[top], yva, y_train=ytr)
    for t in config.TARGET_COLUMNS:
        print(f"      {t:30s} {metrics['per_target_rmse'][t]:8.4f}"
              f"   baseline {metrics['baseline_per_target_rmse'][t]:8.4f}")
    print(f"      {'MEAN':30s} {metrics['mean_rmse']:8.4f}"
          f"   baseline {metrics['baseline_mean_rmse']:8.4f}")

    print("[6/6] Refitting on all labelled rows with the tuned params; predicting submission...")
    best = {t: fitted[t]["best_params"] for t in config.TARGET_COLUMNS}
    final = model.fit_all_targets(X[top], y, config.TARGET_COLUMNS, fast=fast, fixed_params=best)
    pred = submission.copy()
    no_features = test[top].isna().all(axis=1).to_numpy()
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for t in config.TARGET_COLUMNS:
        pred[t] = final[t]["model"].predict(test[top])
        # Rows with no GEE features get the training mean: an explicit fallback,
        # not a feature-based prediction.
        pred.loc[no_features, t] = float(y[t].mean())
        final[t]["model"].save_model(models_dir / f"xgb_{t}.json")
    pred.to_csv(out_dir / "submission.csv", index=False)

    metrics.update({
        "split": split,
        "fast": fast,
        "n_screened_features": len(screened),
        "n_selected_features": len(top),
        "selected_features": top,
        "best_params": best,
        "n_train_rows": int(len(train)),
        "n_submission_rows": int(len(test)),
        "n_submission_rows_without_gee": int(n_missing),
        "runtime_minutes": round((time.time() - t0) / 60, 2),
    })
    evaluate.save_metrics(metrics, out_dir / "metrics.json")
    print(f"\nDone in {metrics['runtime_minutes']} min. Outputs in {out_dir}/")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="folder with the competition CSV files")
    src.add_argument("--synthetic", action="store_true",
                     help="generate synthetic data in data/synthetic and run on it")
    p.add_argument("--fast", action="store_true", help="small search/SHAP budget")
    p.add_argument("--split", choices=["random", "country"], default="random")
    p.add_argument("--n-screen", type=int, default=None,
                   help=f"size of the screened superset (default {config.N_SCREEN_FEATURES})")
    p.add_argument("--gpu", action="store_true",
                   help="train XGBoost on an NVIDIA GPU (device='cuda')")
    p.add_argument("--out-dir", default=None, help="default: reports/")
    args = p.parse_args()

    data_dir = args.data_dir
    if args.gpu:
        config.USE_GPU = True
        print("XGBoost will train on the GPU (device='cuda')")
    if args.synthetic:
        from .synthetic_data import write_competition_files
        data_dir = write_competition_files(config.DATA_SYNTHETIC_DIR)
        print(f"Synthetic competition files written to {data_dir}")
    metrics = run(data_dir, fast=args.fast, split=args.split,
                  n_screen=args.n_screen, out_dir=args.out_dir)
    print(json.dumps({"mean_rmse": metrics["mean_rmse"],
                      "baseline_mean_rmse": metrics["baseline_mean_rmse"]}, indent=2))


if __name__ == "__main__":
    main()
