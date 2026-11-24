"""Per-target RMSE, mean RMSE across targets, and a mean-prediction baseline."""
import json

import numpy as np
import pandas as pd

from . import config


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_all(fitted_models: dict, X_val: pd.DataFrame, y_val: pd.DataFrame,
                 y_train: pd.DataFrame | None = None) -> dict:
    """RMSE per target on rows where the target is labelled.

    If ``y_train`` is given, also report the RMSE of always predicting the
    training mean. A model is only useful if it beats this baseline.
    """
    per_target, baseline = {}, {}
    for target, artifact in fitted_models.items():
        valid = y_val[target].notna()
        preds = artifact["model"].predict(X_val.loc[valid])
        per_target[target] = rmse(y_val.loc[valid, target], preds)
        if y_train is not None:
            baseline[target] = rmse(y_val.loc[valid, target],
                                    np.full(valid.sum(), y_train[target].mean()))
    out = {"per_target_rmse": per_target, "mean_rmse": float(np.mean(list(per_target.values())))}
    if baseline:
        out["baseline_per_target_rmse"] = baseline
        out["baseline_mean_rmse"] = float(np.mean(list(baseline.values())))
    return out


def save_metrics(metrics: dict, path=None) -> None:
    path = path or (config.REPORTS_DIR / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
