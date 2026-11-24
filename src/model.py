"""Per-target XGBoost regressors with a pseudo-Huber loss.

One model per target, not one multi-output model: each DHS indicator has its
own scale, noise level and missing-label pattern, and benefits from its own
tuned hyper-parameters (especially ``huber_slope``).

The base_score fix
------------------
With ``objective="reg:pseudohubererror"`` the boosting starts from
``base_score``. The pseudo-Huber gradient is capped at about ``huber_slope``
and its hessian shrinks towards zero for large residuals. If the starting
point is far from the targets (e.g. 0.5 when BMI is ~23), every residual is
"large": gradients saturate, hessians vanish, Newton steps (-G/H) blow up and
training can diverge to a constant, wildly-off prediction. Starting at the
target mean puts residuals in the well-behaved region of the loss from the
first tree. See ``notebooks/`` for a side-by-side demonstration.
"""
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor

from . import config


def make_model(y: pd.Series, fast: bool = False, **overrides) -> XGBRegressor:
    params = config.FAST_XGB_FIXED_PARAMS if fast else config.XGB_FIXED_PARAMS
    if config.USE_GPU:
        params = {**params, "device": "cuda"}
    # The single most important line in this file (see module docstring).
    return XGBRegressor(**{**params, "base_score": float(y.mean()), **overrides})


def tune_and_fit(X: pd.DataFrame, y: pd.Series, fast: bool = False):
    """Randomised search with K-fold CV; returns (best refitted model, best params)."""
    n_iter = config.FAST_N_SEARCH_ITER if fast else config.N_SEARCH_ITER
    cv_folds = config.FAST_CV_FOLDS if fast else config.CV_FOLDS
    search = RandomizedSearchCV(
        estimator=make_model(y, fast=fast),
        param_distributions=config.XGB_SEARCH_SPACE,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE,
        n_jobs=1 if config.USE_GPU else config.SEARCH_N_JOBS,  # parallel across CV fits, not inside each tree
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def fit_all_targets(X: pd.DataFrame, targets: pd.DataFrame, target_columns, fast=False,
                    fixed_params: dict | None = None) -> dict:
    """Fit one model per target on rows where that target is labelled.

    If ``fixed_params`` ({target: params}) is given, skip tuning and reuse those
    hyper-parameters, e.g. for the final refit on all labelled rows.
    Returns ``{target: {"model", "best_params", "n_rows"}}``.
    """
    fitted = {}
    for target in target_columns:
        valid = targets[target].notna()
        Xt, yt = X.loc[valid], targets.loc[valid, target]
        print(f"      {target:30s} rows={len(yt):,}")
        if fixed_params and target in fixed_params:
            params = fixed_params[target]
            model = make_model(yt, fast=fast, **params).fit(Xt, yt)
        else:
            model, params = tune_and_fit(Xt, yt, fast=fast)
        fitted[target] = {"model": model, "best_params": params, "n_rows": int(len(yt))}
    return fitted
