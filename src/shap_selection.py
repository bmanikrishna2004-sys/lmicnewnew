"""Stage 2 of feature selection: neural network + Kernel SHAP ranking.

For each target we fit a small MLP, then use Kernel SHAP to measure how much
each input feature moves the MLP's prediction. Importances are normalised per
target (so no single target dominates), averaged across the six targets, and
the top ``N_SHAP_FEATURES`` are kept.

Why a NN and not XGBoost for the ranking? The ranking model is deliberately a
different model family from the final model, which gives a ranking that does
not simply reward features the final trees already split on.
"""
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import config


def rank_features_for_target(X, y, background_size, explain_size, extra_nsamples):
    """Mean |SHAP| per feature for one target. Rows with a missing label are skipped."""
    valid = y.notna()
    Xv, yv = X.loc[valid], y.loc[valid].astype(float)
    if len(Xv) < 20:
        return pd.Series(0.0, index=X.columns)

    rng = np.random.RandomState(config.RANDOM_STATE)
    if len(Xv) > config.SHAP_MAX_ROWS:
        idx = rng.choice(len(Xv), config.SHAP_MAX_ROWS, replace=False)
        Xv, yv = Xv.iloc[idx], yv.iloc[idx]

    # NNs need complete, scaled inputs (XGBoost later does not).
    Xs = StandardScaler().fit_transform(
        SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(Xv)
    ).astype("float32")
    nn = MLPRegressor(hidden_layer_sizes=config.NN_HIDDEN_LAYERS,
                      max_iter=config.NN_MAX_ITER, early_stopping=True,
                      random_state=config.RANDOM_STATE)
    nn.fit(Xs, yv.to_numpy())

    background = Xs[rng.choice(len(Xs), min(background_size, len(Xs)), replace=False)]
    explain = Xs[rng.choice(len(Xs), min(explain_size, len(Xs)), replace=False)]
    explainer = shap.KernelExplainer(nn.predict, background)
    nsamples = 2 * X.shape[1] + extra_nsamples   # never fewer samples than features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # LARS convergence chatter inside shap
        sv = explainer.shap_values(explain, nsamples=nsamples, silent=True)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)

    # Sanity check: SHAP values must add up to prediction - expected value.
    additivity_err = np.abs(sv.sum(axis=1) + explainer.expected_value - nn.predict(explain)).max()
    if not np.isfinite(sv).all() or additivity_err > 1e-3 * (1.0 + float(yv.std())):
        print(f"      WARNING: SHAP additivity error {additivity_err:.3g}; raise the SHAP budget")
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns)


def rank_all_targets(X, targets, target_columns, fast=False):
    """DataFrame of per-target SHAP importances (rows = features)."""
    if fast:
        bg, exp, ns = config.SHAP_BACKGROUND_FAST, config.SHAP_EXPLAIN_FAST, config.SHAP_EXTRA_NSAMPLES_FAST
    else:
        bg, exp, ns = config.SHAP_BACKGROUND_FULL, config.SHAP_EXPLAIN_FULL, config.SHAP_EXTRA_NSAMPLES_FULL
    ranks = {}
    for target in target_columns:
        print(f"      SHAP ranking: {target}")
        ranks[target] = rank_features_for_target(X, targets[target], bg, exp, ns)
    return pd.concat(ranks, axis=1).fillna(0.0)


def combine_rankings(per_target: pd.DataFrame) -> pd.Series:
    """Normalise each target's importances to [0, 1] and average across targets."""
    normalized = per_target.apply(lambda c: c / c.max() if c.max() > 0 else c)
    return normalized.mean(axis=1).sort_values(ascending=False)


def select_top_features(X, targets, target_columns, fast=False, n_features=None,
                        return_rankings=False):
    per_target = rank_all_targets(X, targets, target_columns, fast=fast)
    combined = combine_rankings(per_target)
    top = combined.head(n_features or config.N_SHAP_FEATURES).index.tolist()
    if return_rankings:
        return top, per_target.assign(combined=combined).sort_values("combined", ascending=False)
    return top
