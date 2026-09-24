"""Nested, family-stratified cross-validation of modelling recipes.

A recipe is a feature set + model type + tuning procedure. The outer loop
(repeated, stratified by chemical family) estimates how well a recipe
generalises; the inner loop tunes hyperparameters using only the outer
training fold, so no outer-test compound ever influences tuning. The fold
models are measuring instruments and are discarded; fit_final_model then
runs the winning recipe's tuning on all the data.

Models: Ridge, XGBoost and an MLP, each with its own preprocessing
pipeline, plus their average (the ensemble). Ridge and the MLP standardise
X and y; all models drop constant features; Ridge and the MLP also drop
near-duplicate features (DropCorrelated), which trees don't need.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .descriptors import DropCorrelated
from .models import MLP_PARAM_GRID, RIDGE_PARAM_GRID, XGB_PARAM_GRID

BASE_MODELS = ("ridge", "xgboost", "mlp")
ENSEMBLE = "ensemble"


def stratification_labels(families: pd.Series, n_splits: int) -> np.ndarray:
    """Family labels for splitting. Families too small to appear in every
    fold are grouped with the most common family *for splitting only*;
    errors are still reported under their own family."""
    families = pd.Series(families).reset_index(drop=True)
    counts = families.value_counts()
    too_small = counts[counts < n_splits].index
    return families.where(~families.isin(too_small), counts.index[0]).to_numpy()


def _scaled(model):
    """Standardise X and y around a Ridge/MLP, after dropping constant and
    near-duplicate features."""
    pipe = Pipeline([("constant", VarianceThreshold()), ("correlated", DropCorrelated()),
                     ("scale", StandardScaler()), ("model", model)])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def make_search(model: str, inner_cv, n_iter: int, random_state: int = 0) -> RandomizedSearchCV:
    """Randomized search over the original project's grids (models.py)."""
    if model == "ridge":
        estimator, grid, prefix = _scaled(Ridge()), RIDGE_PARAM_GRID, "regressor__model__"
    elif model == "mlp":
        estimator = _scaled(MLPRegressor(max_iter=1000, tol=1e-4, random_state=random_state))
        grid, prefix = MLP_PARAM_GRID, "regressor__model__"
    elif model == "xgboost":
        estimator = Pipeline([("constant", VarianceThreshold()),
                              ("model", XGBRegressor(objective="reg:squarederror", n_jobs=1,
                                                     random_state=random_state))])
        grid, prefix = XGB_PARAM_GRID, "model__"
    else:
        raise ValueError(f"unknown model {model!r}")
    grid = {prefix + k: v for k, v in grid.items()}
    n_candidates = int(np.prod([len(v) for v in grid.values()]))
    return RandomizedSearchCV(estimator, grid, n_iter=min(n_iter, n_candidates), cv=inner_cv,
                              scoring="neg_mean_squared_error", random_state=random_state, n_jobs=1)


def _inner_cv(strat_train, n_splits, random_state):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
                .split(np.zeros(len(strat_train)), strat_train))


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _tune(model, X, y, strat, n_inner, n_iter, random_state):
    search = make_search(model, _inner_cv(strat, n_inner, random_state), n_iter, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X, y)
    return search


def run_outer_fold(X, y, strat, families, train_idx, test_idx, feature_set: str, repeat: int, fold: int,
                   n_inner: int = 3, n_iter: int = 25, per_family_min: int = None, random_state: int = 0):
    """Tune and evaluate every model on one outer fold.

    Returns (predictions, fold_rows): out-of-fold predictions per compound
    and model, and per-model fold summaries (train/test RMSE, tuned params).
    per_family_min: if set, also train XGBoost separately for each family
    with at least this many training compounds ('xgboost_per_family').
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    families = np.asarray(families)
    strat = np.asarray(strat)
    X_tr, y_tr, X_te = X[train_idx], y[train_idx], X[test_idx]

    preds, fold_rows, train_preds = {}, [], {}
    for model in BASE_MODELS:
        search = _tune(model, X_tr, y_tr, strat[train_idx], n_inner, n_iter, random_state)
        preds[model] = search.predict(X_te)
        train_preds[model] = search.predict(X_tr)
        fold_rows.append({"model": model, "params": json.dumps(search.best_params_, default=str)})
    preds[ENSEMBLE] = np.mean([preds[m] for m in BASE_MODELS], axis=0)
    train_preds[ENSEMBLE] = np.mean([train_preds[m] for m in BASE_MODELS], axis=0)
    fold_rows.append({"model": ENSEMBLE, "params": ""})

    if per_family_min is not None:
        per_family = np.full(len(test_idx), np.nan)
        per_family_train = np.full(len(train_idx), np.nan)
        tuned = {}
        for family in np.unique(families[train_idx]):
            fam_train = families[train_idx] == family
            if fam_train.sum() < per_family_min:
                continue
            search = _tune("xgboost", X_tr[fam_train], y_tr[fam_train], strat[train_idx][fam_train],
                           n_inner, n_iter, random_state)
            fam_test = families[test_idx] == family
            if fam_test.any():
                per_family[fam_test] = search.predict(X_te[fam_test])
            per_family_train[fam_train] = search.predict(X_tr[fam_train])
            tuned[family] = search.best_params_
        preds["xgboost_per_family"] = per_family
        train_preds["xgboost_per_family"] = per_family_train
        fold_rows.append({"model": "xgboost_per_family", "params": json.dumps(tuned, default=str)})

    for row in fold_rows:
        m = row["model"]
        covered_tr, covered_te = ~np.isnan(train_preds[m]), ~np.isnan(preds[m])
        row.update(feature_set=feature_set, repeat=repeat, fold=fold,
                   train_rmse=_rmse(y_tr[covered_tr], train_preds[m][covered_tr]),
                   test_rmse=_rmse(y[test_idx][covered_te], preds[m][covered_te]))
    predictions = pd.concat([
        pd.DataFrame({"feature_set": feature_set, "repeat": repeat, "fold": fold, "model": m,
                      "row": test_idx, "family": families[test_idx], "y_true": y[test_idx], "y_pred": p})
        for m, p in preds.items()
    ], ignore_index=True)
    return predictions.dropna(subset=["y_pred"]), pd.DataFrame(fold_rows)


def outer_splits(strat, n_splits: int = 5, n_repeats: int = 3, random_state: int = 0):
    """[(repeat, fold, train_idx, test_idx)] for repeated stratified K-fold."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return [(i // n_splits, i % n_splits, tr, te)
            for i, (tr, te) in enumerate(splitter.split(np.zeros(len(strat)), strat))]


def fold_metrics(predictions: pd.DataFrame, hard_mask=None) -> pd.DataFrame:
    """RMSE / MAE / R² per (feature_set, model, repeat, fold) from
    out-of-fold predictions; plus hard-region RMSE if a mask over rows is given."""
    def summarise(g):
        err = g["y_pred"] - g["y_true"]
        out = {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(err.abs().mean()),
               "r2": float(1 - (err ** 2).sum() / ((g["y_true"] - g["y_true"].mean()) ** 2).sum()),
               "n": len(g)}
        if hard_mask is not None:
            h = np.asarray(hard_mask)[g["row"].to_numpy()]
            out["rmse_hard"] = float(np.sqrt((err[h] ** 2).mean())) if h.any() else np.nan
        return pd.Series(out)
    return (predictions.groupby(["feature_set", "model", "repeat", "fold"])
            .apply(summarise, include_groups=False).reset_index())


def fit_final_model(model: str, X, y, strat, n_inner: int = 5, n_iter: int = 25, random_state: int = 0):
    """Run a recipe's tuning on all the data and fit the final model.
    For the ensemble, each member is tuned on all the data and the tuned
    members are averaged (VotingRegressor). Returns (fitted model, params)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    members = [model] if model in BASE_MODELS else list(BASE_MODELS)
    tuned = {m: _tune(m, X, y, np.asarray(strat), n_inner, n_iter, random_state) for m in members}
    params = {m: s.best_params_ for m, s in tuned.items()}
    if model in BASE_MODELS:
        return tuned[model].best_estimator_, params
    ensemble = VotingRegressor([(m, clone(s.best_estimator_)) for m, s in tuned.items()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return ensemble.fit(X, y), params
