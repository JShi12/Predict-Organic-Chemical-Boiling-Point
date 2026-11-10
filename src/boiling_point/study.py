"""Frozen-representation comparison for the representation study (Step 3).

Tasks come from the fixed splits (results/representation_study/): 15
familiar-chemistry folds, 4 held-out families, and the external NIST set.
Each task has its own inner folds, shared by every representation and
regressor, so all pipelines are tuned on the same data:
- familiar / NIST: StratifiedGroupKFold(3) on the training rows (family
  strata, structure groups kept whole);
- family holdout: leave one of the three training families out, so the
  hyperparameters are chosen for extrapolation to an unseen family.

Pipelines (fixed before any result):
- Ridge: drop constant columns, [drop near-duplicates: descriptors only],
  standardise X and y. Alpha grid ALPHAS, the same for every representation.
- XGBoost: drop constant columns; the project's XGB_PARAM_GRID, fully searched.
- MoLFormer: the layer is chosen among the 14 by Ridge in the inner CV;
  XGBoost uses that layer.
Selection metric: inner-CV MAE.

No torch here: this module fits xgboost, and the encoder must run in a
separate process (see representations.py).
"""
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import validation
from .models import XGB_PARAM_GRID

ALPHAS = [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
N_INNER = 3
REPRESENTATIONS = ("curated", "morgan", "molformer")


# --- tasks and inner folds ----------------------------------------------------

def familiar_tasks(splits: pd.DataFrame, n_repeats: int = 3, n_folds: int = 5) -> list:
    tasks = []
    for r in range(n_repeats):
        for k in range(n_folds):
            fold = splits[f"fold_r{r}"].to_numpy()
            tasks.append({"regime": "familiar", "task": f"r{r}f{k}", "repeat": r, "fold": k,
                          "train": np.flatnonzero(fold != k), "test": np.flatnonzero(fold == k)})
    return tasks


def holdout_tasks(splits: pd.DataFrame) -> list:
    hold = splits["holdout_family"].fillna("").to_numpy()
    tasks = []
    for family in sorted(set(hold) - {""}):
        tasks.append({"regime": "holdout", "task": family, "repeat": 0, "fold": 0,
                      "train": np.flatnonzero((hold != "") & (hold != family)), "test": np.flatnonzero(hold == family)})
    return tasks


def learning_curve_tasks(splits: pd.DataFrame, sizes=(100, 250, 500), draws: int = 3, n_folds: int = 5) -> list:
    """Repeat-0 folds trained on the nested subsets saved in splits.csv
    (`lc_entry_f{k}_d{d}`: -1 in the test fold; a row is in the size-n subset
    if its entry is between 0 and n). Test rows are the unchanged fold."""
    tasks = []
    fold = splits["fold_r0"].to_numpy()
    for k in range(n_folds):
        for d in range(draws):
            entry = splits[f"lc_entry_f{k}_d{d}"].to_numpy()
            for n in sizes:
                tasks.append({"regime": "learning_curve", "task": f"n{n}_f{k}_d{d}", "repeat": 0, "fold": k,
                              "size": n, "draw": d, "train": np.flatnonzero((entry >= 0) & (entry <= n)),
                              "test": np.flatnonzero(fold == k)})
    return tasks


def nist_task(n_literature: int, nist_rows: np.ndarray) -> dict:
    """Train on every literature row; test on the NIST rows, which sit after
    the literature rows in the combined arrays."""
    return {"regime": "nist", "task": "nist", "repeat": 0, "fold": 0,
            "train": np.arange(n_literature), "test": n_literature + np.asarray(nist_rows)}


def inner_folds(task: dict, families: np.ndarray, groups: np.ndarray, random_state: int = 0) -> list:
    """[(train, validation)] as positions within task['train']."""
    train = task["train"]
    fam, grp = families[train], groups[train]
    if task["regime"] == "holdout":
        return [(np.flatnonzero(fam != f), np.flatnonzero(fam == f)) for f in sorted(set(fam))]
    strat = validation.stratification_labels(pd.Series(fam), N_INNER)
    splitter = StratifiedGroupKFold(N_INNER, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(train)), strat, grp))


# --- pipelines ------------------------------------------------------------------

def ridge_pipeline(representation: str):
    if representation == "curated":
        return validation._scaled(Ridge())   # includes DropCorrelated, as in the feature study
    pipe = Pipeline([("constant", VarianceThreshold()), ("scale", StandardScaler()), ("model", Ridge())])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def xgb_pipeline(random_state: int = 0):
    return Pipeline([("constant", VarianceThreshold()),
                     ("model", XGBRegressor(objective="reg:squarederror", n_jobs=1, random_state=random_state))])


def ridge_grid(alphas=ALPHAS) -> list:
    return [{"regressor__model__alpha": a} for a in alphas]


def xgb_grid(grid: dict = XGB_PARAM_GRID) -> list:
    keys = sorted(grid)
    return [{f"model__{k}": v for k, v in zip(keys, values)} for values in itertools.product(*(grid[k] for k in keys))]


# --- tuning -------------------------------------------------------------------------

def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def inner_scores(estimator, params: dict, X, y, folds) -> float:
    """Mean inner-validation MAE of one configuration (inf if a fit fails)."""
    maes = []
    for tr, va in folds:
        try:
            model = clone(estimator).set_params(**params).fit(X[tr], y[tr])
        except (ValueError, FloatingPointError):
            return np.inf
        maes.append(_mae(y[va], model.predict(X[va])))
    return float(np.mean(maes))


def tune(estimator, grid: list, X, y, folds) -> dict:
    """Grid search on the given inner folds. Returns the best configuration,
    its inner MAE, and every configuration's score."""
    scores = [inner_scores(estimator, p, X, y, folds) for p in grid]
    best = int(np.argmin(scores))
    return {"params": grid[best], "inner_mae": scores[best], "scores": scores}


def fit_and_record(estimator, params: dict, X_train, y_train, X_test, folds) -> dict:
    """Refit the chosen configuration on the whole training set, predict the
    test rows; and, for CV+ intervals (Step 5), refit it on each inner fold,
    keeping that model's test predictions and its out-of-fold residuals."""
    start = time.time()
    model = clone(estimator).set_params(**params).fit(X_train, y_train)
    pred = model.predict(X_test)
    fit_seconds = time.time() - start
    inner_test, inner_resid = [], []
    for i, (tr, va) in enumerate(folds):
        m = clone(estimator).set_params(**params).fit(X_train[tr], y_train[tr])
        inner_test.append(m.predict(X_test))
        inner_resid.append((i, va, y_train[va] - m.predict(X_train[va])))
    return {"pred": pred, "fit_seconds": fit_seconds, "inner_test": inner_test, "inner_resid": inner_resid}


def run_task(task: dict, representation: str, regressor: str, X, y, families, groups, layer=None,
             alphas=ALPHAS, xgb_param_grid: dict = XGB_PARAM_GRID, layer_probe: bool = False) -> dict:
    """Tune and evaluate one (task, representation, regressor).

    X: (n, d) features, or for MoLFormer (n_layers, n, d). Ridge on MoLFormer
    chooses the layer and alpha jointly in the inner CV; XGBoost on MoLFormer
    needs `layer` (the one Ridge chose). With layer_probe, also returns each
    layer's test MAE at its own inner-CV-best alpha (a diagnostic only)."""
    start = time.time()
    train, test = task["train"], task["test"]
    folds = inner_folds(task, families, groups)
    y_tr = y[train]
    probe = []
    if regressor == "ridge":
        estimator, grid = ridge_pipeline(representation), ridge_grid(alphas)
        if representation == "molformer":
            per_layer = [tune(estimator, grid, X[l][train], y_tr, folds) for l in range(X.shape[0])]
            layer = int(np.argmin([t["inner_mae"] for t in per_layer]))
            best = per_layer[layer]
            if layer_probe:
                for l, t in enumerate(per_layer):
                    m = clone(estimator).set_params(**t["params"]).fit(X[l][train], y_tr)
                    probe.append({"layer": l, "alpha": t["params"]["regressor__model__alpha"],
                                  "inner_mae": t["inner_mae"], "test_mae": _mae(y[test], m.predict(X[l][test]))})
        else:
            best = tune(estimator, grid, X[train], y_tr, folds)
    elif regressor == "xgboost":
        estimator, grid = xgb_pipeline(), xgb_grid(xgb_param_grid)
        if representation == "molformer" and layer is None:
            raise ValueError("XGBoost on MoLFormer needs the layer chosen by Ridge")
        best = tune(estimator, grid, (X[layer] if representation == "molformer" else X)[train], y_tr, folds)
    else:
        raise ValueError(f"unknown regressor {regressor!r}")
    Xm = X[layer] if representation == "molformer" else X
    record = fit_and_record(estimator, best["params"], Xm[train], y_tr, Xm[test], folds)
    return {**{k: task[k] for k in ("regime", "task", "repeat", "fold")}, "representation": representation,
            "regressor": regressor, "layer": layer, "params": best["params"], "inner_mae": best["inner_mae"],
            "train": train, "test": test, "y_true": y[test], **record, "probe": probe,
            "seconds": time.time() - start}
