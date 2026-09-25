"""Active learning: which compounds should be measured next to improve the
model fastest?

A selector model ("scout") picks each batch of compounds to label; the
production ensemble (Ridge + XGBoost + NN) is then retrained on whatever has
been labelled so far and scored on a fixed test set. The selector only
decides what to measure -- the ensemble's test RMSE is the success measure,
so every strategy (including ones with no model, like random) is judged the
same way.

Selection strategies compared:
- random: uniform picks from the unlabelled pool.
- heuristic: the hand-written rule behind the NIST collection -- random
  picks from the high polar area / rotatable bond region until it runs out.
- gp_uncertainty: a Gaussian process picks the compound it is least sure
  about, is conditioned on that pick, and repeats (keeps a batch diverse
  instead of 25 near-identical isomers).
- bootstrap_xgb: a bootstrap ensemble of XGBoost models picks the compounds
  its members disagree on most.
"""
import warnings

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.utils.cholesky import psd_safe_cholesky
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from .ensemble import AverageEnsembleRegressor

# torch and xgboost each bring their own OpenMP runtime; on macOS, both
# running thread pools in one process segfaults xgboost (see also
# __init__.py). The GP fits here are small, so torch stays single-threaded;
# parallelism comes from running seeds in separate processes instead.
torch.set_num_threads(1)

STRATEGIES = ("random", "heuristic", "gp_uncertainty", "bootstrap_xgb")

# Hyperparameters from the main notebook's train+val re-tune of the ensemble
# members (Boiling_Point_Predicter.ipynb, "Build the Ensemble"). Held fixed
# here: re-running GridSearchCV at every checkpoint would be slow and would
# add noise unrelated to which compounds were labelled.
XGB_PARAMS = dict(objective="reg:squarederror", learning_rate=0.1, max_depth=3,
                  min_child_weight=3, n_estimators=300)
MLP_PARAMS = dict(hidden_layer_sizes=(100, 50, 20), activation="relu", solver="adam",
                  alpha=0.0001, learning_rate="constant", learning_rate_init=0.001,
                  max_iter=1000, tol=1e-4)

# Label counts at which the production ensemble is evaluated during the
# small-batch phase; every round after SWITCH_AT is also evaluated.
EARLY_CHECKPOINTS = (50, 100, 200, 300, 400)


def make_production_ensemble(random_state: int = 0) -> AverageEnsembleRegressor:
    """The main notebook's Ridge + XGBoost + NN ensemble, fixed hyperparameters."""
    return AverageEnsembleRegressor(
        ridge=Ridge(alpha=0.1),
        xgb=XGBRegressor(random_state=random_state, n_jobs=1, **XGB_PARAMS),
        nn=MLPRegressor(random_state=random_state, **MLP_PARAMS),
    )


def batch_schedule(n_pool: int, n_initial: int = 50, small_batch: int = 25,
                   switch_at: int = 550, large_batch: int = 100) -> list:
    """Batch sizes that take the labelled set from n_initial to the whole
    pool: small batches up to switch_at labels (where strategies should
    differ most), then large batches until the pool is used up."""
    sizes, n = [], n_initial
    while n < n_pool:
        size = small_batch if n < switch_at else large_batch
        size = min(size, n_pool - n)
        sizes.append(size)
        n += size
    return sizes


def is_checkpoint(n_labels: int, n_pool: int, switch_at: int = 550) -> bool:
    return n_labels in EARLY_CHECKPOINTS or n_labels >= switch_at or n_labels == n_pool


# --- Gaussian process selector -------------------------------------------------

def gp_inputs(X) -> np.ndarray:
    """log1p of the (all non-negative, right-skewed) size/count features, so
    one length scale per feature is reasonable across their range."""
    return np.log1p(np.asarray(X, dtype=float))


# Kernel hyperparameters (length scales, output scale, noise) are
# re-optimised only once the labelled set has grown by this factor since the
# last optimisation; in between, the GP is refit on the new data with the
# previous hyperparameters. A few dozen new labels barely move them, and
# optimising on ~1,000+ compounds takes minutes.
REOPTIMISE_GROWTH = 1.2


def fit_gp(X_train, y_train, bounds: np.ndarray, hyperparameters_from: SingleTaskGP = None,
           optimise: bool = True) -> SingleTaskGP:
    """Exact GP with an ARD kernel (one length scale per feature, BoTorch's
    default) and a learned noise term, which absorbs isomers that share a
    feature vector but differ in boiling point. X_train must already be
    passed through gp_inputs; bounds is its (2, d) min/max over the whole
    candidate pool, so Normalize doesn't shift as labels are added.

    hyperparameters_from: copy kernel/noise/mean hyperparameters from an
    earlier fit -- as the starting point when optimise=True (converges much
    faster), or as-is when optimise=False (no optimisation, still an exact
    GP posterior on the new data)."""
    train_X = torch.as_tensor(np.asarray(X_train), dtype=torch.double)
    train_Y = torch.as_tensor(np.asarray(y_train, dtype=float), dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=train_X.shape[-1], bounds=torch.as_tensor(bounds, dtype=torch.double)),
        outcome_transform=Standardize(m=1),
    )
    if hyperparameters_from is not None:
        for name in ("covar_module", "likelihood", "mean_module"):
            getattr(model, name).load_state_dict(getattr(hyperparameters_from, name).state_dict())
    if optimise or hyperparameters_from is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


def gp_predict(model: SingleTaskGP, X) -> tuple:
    """Posterior mean and standard deviation (K) at X (already gp_inputs'd)."""
    with torch.no_grad():
        posterior = model.posterior(torch.as_tensor(np.asarray(X), dtype=torch.double))
    return (posterior.mean.squeeze(-1).numpy(),
            posterior.variance.clamp_min(0).sqrt().squeeze(-1).numpy())


def gp_select_batch(model: SingleTaskGP, X_candidates, batch_size: int) -> list:
    """Greedy batch uncertainty sampling: pick the most uncertain candidate,
    condition the GP on it, recompute uncertainties, repeat. This keeps a
    batch spread out instead of 25 near-identical isomers.

    A GP's posterior variance doesn't depend on the observed value, so
    conditioning on a pick j is exact without knowing its label: every
    candidate's variance drops by cov(i, j)^2 / (var(j) + noise). That needs
    one Cholesky factorisation per batch plus a rank-1 update per pick,
    instead of refactorising the GP after every pick. Works in the
    standardized outcome space, which rescales every variance equally and so
    doesn't change the ranking. Returns positions into X_candidates, in pick
    order.
    """
    kernel = model.covar_module
    with torch.no_grad():
        train_Z = model.train_inputs[0]  # already normalized in eval mode
        Z = model.input_transform(torch.as_tensor(np.asarray(X_candidates), dtype=torch.double))
        noise = model.likelihood.noise.squeeze()
        K = kernel(train_Z).to_dense() + noise * torch.eye(len(train_Z), dtype=train_Z.dtype)
        L = psd_safe_cholesky(K)
        V = torch.linalg.solve_triangular(L, kernel(train_Z, Z).to_dense(), upper=False)
        var = kernel(Z, diag=True) - (V ** 2).sum(dim=0)

        picked = torch.zeros(len(Z), dtype=torch.bool)
        picks, updates = [], []
        for _ in range(min(batch_size, len(Z))):
            j = int(torch.argmax(var.masked_fill(picked, -torch.inf)))
            picked[j] = True
            picks.append(j)
            # Posterior covariance of every candidate with j, given the
            # training data and the picks so far.
            cov_j = kernel(Z, Z[j:j + 1]).to_dense().squeeze(-1) - V.T @ V[:, j]
            for u in updates:
                cov_j -= u * u[j]
            u = cov_j / torch.sqrt(var[j] + noise)
            var = var - u ** 2
            updates.append(u)
    return picks


# --- Bootstrap XGBoost selector ------------------------------------------------

def bootstrap_xgb_models(X_train, y_train, n_models: int = 15, random_state: int = 0) -> list:
    rng = np.random.default_rng(random_state)
    X_arr, y_arr = np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float)
    models = []
    for i in range(n_models):
        idx = rng.integers(0, len(X_arr), len(X_arr))
        models.append(XGBRegressor(random_state=i, n_jobs=1, **XGB_PARAMS).fit(X_arr[idx], y_arr[idx]))
    return models


def bootstrap_predict(models: list, X) -> tuple:
    preds = np.stack([m.predict(np.asarray(X, dtype=float)) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


def bootstrap_select_batch(models: list, X_candidates, batch_size: int) -> list:
    """Top batch_size candidates by spread across the bootstrap members."""
    _, std = bootstrap_predict(models, X_candidates)
    return list(np.argsort(-std, kind="stable")[:batch_size])


# --- Feasibility-aware selection -----------------------------------------------
# Uncertainty alone favours exotic compounds (large drug/dye-like molecules)
# that often have no normal boiling point at all -- they decompose first -- so
# a NIST lookup for them is wasted. Weighting uncertainty by the predicted
# chance that the lookup succeeds ranks by expected useful uncertainty per
# lookup instead.

def fit_feasibility_classifier(X, found, random_state: int = 0) -> XGBClassifier:
    """P(a NIST lookup returns a boiling point) from the molecular features,
    trained on recorded lookup outcomes (found = 1, anything else = 0)."""
    clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=3,
                        random_state=random_state, n_jobs=1)
    return clf.fit(np.asarray(X, dtype=float), np.asarray(found, dtype=int))


def feasibility_weighted_batch(models: list, classifier, X_candidates, batch_size: int) -> list:
    """Top batch_size candidates by bootstrap spread x P(lookup succeeds)."""
    X_cand = np.asarray(X_candidates, dtype=float)
    _, std = bootstrap_predict(models, X_cand)
    p_found = classifier.predict_proba(X_cand)[:, 1]
    return list(np.argsort(-(std * p_found), kind="stable")[:batch_size])


# --- Simulation ----------------------------------------------------------------

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def model_free_batch(strategy: str, rng, hard_unlabelled, batch_size: int) -> list:
    """Positions of the next batch for the strategies that use no model.
    hard_unlabelled: hard-region mask over the unlabelled pool."""
    n = len(hard_unlabelled)
    if strategy == "random":
        return list(rng.choice(n, batch_size, replace=False))
    if strategy == "heuristic":
        in_region = np.flatnonzero(hard_unlabelled)
        if len(in_region) >= batch_size:
            return list(rng.choice(in_region, batch_size, replace=False))
        rest = np.flatnonzero(~np.asarray(hard_unlabelled))
        return list(in_region) + list(rng.choice(rest, batch_size - len(in_region), replace=False))
    raise ValueError(f"unknown strategy {strategy!r}")


def run_strategy(X, y, hard_mask, test_idx, pool_idx, strategy: str, seed: int,
                 n_initial: int = 50, small_batch: int = 25, switch_at: int = 550,
                 large_batch: int = 100, evaluate=None, max_labels: int = None) -> pd.DataFrame:
    """One active-learning campaign on a fixed test set.

    Each round the selector is refit on every label so far and picks the
    next batch (the active learning); only at checkpoints is the production
    ensemble retrained on the labels so far and scored on the test set. The
    initial random labels depend only on seed, so strategies are compared
    from the same starting point.

    evaluate: optional callable(X_labelled, y_labelled) -> fitted model with
    .predict, defaults to make_production_ensemble() -- overridable in tests.

    max_labels: stop once this many compounds are labelled (for quick
    smoke runs); by default the campaign runs until the pool is used up.

    Returns one row per checkpoint; results.attrs["labelled_order"] holds
    the row indices in the order they were labelled.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    hard = np.asarray(hard_mask, dtype=bool)
    pool_idx = np.asarray(pool_idx)
    test_idx = np.asarray(test_idx)
    X_gp = gp_inputs(X)
    bounds = np.stack([X_gp.min(axis=0), X_gp.max(axis=0)])
    if evaluate is None:
        def evaluate(X_lab, y_lab):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                return make_production_ensemble().fit(X_lab, y_lab)

    init_rng = np.random.default_rng(seed)
    labelled = list(init_rng.choice(pool_idx, n_initial, replace=False))
    strategy_rng = np.random.default_rng([seed, STRATEGIES.index(strategy)])
    sizes = batch_schedule(len(pool_idx), n_initial, small_batch, switch_at, large_batch)
    if max_labels is not None:
        sizes = [s for s, total in zip(sizes, n_initial + np.cumsum(sizes)) if total <= max_labels]
    hard_test = hard[test_idx]

    rows = []
    selector, last_optimised_n = None, 0
    for round_no in range(len(sizes) + 1):
        unlabelled = np.setdiff1d(pool_idx, labelled)
        if strategy == "gp_uncertainty":
            optimise = len(labelled) >= REOPTIMISE_GROWTH * last_optimised_n
            selector = fit_gp(X_gp[labelled], y[labelled], bounds,
                              hyperparameters_from=selector, optimise=optimise)
            if optimise:
                last_optimised_n = len(labelled)
        elif strategy == "bootstrap_xgb":
            selector = bootstrap_xgb_models(X[labelled], y[labelled], random_state=seed * 1000 + round_no)

        n_labels = len(labelled)
        if is_checkpoint(n_labels, len(pool_idx), switch_at):
            model = evaluate(X[labelled], y[labelled])
            pred = model.predict(X[test_idx])
            row = {
                "seed": seed, "strategy": strategy, "n_labels": n_labels,
                "rmse": _rmse(y[test_idx], pred),
                "rmse_hard": _rmse(y[test_idx][hard_test], pred[hard_test]) if hard_test.any() else np.nan,
                "frac_hard_labelled": float(hard[labelled].mean()),
                "selector_rmse": np.nan,
            }
            if strategy == "gp_uncertainty":
                row["selector_rmse"] = _rmse(y[test_idx], gp_predict(selector, X_gp[test_idx])[0])
            elif strategy == "bootstrap_xgb":
                row["selector_rmse"] = _rmse(y[test_idx], bootstrap_predict(selector, X[test_idx])[0])
            rows.append(row)

        if round_no == len(sizes):
            break
        batch_size = sizes[round_no]
        if strategy == "gp_uncertainty":
            picks = gp_select_batch(selector, X_gp[unlabelled], batch_size)
        elif strategy == "bootstrap_xgb":
            picks = bootstrap_select_batch(selector, X[unlabelled], batch_size)
        else:
            picks = model_free_batch(strategy, strategy_rng, hard[unlabelled], batch_size)
        labelled.extend(int(i) for i in unlabelled[picks])
    results = pd.DataFrame(rows)
    results.attrs["labelled_order"] = [int(i) for i in labelled]
    return results


def labels_to_match(results: pd.DataFrame, budgets=(200, 400, 550), reference: str = "random",
                    metric: str = "rmse") -> pd.DataFrame:
    """For each strategy, how many labels it needs (linearly interpolated on
    the seed-averaged curve) to reach the reference strategy's RMSE at each
    budget. NaN if it never gets there within the run."""
    curves = results.groupby(["strategy", "n_labels"])[metric].mean().unstack("strategy")
    out = {}
    for budget in budgets:
        target = curves.loc[budget, reference]
        needed = {}
        for strategy in curves.columns:
            curve = curves[strategy].dropna()
            below = curve[curve <= target]
            if below.empty:
                needed[strategy] = np.nan
                continue
            n_hit = below.index[0]
            earlier = curve[curve.index < n_hit]
            if earlier.empty:
                needed[strategy] = float(n_hit)
                continue
            n_prev, r_prev, r_hit = earlier.index[-1], earlier.iloc[-1], curve[n_hit]
            frac = (r_prev - target) / (r_prev - r_hit)
            needed[strategy] = float(n_prev + frac * (n_hit - n_prev))
        out[budget] = needed
    return pd.DataFrame(out).T.rename_axis(f"{reference} budget")


def uncertainty_calibration(X, y, hard_mask, labelled, test_idx, seed: int = 0) -> pd.DataFrame:
    """Does each selector's uncertainty point at where the production model
    is actually wrong? Fits the GP, the bootstrap XGBoost ensemble and the
    production ensemble on the same labelled rows, then on test_idx reports
    per selector:
    - rho_own / rho_ensemble: Spearman correlation of predicted uncertainty
      with the selector's own / the production ensemble's absolute error;
    - rho_distance: correlation of uncertainty with distance (in the GP's
      normalized input space) to the nearest labelled compound;
    - hard_over_easy: mean uncertainty in the hard region / elsewhere,
      next to the same ratio for the ensemble's actual error;
    - top10_error_ratio: ensemble error on the 10% of test compounds the
      selector is most uncertain about, relative to the average error.
    """
    from scipy.stats import spearmanr

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    hard_test = np.asarray(hard_mask, dtype=bool)[test_idx]
    X_gp = gp_inputs(X)
    bounds = np.stack([X_gp.min(axis=0), X_gp.max(axis=0)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ensemble = make_production_ensemble().fit(X[labelled], y[labelled])
    ens_err = np.abs(ensemble.predict(X[test_idx]) - y[test_idx])
    Z = (X_gp - bounds[0]) / (bounds[1] - bounds[0])
    distance = np.sqrt(((Z[test_idx][:, None] - Z[labelled][None]) ** 2).sum(-1)).min(axis=1)

    gp = fit_gp(X_gp[labelled], y[labelled], bounds)
    boot = bootstrap_xgb_models(X[labelled], y[labelled], random_state=seed)
    selectors = {
        "gp_uncertainty": gp_predict(gp, X_gp[test_idx]),
        "bootstrap_xgb": bootstrap_predict(boot, X[test_idx]),
    }
    rows = []
    for name, (mean, std) in selectors.items():
        top = np.argsort(-std)[:max(1, len(std) // 10)]
        rows.append({
            "selector": name, "n_labels": len(labelled),
            "rho_own": spearmanr(std, np.abs(mean - y[test_idx]))[0],
            "rho_ensemble": spearmanr(std, ens_err)[0],
            "rho_distance": spearmanr(std, distance)[0],
            "hard_over_easy": std[hard_test].mean() / std[~hard_test].mean(),
            "ensemble_error_hard_over_easy": ens_err[hard_test].mean() / ens_err[~hard_test].mean(),
            "top10_error_ratio": ens_err[top].mean() / ens_err.mean(),
        })
    return pd.DataFrame(rows)


def compare_additions(X_base, y_base, X_test, y_test, hard_test, additions: dict,
                      seeds=range(5), n_boot: int = 2000) -> pd.DataFrame:
    """Does adding a set of new compounds improve the production ensemble?

    Trains the ensemble on the base data alone and on base + each addition
    (same fixed hyperparameters; predictions averaged over `seeds` of the
    ensemble's own randomness), then scores all of them on the same test
    set. Differences from the base model come with 95% bootstrap intervals
    over test rows (paired: same resampled rows for both models).

    additions: {name: (X_add, y_add)}. Returns one row per arm (base first)
    with rmse / rmse_hard and, for additions, the change vs base.
    """
    X_base, y_base = np.asarray(X_base, dtype=float), np.asarray(y_base, dtype=float)
    X_test, y_test = np.asarray(X_test, dtype=float), np.asarray(y_test, dtype=float)
    hard_test = np.asarray(hard_test, dtype=bool)

    def averaged_predictions(X_train, y_train):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return np.mean([make_production_ensemble(random_state=s).fit(X_train, y_train).predict(X_test)
                            for s in seeds], axis=0)

    arms = {"base": averaged_predictions(X_base, y_base)}
    for name, (X_add, y_add) in additions.items():
        arms[name] = averaged_predictions(np.vstack([X_base, np.asarray(X_add, dtype=float)]),
                                          np.concatenate([y_base, np.asarray(y_add, dtype=float)]))

    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(y_test), size=(n_boot, len(y_test)))
    hard_idx = np.flatnonzero(hard_test)
    boot_hard = hard_idx[rng.integers(0, len(hard_idx), size=(n_boot, len(hard_idx)))]

    def rmse_rows(pred, rows):
        return np.sqrt(((pred[rows] - y_test[rows]) ** 2).mean(axis=-1))

    out = []
    for name, pred in arms.items():
        row = {"arm": name, "n_added": 0 if name == "base" else len(additions[name][1]),
               "rmse": _rmse(y_test, pred), "rmse_hard": _rmse(y_test[hard_test], pred[hard_test])}
        if name != "base":
            for metric, rows in (("rmse", boot), ("rmse_hard", boot_hard)):
                diff = rmse_rows(pred, rows) - rmse_rows(arms["base"], rows)
                row[f"{metric}_change"] = row[metric] - out[0][metric]
                row[f"{metric}_change_ci"] = tuple(float(v) for v in np.round(np.percentile(diff, [2.5, 97.5]), 2))
        out.append(row)
    return pd.DataFrame(out).set_index("arm")
