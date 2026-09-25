"""Bayesian optimisation for a specification window.

The question a formulation lab asks is rarely "what's the maximum?" but
"which candidates meet the spec?". Here the spec is a boiling-point
window, and each 'experiment' reveals one compound's measured boiling
point. A campaign starts from a few random experiments and then picks
batches of candidates; success is how many in-spec compounds it finds
per experiment.

Strategies:
- random: uniform picks.
- mw_heuristic: a one-feature model (boiling point ~ log molecular
  weight, refitted each round) picks compounds predicted closest to the
  window centre -- a chemist's quick rule.
- greedy_gp: GP predictive mean closest to the centre (pure exploitation).
- prob_in_spec: GP probability that the boiling point lies in the window;
  batches are built greedily, conditioning the GP on each pick at its
  predicted value ("kriging believer") so a batch spreads out.
- qlogei_target: BoTorch's qLogExpectedImprovement on the objective
  -|Tb - centre|, optimised jointly over each batch (optimize_acqf_discrete).
"""
import warnings

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf_discrete
from scipy.stats import norm

from .active_learning import fit_gp

STRATEGIES = ("random", "mw_heuristic", "greedy_gp", "prob_in_spec", "qlogei_target")


def predictive(model, X) -> tuple:
    """GP predictive mean and std (K) including observation noise -- the
    distribution of the value an experiment would actually report."""
    with torch.no_grad():
        post = model.posterior(torch.as_tensor(np.asarray(X), dtype=torch.double), observation_noise=True)
        mean, var = post.mean.detach(), post.variance.detach()
    return mean.squeeze(-1).numpy(), var.clamp_min(1e-12).sqrt().squeeze(-1).numpy()


def prob_in_window(mean, std, lo: float, hi: float) -> np.ndarray:
    return norm.cdf((hi - mean) / std) - norm.cdf((lo - mean) / std)


def prob_in_spec_batch(model, X_cand, batch_size: int, lo: float, hi: float) -> list:
    """Greedy batch: pick the candidate most likely to be in spec, condition
    the GP on it at its predicted value (which leaves the mean unchanged
    but shrinks nearby uncertainty), repeat."""
    X_t = torch.as_tensor(np.asarray(X_cand), dtype=torch.double)
    picks, current = [], model
    for _ in range(min(batch_size, len(X_t))):
        mean, std = predictive(current, X_t)
        score = prob_in_window(mean, std, lo, hi)
        score[picks] = -np.inf
        j = int(np.argmax(score))
        picks.append(j)
        with torch.no_grad():
            believed = current.posterior(X_t[j:j + 1]).mean
        current = current.condition_on_observations(X_t[j:j + 1], believed)
    return picks


def qlogei_target_batch(model, X_cand, batch_size: int, y_observed, centre: float) -> list:
    """BoTorch qLogEI on -|Tb - centre|, jointly over a batch of candidates."""
    objective = GenericMCObjective(lambda samples, X=None: -(samples[..., 0] - centre).abs())
    best_f = float(-np.min(np.abs(np.asarray(y_observed) - centre)))
    acq = qLogExpectedImprovement(model, best_f=best_f, objective=objective)
    choices = torch.as_tensor(np.asarray(X_cand), dtype=torch.double)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chosen, _ = optimize_acqf_discrete(acq, q=min(batch_size, len(choices)), choices=choices, unique=True)
    # map chosen rows back to candidate positions
    matches = [int(torch.nonzero((choices == row).all(dim=1))[0]) for row in chosen]
    return matches


def run_campaign(X, y, mw, strategy: str, seed: int, lo: float, hi: float,
                 n_initial: int = 10, batch_size: int = 5, budget: int = 150) -> pd.DataFrame:
    """One campaign over the pool (rows of X). Returns one row per
    experiment: order, row index, boiling point, whether it's in spec,
    and the cumulative number of in-spec compounds found."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mw = np.asarray(mw, dtype=float)
    centre = (lo + hi) / 2
    bounds = np.stack([X.min(axis=0), X.max(axis=0)])
    rng = np.random.default_rng(seed)
    labelled = list(rng.choice(len(X), n_initial, replace=False))
    strategy_rng = np.random.default_rng([seed, STRATEGIES.index(strategy)])

    while len(labelled) < budget:
        unlabelled = np.setdiff1d(np.arange(len(X)), labelled)
        q = min(batch_size, budget - len(labelled))
        if strategy == "random":
            picks = list(strategy_rng.choice(len(unlabelled), q, replace=False))
        elif strategy == "mw_heuristic":
            slope, intercept = np.polyfit(np.log(mw[labelled]), y[labelled], 1)
            predicted = slope * np.log(mw[unlabelled]) + intercept
            picks = list(np.argsort(np.abs(predicted - centre), kind="stable")[:q])
        else:
            model = fit_gp(X[labelled], y[labelled], bounds)
            if strategy == "greedy_gp":
                mean, _ = predictive(model, X[unlabelled])
                picks = list(np.argsort(np.abs(mean - centre), kind="stable")[:q])
            elif strategy == "prob_in_spec":
                picks = prob_in_spec_batch(model, X[unlabelled], q, lo, hi)
            elif strategy == "qlogei_target":
                picks = qlogei_target_batch(model, X[unlabelled], q, y[labelled], centre)
            else:
                raise ValueError(f"unknown strategy {strategy!r}")
        labelled.extend(int(i) for i in unlabelled[picks])

    order = np.array(labelled)
    in_spec = (y[order] >= lo) & (y[order] <= hi)
    return pd.DataFrame({"strategy": strategy, "seed": seed, "experiment": np.arange(1, len(order) + 1),
                         "row": order, "boiling_point": y[order], "in_spec": in_spec,
                         "hits": np.cumsum(in_spec)})


def experiments_to_reach(results: pd.DataFrame, targets=(10, 25, 50)) -> pd.DataFrame:
    """Per strategy: mean number of experiments to find each target number
    of in-spec compounds (NaN where some seed never got there), plus how
    many seeds reached it."""
    rows = []
    for strategy, g in results.groupby("strategy"):
        row = {"strategy": strategy}
        for t in targets:
            per_seed = g[g["hits"] >= t].groupby("seed")["experiment"].min()
            n_seeds = g["seed"].nunique()
            row[f"experiments to {t} hits"] = per_seed.mean() if len(per_seed) == n_seeds else np.nan
            row[f"seeds reaching {t}"] = f"{len(per_seed)}/{n_seeds}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")
