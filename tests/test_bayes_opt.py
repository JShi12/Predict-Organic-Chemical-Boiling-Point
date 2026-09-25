import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from boiling_point.active_learning import fit_gp
from boiling_point.bayes_opt import (
    STRATEGIES, experiments_to_reach, prob_in_spec_batch, prob_in_window, qlogei_target_batch, run_campaign,
)


def _pool(n=160, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 2))
    y = 400 + 8 * X[:, 0] + 3 * X[:, 1] + rng.normal(scale=1.0, size=n)
    mw = 50 + 10 * X[:, 0] + rng.uniform(0, 5, size=n)
    return X, y, mw


def test_prob_in_window_matches_the_normal_distribution():
    p = prob_in_window(np.array([463.0]), np.array([10.0]), 453, 473)

    assert p[0] == pytest.approx(norm.cdf(1) - norm.cdf(-1))


def test_prob_in_spec_batch_spreads_across_identical_candidates():
    X, y, _ = _pool()
    bounds = np.stack([X.min(axis=0), X.max(axis=0)])
    model = fit_gp(X[:40], y[:40], bounds)
    # 3 copies of the best candidate plus other candidates: conditioning on
    # the first pick should make the copies less attractive than new points
    X_cand = np.vstack([np.repeat(X[40:41], 3, axis=0), X[41:80]])

    picks = prob_in_spec_batch(model, X_cand, 4, lo=440, hi=460)

    assert len(set(picks)) == 4
    assert sum(p < 3 for p in picks) < 3


def test_qlogei_target_batch_returns_distinct_candidate_positions():
    X, y, _ = _pool()
    bounds = np.stack([X.min(axis=0), X.max(axis=0)])
    model = fit_gp(X[:40], y[:40], bounds)

    picks = qlogei_target_batch(model, X[40:], 5, y[:40], centre=450)

    assert len(set(picks)) == 5
    assert all(0 <= p < len(X) - 40 for p in picks)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_run_campaign_respects_budget_and_never_repeats(strategy):
    X, y, mw = _pool()

    r = run_campaign(X, y, mw, strategy, seed=1, lo=440, hi=460, n_initial=8, batch_size=4, budget=24)

    assert len(r) == 24 and r["row"].is_unique
    assert r["hits"].iloc[-1] == ((r["boiling_point"] >= 440) & (r["boiling_point"] <= 460)).sum()


def test_campaigns_share_the_initial_experiments_and_bo_beats_random_on_a_smooth_pool():
    X, y, mw = _pool(n=300)
    kwargs = dict(seed=3, lo=440, hi=460, n_initial=10, batch_size=5, budget=50)

    rand = run_campaign(X, y, mw, "random", **kwargs)
    bo = run_campaign(X, y, mw, "prob_in_spec", **kwargs)

    assert rand["row"].iloc[:10].tolist() == bo["row"].iloc[:10].tolist()
    assert bo["hits"].iloc[-1] > rand["hits"].iloc[-1]


def test_experiments_to_reach_averages_over_seeds():
    results = pd.DataFrame({"strategy": "s", "seed": [0, 0, 1, 1], "experiment": [1, 2, 1, 2], "hits": [0, 1, 1, 1]})

    table = experiments_to_reach(results, targets=(1, 5))

    assert table.loc["s", "experiments to 1 hits"] == 1.5
    assert np.isnan(table.loc["s", "experiments to 5 hits"]) and table.loc["s", "seeds reaching 5"] == "0/2"
