import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import Ridge

from boiling_point.active_learning import (
    batch_schedule, bootstrap_xgb_models, compare_additions, feasibility_weighted_batch,
    fit_feasibility_classifier, fit_gp, gp_inputs, gp_select_batch, is_checkpoint, labels_to_match,
    model_free_batch, run_strategy, uncertainty_calibration,
)


def _fit_ridge(X, y):
    return Ridge().fit(X, y)


def _synthetic_campaign_data(n=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, d))
    y = 300 + 20 * X[:, 0] - 5 * X[:, 1] + rng.normal(scale=1.0, size=n)
    hard = X[:, 2] > 7
    return X, y, hard


def test_batch_schedule_uses_small_then_large_batches_and_exhausts_the_pool():
    sizes = batch_schedule(n_pool=1000, n_initial=50, small_batch=25, switch_at=550, large_batch=100)

    assert sizes[:20] == [25] * 20
    assert set(sizes[20:-1]) == {100}
    assert 50 + sum(sizes) == 1000


def test_is_checkpoint_covers_early_counts_every_later_round_and_the_full_pool():
    assert is_checkpoint(100, n_pool=1000)
    assert not is_checkpoint(125, n_pool=1000)
    assert is_checkpoint(650, n_pool=1000)
    assert is_checkpoint(1000, n_pool=1000)


def test_random_batch_is_unique_and_the_requested_size():
    picks = model_free_batch("random", np.random.default_rng(0), np.zeros(30, dtype=bool), 10)

    assert len(picks) == 10
    assert len(set(picks)) == 10


def test_heuristic_picks_only_from_the_hard_region_until_it_runs_out():
    hard = np.array([True] * 5 + [False] * 20)

    enough = model_free_batch("heuristic", np.random.default_rng(0), hard, 4)
    too_few = model_free_batch("heuristic", np.random.default_rng(0), hard, 8)

    assert all(hard[i] for i in enough)
    assert set(range(5)) <= set(too_few)
    assert len(set(too_few)) == 8


def test_gp_batch_spreads_across_uncertain_regions_instead_of_repeating_one():
    # Known data near the origin; two far clusters of identical candidates.
    # Plain top-k by uncertainty could take both picks from one cluster;
    # conditioning after each pick should send the second pick to the other.
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0, 1, size=(30, 2))
    y_train = X_train.sum(axis=1)
    X_cand = np.array([[5.0, 0.0]] * 3 + [[0.0, 5.0]] * 3 + [[0.5, 0.5]])
    bounds = np.array([[0.0, 0.0], [6.0, 6.0]])
    model = fit_gp(X_train, y_train, bounds)

    picks = gp_select_batch(model, X_cand, 2)

    assert len(set(picks)) == 2
    clusters = {tuple(X_cand[i]) for i in picks}
    assert clusters == {(5.0, 0.0), (0.0, 5.0)}


def _select_by_explicit_conditioning(model, X_cand, batch_size):
    """Reference implementation: refit-free BoTorch conditioning after each pick."""
    X_t = torch.as_tensor(X_cand, dtype=torch.double)
    picks, current = [], model
    for _ in range(batch_size):
        with torch.no_grad():
            var = current.posterior(X_t).variance.squeeze(-1)
        var[picks] = -torch.inf
        j = int(torch.argmax(var))
        picks.append(j)
        with torch.no_grad():
            placeholder = current.posterior(X_t[j:j + 1]).mean
        current = current.condition_on_observations(X_t[j:j + 1], placeholder)
    return picks


def test_gp_batch_matches_explicitly_conditioning_the_gp_after_each_pick():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 20, size=(260, 4))
    y = 300 + 5 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=2.0, size=len(X))
    X_gp = gp_inputs(X)
    model = fit_gp(X_gp[:60], y[:60], np.stack([X_gp.min(axis=0), X_gp.max(axis=0)]))

    fast = gp_select_batch(model, X_gp[60:], 8)
    reference = _select_by_explicit_conditioning(model, X_gp[60:], 8)

    assert fast == reference


def test_fit_gp_can_reuse_hyperparameters_without_optimising():
    X, y, _ = _synthetic_campaign_data()
    X_gp = gp_inputs(X)
    bounds = np.stack([X_gp.min(axis=0), X_gp.max(axis=0)])
    first = fit_gp(X_gp[:80], y[:80], bounds)

    reused = fit_gp(X_gp[:120], y[:120], bounds, hyperparameters_from=first, optimise=False)

    torch.testing.assert_close(reused.covar_module.state_dict(), first.covar_module.state_dict())
    torch.testing.assert_close(reused.likelihood.noise, first.likelihood.noise)
    assert reused.train_inputs[0].shape[0] == 120


def test_gp_uncertainty_samples_a_sparse_far_cluster_before_random_does():
    rng = np.random.default_rng(1)
    # Non-negative, like the real size/count features (gp_inputs takes log1p).
    dense = rng.uniform(0, 2, size=(180, 2))
    far = rng.uniform(8, 9, size=(20, 2))
    X = np.vstack([dense, far])
    y = 300 + 10 * X[:, 0] + rng.normal(scale=0.5, size=len(X))
    is_far = np.r_[np.zeros(180, bool), np.ones(20, bool)]
    test_idx = np.arange(0, 200, 5)
    pool_idx = np.setdiff1d(np.arange(200), test_idx)

    far_labelled = {}
    for strategy in ("gp_uncertainty", "random"):
        results = run_strategy(X, y, np.zeros(200, bool), test_idx, pool_idx, strategy, seed=0,
                               n_initial=20, small_batch=10, switch_at=40, large_batch=10,
                               evaluate=_fit_ridge)
        first_40 = results.attrs["labelled_order"][:40]
        far_labelled[strategy] = is_far[first_40].sum()

    assert far_labelled["gp_uncertainty"] > far_labelled["random"]


@pytest.mark.parametrize("strategy", ["random", "heuristic", "gp_uncertainty", "bootstrap_xgb"])
def test_run_strategy_labels_every_pool_row_exactly_once_and_never_a_test_row(strategy):
    X, y, hard = _synthetic_campaign_data()
    test_idx = np.arange(0, 200, 4)
    pool_idx = np.setdiff1d(np.arange(200), test_idx)

    results = run_strategy(X, y, hard, test_idx, pool_idx, strategy, seed=0,
                           n_initial=20, small_batch=20, switch_at=100, large_batch=40,
                           evaluate=_fit_ridge)
    order = results.attrs["labelled_order"]

    assert sorted(order) == sorted(pool_idx)
    assert not set(order) & set(test_idx)


def test_run_strategy_evaluates_only_at_checkpoints():
    X, y, hard = _synthetic_campaign_data()
    test_idx = np.arange(0, 200, 4)
    pool_idx = np.setdiff1d(np.arange(200), test_idx)
    calls = []

    def counting_fit(X_lab, y_lab):
        calls.append(len(X_lab))
        return _fit_ridge(X_lab, y_lab)

    results = run_strategy(X, y, hard, test_idx, pool_idx, "random", seed=0,
                           n_initial=50, small_batch=25, switch_at=125, large_batch=10,
                           evaluate=counting_fit)

    # 75 isn't a checkpoint; 50/100 are early checkpoints, then every round from 125.
    assert calls == [50, 100, 125, 135, 145, 150]
    assert list(results["n_labels"]) == calls


def test_run_strategy_is_reproducible_and_shares_the_initial_labels_across_strategies():
    X, y, hard = _synthetic_campaign_data()
    test_idx = np.arange(0, 200, 4)
    pool_idx = np.setdiff1d(np.arange(200), test_idx)
    kwargs = dict(n_initial=20, small_batch=20, switch_at=100, large_batch=40, evaluate=_fit_ridge)

    a = run_strategy(X, y, hard, test_idx, pool_idx, "random", seed=3, **kwargs)
    b = run_strategy(X, y, hard, test_idx, pool_idx, "random", seed=3, **kwargs)
    c = run_strategy(X, y, hard, test_idx, pool_idx, "heuristic", seed=3, **kwargs)

    pd.testing.assert_frame_equal(a, b)
    assert a.attrs["labelled_order"] == b.attrs["labelled_order"]
    assert a.attrs["labelled_order"][:20] == c.attrs["labelled_order"][:20]


def test_labels_to_match_interpolates_where_a_strategy_reaches_the_reference_rmse():
    results = pd.DataFrame({
        "strategy": ["random"] * 3 + ["fast"] * 3,
        "n_labels": [100, 200, 300] * 2,
        "rmse": [40.0, 35.0, 30.0, 36.0, 30.0, 28.0],
    })

    table = labels_to_match(results, budgets=(200, 300))

    assert table.loc[200, "random"] == 200
    assert table.loc[200, "fast"] == pytest.approx(100 + (36 - 35) / (36 - 30) * 100)
    assert table.loc[300, "fast"] == 200


def test_gp_inputs_log_transforms_nonnegative_features():
    np.testing.assert_allclose(gp_inputs([[0.0, np.e - 1]]), [[0.0, 1.0]])


def test_uncertainty_calibration_reports_both_selectors_and_sees_distance():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.uniform(0, 2, size=(150, 2)), rng.uniform(6, 9, size=(30, 2))])
    y = 300 + 10 * X[:, 0] + rng.normal(scale=0.5, size=len(X))
    hard = np.r_[np.zeros(150, bool), np.ones(30, bool)]
    labelled = np.arange(0, 150, 2)          # only the dense region is labelled
    test_idx = np.r_[np.arange(1, 150, 6), np.arange(150, 180, 2)]

    table = uncertainty_calibration(X, y, hard, labelled, test_idx)

    assert set(table["selector"]) == {"gp_uncertainty", "bootstrap_xgb"}
    gp = table.set_index("selector").loc["gp_uncertainty"]
    # Unlabelled far cluster: the GP should be far less sure there.
    assert gp["rho_distance"] > 0.5
    assert gp["hard_over_easy"] > 1


def test_compare_additions_detects_an_addition_that_fills_a_gap():
    rng = np.random.default_rng(5)
    f = lambda X: 300 + 30 * X[:, 0] + 20 * np.sin(X[:, 1])
    X_base = rng.uniform(0, 3, size=(150, 2))           # base data covers only part of the range
    X_test = rng.uniform(0, 6, size=(80, 2))
    X_gap = rng.uniform(3, 6, size=(60, 2))             # the missing part
    X_same = rng.uniform(0, 3, size=(60, 2))            # more of what we already have
    hard_test = X_test[:, 0] > 3

    table = compare_additions(X_base, f(X_base), X_test, f(X_test), hard_test,
                              {"fills_gap": (X_gap, f(X_gap)), "more_of_same": (X_same, f(X_same))},
                              seeds=[0], n_boot=200)

    assert list(table.index) == ["base", "fills_gap", "more_of_same"]
    assert table.loc["fills_gap", "rmse_hard"] < table.loc["more_of_same", "rmse_hard"]
    low, high = table.loc["fills_gap", "rmse_hard_change_ci"]
    assert high < 0  # confidently better than base in the gap region


def test_feasibility_weighting_skips_uncertain_but_infeasible_candidates():
    rng = np.random.default_rng(6)
    X_lab = rng.uniform(0, 5, size=(120, 1))
    models = bootstrap_xgb_models(X_lab, 300 + 10 * X_lab[:, 0] + rng.normal(size=120), n_models=5)
    # Past lookups: small molecules (x < 12) were found, large ones never.
    X_seen = rng.uniform(0, 30, size=(400, 1))
    classifier = fit_feasibility_classifier(X_seen, X_seen[:, 0] < 12)
    # Both far from the labelled range (uncertain); only the first is feasible.
    X_cand = np.array([[10.0], [25.0]])

    assert feasibility_weighted_batch(models, classifier, X_cand, 1) == [0]
