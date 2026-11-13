import numpy as np
import pandas as pd
import pytest

from boiling_point import study, study_stats


def _splits(n_groups=150, seed=0):
    """A small splits.csv-like table: families, structure groups, folds."""
    from boiling_point import splits as sp
    rng = np.random.default_rng(seed)
    fams = rng.choice(["alkane", "alkene/alkyne", "alcohol/phenol", "amine"], size=n_groups, p=[.3, .3, .3, .1])
    reps = rng.choice([1, 1, 2], size=n_groups)
    df = pd.DataFrame({"family": np.repeat(fams, reps), "structure_group": np.repeat(np.arange(n_groups), reps)})
    folds = sp.familiar_splits(df["family"], df["structure_group"], 5, 3)
    for r in range(3):
        df[f"fold_r{r}"] = folds[:, r]
    df["holdout_family"] = sp.holdout_family(df["family"])
    return df


def test_familiar_inner_folds_stay_inside_training_and_keep_groups_whole():
    df = _splits()
    fam, grp = df["family"].to_numpy(), df["structure_group"].to_numpy()
    for task in study.familiar_tasks(df):
        assert not set(task["train"]) & set(task["test"])
        folds = study.inner_folds(task, fam, grp)
        assert len(folds) == study.N_INNER
        for tr, va in folds:
            outer_tr, outer_va = task["train"][tr], task["train"][va]
            assert set(outer_va) <= set(task["train"])              # no outer-test row in tuning
            assert not set(grp[outer_tr]) & set(grp[outer_va])      # groups never straddle inner folds


def test_holdout_inner_folds_are_the_three_training_families():
    df = _splits()
    fam, grp = df["family"].to_numpy(), df["structure_group"].to_numpy()
    for task in study.holdout_tasks(df):
        held = task["task"]
        assert set(fam[task["test"]]) == {held} and held not in set(fam[task["train"]])
        folds = study.inner_folds(task, fam, grp)
        assert sorted(set(fam[task["train"][va]]).pop() for _, va in folds) == sorted(set(fam[task["train"]]))
        assert all(len(set(fam[task["train"][va]])) == 1 for _, va in folds)


def test_nist_task_trains_on_literature_and_tests_on_nist_rows():
    t = study.nist_task(10, np.arange(3))
    assert t["train"].tolist() == list(range(10)) and t["test"].tolist() == [10, 11, 12]


def test_ridge_pipeline_drops_near_duplicates_only_for_descriptors():
    assert "correlated" in study.ridge_pipeline("curated").regressor.named_steps
    assert "correlated" not in study.ridge_pipeline("molformer").regressor.named_steps


def test_xgb_grid_covers_every_combination():
    grid = study.xgb_grid({"max_depth": [3, 5], "learning_rate": [0.1, 0.01]})
    assert len(grid) == 4 and {"model__max_depth": 5, "model__learning_rate": 0.01} in grid


def test_run_task_ridge_picks_the_informative_layer():
    rng = np.random.default_rng(0)
    df = _splits()
    n = len(df)
    y = rng.normal(size=n) * 50 + 400
    X = rng.normal(size=(3, n, 5))
    X[1, :, 0] = y + rng.normal(size=n)                            # only layer 1 carries the target
    task = study.familiar_tasks(df)[0]
    out = study.run_task(task, "molformer", "ridge", X, y, df["family"].to_numpy(),
                         df["structure_group"].to_numpy(), layer_probe=True)
    assert out["layer"] == 1
    assert len(out["probe"]) == 3 and len(out["inner_test"]) == study.N_INNER
    assert np.mean(np.abs(out["pred"] - out["y_true"])) < 5


def test_corrected_ttest_is_wider_than_the_naive_one():
    d = np.array([-1.2, -0.8, -1.0, -1.5, -0.6, -1.1, -0.9, -1.3, -0.7, -1.0])
    corrected = study_stats.corrected_resampled_ttest(d, n_train=1000, n_test=250)
    naive_se = d.std(ddof=1) / np.sqrt(len(d))
    assert corrected["estimate"] == pytest.approx(d.mean())
    assert corrected["se"] == pytest.approx(np.sqrt((1 / 10 + 250 / 1000) * d.var(ddof=1)))
    assert corrected["se"] > naive_se and corrected["df"] == 9


def test_bootstrap_macro_difference_weights_strata_equally_and_is_null_for_identical_errors():
    a = np.array([1.0] * 90 + [5.0] * 10)
    b = np.array([1.0] * 90 + [1.0] * 10)
    strata = np.array(["big"] * 90 + ["small"] * 10)
    r = study_stats.bootstrap_macro_difference(a, b, strata, n_boot=2000)
    assert r["estimate"] == pytest.approx((0 + 4) / 2)              # not the pooled 0.4
    same = study_stats.bootstrap_macro_difference(a, a, strata, n_boot=500)
    assert same["estimate"] == 0 and same["p"] == 1.0


def test_holm_adjusts_in_rank_order_and_stays_monotone():
    adjusted = study_stats.holm([0.01, 0.04, 0.03, 0.2, 0.001])
    assert adjusted.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.2, 0.005])


def test_a_finding_needs_both_holm_and_the_minimum_difference():
    assert study_stats.finding(0.01, -1.5)
    assert not study_stats.finding(0.01, -0.5)
    assert not study_stats.finding(0.2, -3.0)


def test_learning_curve_tasks_are_nested_inside_training_and_share_test_folds():
    import os
    path = "results/representation_study/splits.csv"
    if not os.path.exists(path):
        pytest.skip("splits.csv not built")
    df = pd.read_csv(path)
    tasks = study.learning_curve_tasks(df)
    assert len(tasks) == 5 * 3 * 3
    grp = df["structure_group"].to_numpy()
    by = {(t["fold"], t["draw"], t["size"]): t for t in tasks}
    for (k, d, n), t in by.items():
        fold_train = np.flatnonzero(df["fold_r0"].to_numpy() != k)
        assert set(t["train"]) <= set(fold_train)
        assert np.array_equal(t["test"], np.flatnonzero(df["fold_r0"].to_numpy() == k))
        assert n <= len(t["train"]) <= n + 5                        # whole groups can overshoot slightly
        assert not set(grp[t["train"]]) & set(grp[t["test"]])
        for smaller in (100, 250):
            if smaller < n:
                assert set(by[(k, d, smaller)]["train"]) <= set(t["train"])
        fam = study.inner_folds(t, df["family"].to_numpy(), grp)
        assert len(fam) == study.N_INNER


def test_confirmatory_table_bounds_bootstrap_p_and_applies_holm_and_the_size_rule():
    t = pd.DataFrame({"comparison": ["C1", "C2", "C3"], "estimate": [0.4, 9.7, 0.5],
                      "p_raw": [0.5, 0.0, 0.001], "test": ["corrected t", "bootstrap", "corrected t"]})
    out = study_stats.confirmatory_table(t, n_boot=10_000)
    assert out["p_used"].tolist() == [0.5, 0.0002, 0.001]
    assert out["p_holm"].tolist() == pytest.approx([0.5, 0.0006, 0.002])
    assert out["finding"].tolist() == [False, True, False]     # C3 significant but below 1.0 K
