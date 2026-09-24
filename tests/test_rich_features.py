import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from boiling_point.descriptors import CURATED_COLUMNS, DropCorrelated, curated_descriptors
from boiling_point.families import assign_family, merge_rare_families
from boiling_point.validation import (
    fit_final_model, fold_metrics, outer_splits, run_outer_fold, stratification_labels,
)


# --- descriptors ---

def test_wiener_index_is_smaller_for_the_branched_isomer():
    d = curated_descriptors(["CCCC", "CC(C)C"])  # n-butane, isobutane

    assert d["WienerIndex"].tolist() == [10.0, 9.0]


def test_element_counts_and_unsaturation_counts():
    d = curated_descriptors(["ClC(Cl)(Cl)Cl", "C=CC#C"]).iloc
    assert d[0]["n_Cl"] == 4 and d[0]["n_C"] == 1
    assert d[1]["n_double_CC"] == 1 and d[1]["n_triple_CC"] == 1


def test_curated_descriptors_have_every_column_and_no_nans():
    d = curated_descriptors(["CCO", "CCN(C)C", "CCCCCCCC", "OCCOCCO"])

    assert list(d.columns) == CURATED_COLUMNS
    assert not d.isna().any().any()


def test_drop_correlated_keeps_one_of_each_near_duplicate_pair():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = np.c_[a, 2 * a + 0.001 * rng.normal(size=200), rng.normal(size=200)]

    kept = DropCorrelated(0.95).fit(X).keep_

    assert kept.tolist() == [0, 2]
    assert DropCorrelated().fit(X[:, :1]).transform(X[:, :1]).shape == (200, 1)


# --- families ---

@pytest.mark.parametrize("smiles, family", [
    ("CCO", "alcohol/phenol"),
    ("CC(=O)O", "carboxylic acid"),          # acid outranks the OH it contains
    ("CCOC(C)=O", "ester"),
    ("CCCCCC", "alkane"),
    ("CCN", "amine"),
    ("C=CC", "alkene/alkyne"),
    ("OCCOCCO", "alcohol/phenol"),           # glycol ether: OH outranks ether
    ("B(CC)(CC)CC", "other"),
])
def test_assign_family(smiles, family):
    assert assign_family(smiles) == family


def test_merge_rare_families_and_stratification_labels():
    fams = pd.Series(["a"] * 30 + ["b"] * 10 + ["c"] * 3)

    assert merge_rare_families(fams, min_count=25).value_counts().to_dict() == {"a": 30, "other": 13}
    labels = stratification_labels(fams, n_splits=5)
    assert set(labels) == {"a", "b"}  # 'c' (3 < 5 folds) is split with the largest family


# --- validation ---

def _synthetic(n=150, seed=0):
    rng = np.random.default_rng(seed)
    families = np.array(["x"] * (n // 2) + ["y"] * (n - n // 2))
    X = rng.uniform(0, 5, size=(n, 3))
    y = 300 + 20 * X[:, 0] + 30 * (families == "y") + rng.normal(scale=2, size=n)
    return X, y, families


def test_outer_splits_predict_every_row_once_per_repeat_with_family_balance():
    _, _, families = _synthetic()
    splits = outer_splits(families, n_splits=5, n_repeats=2)

    for repeat in (0, 1):
        tested = np.concatenate([te for r, _, _, te in splits if r == repeat])
        assert sorted(tested) == list(range(len(families)))
    for _, _, _, te in splits:
        assert abs((families[te] == "x").mean() - 0.5) < 0.05


def test_run_outer_fold_never_tunes_on_outer_test_rows(monkeypatch):
    import boiling_point.validation as v
    X, y, families = _synthetic()
    _, _, train_idx, test_idx = outer_splits(families, n_splits=5, n_repeats=1)[0]
    seen_sizes = []
    original = v._tune

    def spy(model, X_fit, y_fit, strat, n_inner, n_iter, random_state):
        seen_sizes.append(len(X_fit))
        return original(model, X_fit, y_fit, strat, n_inner, n_iter, random_state)

    monkeypatch.setattr(v, "_tune", spy)
    preds, rows = run_outer_fold(X, y, families, families, train_idx, test_idx, "toy", 0, 0,
                                 n_inner=2, n_iter=2)

    assert max(seen_sizes) == len(train_idx)
    assert set(preds["row"]) == set(test_idx)
    assert set(preds["model"]) == {"ridge", "xgboost", "mlp", "ensemble"}
    ens = preds[preds["model"] == "ensemble"].set_index("row")["y_pred"]
    members = preds[preds["model"] != "ensemble"].groupby("row")["y_pred"].mean()
    np.testing.assert_allclose(ens.sort_index(), members.sort_index())


def test_per_family_models_only_for_large_families_and_reproducible():
    X, y, families = _synthetic()
    _, _, train_idx, test_idx = outer_splits(families, n_splits=5, n_repeats=1)[0]
    kwargs = dict(n_inner=2, n_iter=2, per_family_min=50)

    a, _ = run_outer_fold(X, y, families, families, train_idx, test_idx, "toy", 0, 0, **kwargs)
    b, _ = run_outer_fold(X, y, families, families, train_idx, test_idx, "toy", 0, 0, **kwargs)

    per_family = a[a["model"] == "xgboost_per_family"]
    assert set(per_family["family"]) == {"x", "y"}
    pd.testing.assert_frame_equal(a, b)


def test_fold_metrics_and_final_model_use_all_rows():
    X, y, families = _synthetic()
    preds = pd.DataFrame({"feature_set": "toy", "model": "m", "repeat": 0, "fold": 0,
                          "row": np.arange(4), "y_true": [1.0, 2, 3, 4], "y_pred": [1.0, 2, 3, 6]})
    m = fold_metrics(preds, hard_mask=np.array([False, False, True, True])).iloc[0]
    assert m["rmse"] == pytest.approx(1.0) and m["rmse_hard"] == pytest.approx(np.sqrt(2))

    model, params = fit_final_model("ensemble", X, y, families, n_inner=2, n_iter=2)
    assert set(params) == {"ridge", "xgboost", "mlp"}
    assert model.predict(X).shape == (len(y),)
