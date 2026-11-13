import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from boiling_point import stacking, study


@pytest.mark.parametrize("loss", ["squared", "absolute"])
def test_convex_weights_recover_a_known_blend(loss):
    rng = np.random.default_rng(0)
    P = rng.normal(400, 50, size=(300, 3))
    y = P @ np.array([0.6, 0.4, 0.0])
    w = stacking.convex_weights(P, y, loss)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
    assert w == pytest.approx([0.6, 0.4, 0.0], abs=1e-4)


@pytest.mark.parametrize("loss", ["squared", "absolute"])
def test_convex_weights_put_everything_on_a_perfect_input(loss):
    rng = np.random.default_rng(1)
    y = rng.normal(400, 50, 200)
    P = np.column_stack([y + rng.normal(0, 20, 200), y, y + rng.normal(0, 5, 200)])
    assert stacking.convex_weights(P, y, loss) == pytest.approx([0, 1, 0], abs=1e-4)


def test_convex_weights_stay_convex_when_the_unconstrained_fit_would_not():
    rng = np.random.default_rng(2)
    y = rng.normal(400, 50, 200)
    P = np.column_stack([y * 0.5, y * 0.4])          # least squares alone would put weights > 1
    w = stacking.convex_weights(P, y)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)


def test_block_scaler_gives_each_block_equal_total_variance():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(0, 5, (100, 4)), rng.normal(3, 0.1, (100, 16)), np.ones(100)])
    Z = stacking.BlockScaler((4, 17)).fit(X).transform(X)
    assert Z[:, :4].var(axis=0).sum() == pytest.approx(1.0)
    assert Z[:, 4:].var(axis=0).sum() == pytest.approx(1.0)     # the constant column doesn't count
    assert np.all(Z[:, -1] == 0)
    with pytest.raises(ValueError):
        stacking.BlockScaler((4, 4)).fit(X)
    clone(stacking.BlockScaler((4, 17)))                          # a valid sklearn estimator


def test_oof_matrix_aligns_keys_across_pipelines():
    res = pd.DataFrame({"key": ["b", "a", "a", "b"], "pipeline": ["p", "p", "q", "q"],
                        "residual": [1.0, 2.0, -1.0, 0.5]})
    y = pd.Series({"a": 10.0, "b": 20.0, "c": 30.0})
    oof = stacking.oof_matrix(res, y, ["q", "p"])
    assert list(oof.columns) == ["q", "p"] and list(oof.index) == ["a", "b"]
    assert oof.loc["a"].tolist() == [11.0, 8.0] and oof.loc["b"].tolist() == [19.5, 19.0]
    with pytest.raises(ValueError):
        stacking.oof_matrix(res.iloc[:3], y, ["p", "q"])


def test_concat_ridge_runs_through_the_study_engine():
    rng = np.random.default_rng(4)
    n = 120
    desc = rng.normal(size=(n, 3))
    emb = rng.normal(size=(2, n, 10))                             # 2 layers
    X = np.stack([np.hstack([desc, emb[l]]) for l in range(2)])
    y = 300 + 20 * desc[:, 0] + rng.normal(0, 1, n)
    fam = np.array(["a", "b"] * (n // 2), dtype=object)
    task = {"regime": "familiar", "task": "t", "repeat": 0, "fold": 0,
            "train": np.arange(90), "test": np.arange(90, n)}
    r = study.run_task(task, "concat", "ridge", X, y, fam, np.arange(n), alphas=[0.1, 10], blocks=(3, 10))
    assert r["layer"] in (0, 1)
    assert np.mean(np.abs(r["pred"] - y[90:])) < 5
