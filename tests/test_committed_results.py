"""Integrity of the committed representation-study predictions: every method
in a task was tested on exactly the same molecules (Step 9)."""
from pathlib import Path

import pandas as pd
import pytest

R = Path("results/representation_study")


def _key_sets(df, task_cols, method_cols):
    """{task tuple: {method tuple: frozenset of keys}}."""
    out = {}
    for keys, g in df.groupby(task_cols + method_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        out.setdefault(keys[:len(task_cols)], {})[keys[len(task_cols):]] = frozenset(g["key"])
    return out


def _assert_identical(sets):
    for task, by_method in sets.items():
        distinct = set(by_method.values())
        assert len(distinct) == 1, f"{task}: methods were tested on different molecules"


@pytest.fixture(scope="module")
def frozen():
    if not (R / "frozen/ridge_predictions.csv").exists():
        pytest.skip("committed results not present")
    df = pd.concat([pd.read_csv(R / "frozen/ridge_predictions.csv"), pd.read_csv(R / "frozen/xgboost_predictions.csv")])
    return df


@pytest.fixture(scope="module")
def frozen_tests(frozen):
    """{(regime, task): keys} from the frozen comparison, the reference test sets."""
    sets = _key_sets(frozen, ["regime", "task"], ["representation", "regressor"])
    _assert_identical(sets)
    return {task: next(iter(m.values())) for task, m in sets.items()}


def test_frozen_methods_share_test_sets_and_cover_the_splits(frozen_tests):
    splits = pd.read_csv(R / "splits.csv")
    familiar = {t: k for (g, t), k in frozen_tests.items() if g == "familiar"}
    assert len(familiar) == 15
    for r in range(3):          # each repeat's 5 test folds partition the 1,250 molecules
        keys = [familiar[f"r{r}f{k}"] for k in range(5)]
        assert sum(len(k) for k in keys) == len(splits) and len(frozenset().union(*keys)) == len(splits)


def test_best_and_concat_predictions_use_the_same_test_sets(frozen_tests):
    for name in ("frozen/best_predictions.csv", "combined/concat_ridge_predictions.csv"):
        sets = _key_sets(pd.read_csv(R / name), ["regime", "task"], ["representation"])
        for task, by_method in sets.items():
            for keys in by_method.values():
                assert keys == frozen_tests[task], f"{name} {task}"


def test_stacked_predictions_use_the_same_test_sets(frozen_tests):
    sets = _key_sets(pd.read_csv(R / "combined/stacked_predictions.csv"), ["regime", "task"], ["method"])
    for task, by_method in sets.items():
        for keys in by_method.values():
            assert keys == frozen_tests[task]


def test_learning_curve_methods_share_the_repeat_0_test_folds(frozen_tests):
    lc = pd.read_csv(R / "learning_curve/predictions.csv")
    sets = _key_sets(lc, ["task", "fold"], ["representation", "regressor"])
    _assert_identical(sets)
    for (task, fold), by_method in sets.items():
        assert next(iter(by_method.values())) == frozen_tests[("familiar", f"r0f{fold}")], task


def test_fine_tuned_runs_use_the_same_test_sets(frozen_tests):
    for name, method_cols in (("finetune/predictions.csv", ["seed", "inference"]),
                              ("finetune_readout/predictions.csv", ["variant", "seed", "inference"])):
        df = pd.read_csv(R / name)
        for (regime, task), by_method in _key_sets(df, ["regime", "task"], method_cols).items():
            ref = frozen_tests[("familiar", f"r0f{task[-1]}")] if regime == "learning_curve" else frozen_tests[(regime, task)]
            for keys in by_method.values():
                assert keys == ref, f"{name} {task}"
    hc = pd.read_csv(R / "finetune/head_control/predictions.csv")
    for (task,), by_method in _key_sets(hc, ["task"], ["method", "seed"]).items():
        for keys in by_method.values():
            assert keys == frozen_tests[("familiar", task)]
