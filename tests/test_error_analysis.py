import numpy as np
import pandas as pd
import pytest

from boiling_point import error_analysis as ea


def test_molecule_attributes_counts_and_small_polyol():
    a = ea.molecule_attributes(["OCCO", "CCCCO", "OCCCCCCCCCO", "CCCCCC"])
    assert a["heavy_atoms"].tolist() == [4, 5, 11, 6]
    assert a["hbd"].tolist() == [2, 1, 2, 0]
    assert a["n_oh"].tolist() == [2, 1, 2, 0]
    # ethylene glycol is a small polyol; the C9 diol is too big; butanol has one OH
    assert a["small_polyol"].tolist() == [True, False, False, False]


def test_assign_bins_edges():
    labels = ea.assign_bins([1, 6, 7, 8, 9, 10, 11, 12, 13, 40], "heavy_atoms").tolist()
    assert labels == ["≤ 6", "≤ 6", "7–8", "7–8", "9–10", "9–10", "11–12", "11–12", "≥ 13", "≥ 13"]
    assert ea.assign_bins([0, 1, 2, 5], "hbd").tolist() == ["0", "1", "≥ 2", "≥ 2"]


def test_assign_bins_rejects_values_outside_every_bin():
    with pytest.raises(ValueError):
        ea.assign_bins([-1], "hbd")


def test_group_bootstrap_constant_errors_give_a_degenerate_interval():
    err = pd.DataFrame({"a": np.full(30, 5.0), "b": np.full(30, 2.0)})
    out = ea.group_bootstrap_mae(err, np.arange(30) // 3, reference="b", n_boot=200).set_index("method")
    assert out.loc["a", "mae"] == 5.0 and out.loc["a", "ci_low"] == 5.0 and out.loc["a", "ci_high"] == 5.0
    assert out.loc["a", "diff"] == 3.0 and out.loc["a", "diff_ci_low"] == pytest.approx(3.0)
    assert out.loc["b", "diff"] == 0.0 and out.loc["b", "diff_ci_high"] == 0.0


def test_group_bootstrap_keeps_groups_whole():
    # group 0: 9 molecules with error 0; group 1: 1 molecule with error 10. Drawing 2 groups gives
    # MAE 0 (group 0 twice), 10 / 10 = 1 (one of each) or 10 (group 1 twice), so the interval spans
    # 0 to 10; a molecule-level bootstrap would give a narrow interval around 1
    err = pd.DataFrame({"a": [0.0] * 9 + [10.0]})
    out = ea.group_bootstrap_mae(err, [0] * 9 + [1], n_boot=500, seed=1)
    assert out.loc[0, "mae"] == 1.0
    assert out.loc[0, "ci_low"] == 0.0 and out.loc[0, "ci_high"] == 10.0


def test_group_bootstrap_interval_covers_the_estimate():
    rng = np.random.default_rng(0)
    err = pd.DataFrame({"a": rng.exponential(10, 400), "b": rng.exponential(12, 400)})
    out = ea.group_bootstrap_mae(err, rng.integers(0, 150, 400), reference="a").set_index("method")
    for m in ("a", "b"):
        assert out.loc[m, "ci_low"] < out.loc[m, "mae"] < out.loc[m, "ci_high"]
    assert out.loc["b", "diff_ci_low"] < out.loc["b", "diff"] < out.loc["b", "diff_ci_high"]
