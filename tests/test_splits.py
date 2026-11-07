import numpy as np
import pandas as pd
import pytest

from boiling_point import splits


def _toy(n_groups=120, seed=0):
    """Families of different sizes; some structures have 2-3 stereoisomer rows."""
    rng = np.random.default_rng(seed)
    fams = rng.choice(["alkane", "alkene/alkyne", "alcohol/phenol", "amine"], size=n_groups, p=[.35, .3, .25, .1])
    reps = rng.choice([1, 1, 1, 2, 3], size=n_groups)
    return pd.DataFrame({"family": np.repeat(fams, reps), "group": np.repeat(np.arange(n_groups), reps)})


def test_inchikey_connectivity_ignores_stereochemistry():
    keys = {splits.inchikey_connectivity(s) for s in ["C[C@@H](O)[C@@H](C)O", "C[C@H](O)[C@@H](C)O", "CC(O)C(C)O"]}
    assert len(keys) == 1
    assert splits.inchikey_connectivity("OC/C=C/CO") == splits.inchikey_connectivity("OC/C=C\\CO")


def test_familiar_splits_keep_groups_whole_and_use_every_fold():
    df = _toy()
    folds = splits.familiar_splits(df["family"], df["group"], n_splits=5, n_repeats=3)

    assert folds.shape == (len(df), 3) and (folds >= 0).all()
    for r in range(3):
        assert df.assign(f=folds[:, r]).groupby("group")["f"].nunique().max() == 1   # no group split
        assert set(folds[:, r]) == set(range(5))
    assert not (folds[:, 0] == folds[:, 1]).all()                                     # repeats differ


def test_familiar_splits_are_reproducible():
    df = _toy()
    a = splits.familiar_splits(df["family"], df["group"])
    b = splits.familiar_splits(df["family"], df["group"])
    assert (a == b).all()


def test_holdout_family_leaves_out_other_families():
    out = splits.holdout_family(["amine", "other", "alkane"])
    assert out.tolist() == ["amine", "", "alkane"]


def test_nested_subsets_are_nested_group_preserving_and_stratified():
    df = _toy(n_groups=400)
    sizes = [50, 100, 200]
    entry = splits.nested_subset_entry(df["family"], df["group"], sizes, np.random.default_rng(1))

    previous = set()
    for n in sizes:
        members = set(np.flatnonzero(entry <= n))
        assert previous <= members                                    # nested
        assert n <= len(members) <= n + 2 * 4                         # whole groups overshoot slightly
        sub = df.iloc[sorted(members)]
        assert sub.groupby("group").size().equals(df[df["group"].isin(sub["group"])].groupby("group").size())
        share = sub["family"].value_counts(normalize=True)
        full = df["family"].value_counts(normalize=True)
        assert (share - full).abs().max() < 0.05                      # proportions kept
        previous = members
    assert (entry[entry > max(sizes)] == len(df)).all()               # the rest only in the full fold


def test_nested_subsets_reach_the_target_size_when_a_group_spans_families():
    df = _toy(n_groups=200)
    df.loc[len(df)] = {"family": "other", "group": df["group"].iloc[0]}   # like methane + deuterated methane
    for seed in range(20):
        entry = splits.nested_subset_entry(df["family"], df["group"], [50, 100], np.random.default_rng(seed))
        assert (entry <= 50).sum() >= 50 and (entry <= 100).sum() >= 100


@pytest.mark.parametrize("smiles, expected", [
    ("c1ccccc1Cl", "ring-containing"),
    ("CCCCBr", "halogenated acyclic"),
    ("CCCCS", "other"),
])
def test_nist_subgroup(smiles, expected):
    assert splits.nist_subgroup(smiles) == expected
