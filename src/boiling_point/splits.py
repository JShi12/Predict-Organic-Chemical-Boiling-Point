"""Fixed evaluation splits for the representation study.

Three regimes, each a different question:
- familiar chemistry: repeated K-fold, stratified by family and grouped by
  structure, so stereoisomers of one structure never sit on both sides;
- unseen functional group: leave one family out;
- learning curves: nested, family-stratified, structure-grouped subsets of
  each familiar-chemistry training fold, shared by every method.

The splits are generated once (scripts/build_representation_splits.py),
saved, and fixed before any model is compared.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedGroupKFold

HOLDOUT_FAMILIES = ("alkane", "alkene/alkyne", "alcohol/phenol", "amine")
HALOGENS = {"F", "Cl", "Br", "I"}


def inchikey_connectivity(smiles: str) -> str:
    """First block of the InChIKey: the connectivity layer, which ignores
    stereochemistry."""
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)).split("-")[0]


def group_ids(keys) -> np.ndarray:
    """Integer id per row, shared by rows with the same key (order of first appearance)."""
    return pd.factorize(pd.Series(keys))[0]


def familiar_splits(strat, groups, n_splits: int = 5, n_repeats: int = 3, random_state: int = 0) -> np.ndarray:
    """(n_rows, n_repeats) array: the test fold of each row in each repeat.
    StratifiedGroupKFold keeps every group in one fold while balancing the
    stratification labels; each repeat reshuffles with its own seed."""
    strat, groups = np.asarray(strat), np.asarray(groups)
    folds = np.full((len(strat), n_repeats), -1, dtype=int)
    for r in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state + r)
        for k, (_, test) in enumerate(splitter.split(np.zeros(len(strat)), strat, groups)):
            folds[test, r] = k
    return folds


def holdout_family(families) -> np.ndarray:
    """The family each row is held out with in leave-one-family-out, or ''
    for rows outside the four held-out families (they're left out of that
    regime)."""
    fam = pd.Series(families).to_numpy()
    return np.where(np.isin(fam, HOLDOUT_FAMILIES), fam, "")


def proportional_quotas(n: int, counts: pd.Series) -> pd.Series:
    """Split n across strata in proportion to counts (largest-remainder method)."""
    exact = n * counts / counts.sum()
    quota = np.floor(exact).astype(int)
    shortfall = n - quota.sum()
    quota[(exact - quota).sort_values(ascending=False).index[:shortfall]] += 1
    return quota


def nested_subset_entry(families, groups, sizes, rng: np.random.Generator) -> np.ndarray:
    """Entry size of each row in nested, family-stratified, group-preserving
    training subsets: a row belongs to the subset of size n if its entry is
    <= n. Rows outside the largest requested size get len(rows) (the full
    training fold).

    Within each family, whole groups are taken in one random order until that
    family's quota is met. Because the order is fixed and quotas never shrink
    as n grows, every subset contains the smaller ones. Keeping groups whole
    can overshoot a quota by a few molecules, so realised sizes are slightly
    above n."""
    fam = pd.Series(families).reset_index(drop=True)
    grp = pd.Series(groups).reset_index(drop=True)
    entry = np.full(len(fam), len(fam), dtype=int)
    counts = fam.value_counts()
    order = {f: rng.permutation(grp[fam == f].unique()) for f in counts.index}
    previous = pd.Series(0, index=counts.index)
    for n in sorted(sizes):
        quota = np.maximum(proportional_quotas(n, counts), previous)
        for f in counts.index:
            taken = 0
            for g in order[f]:
                if taken >= quota[f]:
                    break
                rows = np.flatnonzero((grp == g).to_numpy())
                entry[rows] = np.minimum(entry[rows], n)
                # A group can span families (methane is an alkane, deuterated methane "other");
                # only this family's rows count towards its quota
                taken += int((fam.iloc[rows] == f).sum())
        previous = quota
    return entry


def nist_subgroup(smiles: str) -> str:
    """'ring-containing' if any ring, else 'halogenated acyclic' if any
    halogen, else 'other'."""
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetRingInfo().NumRings() > 0:
        return "ring-containing"
    if any(a.GetSymbol() in HALOGENS for a in mol.GetAtoms()):
        return "halogenated acyclic"
    return "other"
