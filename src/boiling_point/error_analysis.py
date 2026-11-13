"""Exploratory error analysis for the representation study (Step 7).

- molecule_attributes: per-molecule structural attributes used to slice the
  errors (heavy atoms, H-bond donors/acceptors, hydroxyl count).
- BINS / assign_bins: the fixed bin edges for those attributes.
- group_bootstrap_mae: MAE per method with 95% percentile intervals, and the
  paired difference to a reference method, resampling whole structure groups
  (the unit of independence: stereoisomers share a group).
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Fragments, Lipinski

# (label, lower, upper) inclusive integer ranges
BINS = {
    "heavy_atoms": [("≤ 6", 0, 6), ("7–8", 7, 8), ("9–10", 9, 10), ("11–12", 11, 12), ("≥ 13", 13, 10 ** 6)],
    "hbd": [("0", 0, 0), ("1", 1, 1), ("≥ 2", 2, 10 ** 6)],
    "hba": [("0", 0, 0), ("1", 1, 1), ("≥ 2", 2, 10 ** 6)],
}
SMALL_POLYOL_MAX_HEAVY = 8


def molecule_attributes(smiles) -> pd.DataFrame:
    """Heavy atoms, Lipinski H-bond donors and acceptors, hydroxyl groups
    (aliphatic + aromatic), and small polyol (>= 2 OH, <= 8 heavy atoms)."""
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        n_oh = Fragments.fr_Al_OH(mol) + Fragments.fr_Ar_OH(mol)
        heavy = mol.GetNumHeavyAtoms()
        rows.append({"heavy_atoms": heavy, "hbd": Lipinski.NumHDonors(mol), "hba": Lipinski.NumHAcceptors(mol),
                     "n_oh": n_oh, "small_polyol": bool(n_oh >= 2 and heavy <= SMALL_POLYOL_MAX_HEAVY)})
    return pd.DataFrame(rows)


def assign_bins(values, attribute: str) -> pd.Series:
    """Bin label for each integer value, from BINS[attribute]."""
    values = pd.Series(values).reset_index(drop=True)
    out = pd.Series(pd.NA, index=values.index, dtype=object)
    for label, lo, hi in BINS[attribute]:
        out[(values >= lo) & (values <= hi)] = label
    if out.isna().any():
        raise ValueError(f"{attribute}: values outside every bin: {sorted(values[out.isna()].unique())}")
    return out


def group_bootstrap_mae(abs_err: pd.DataFrame, groups, reference: str = None, n_boot: int = 2000,
                        seed: int = 0) -> pd.DataFrame:
    """abs_err: (molecules, methods) absolute errors; groups: one label per
    molecule. Each draw resamples groups with replacement, keeping their
    molecules together, and recomputes every method's MAE on the same draw.

    Returns one row per method: n, mae, ci_low, ci_high, and, if `reference`
    is given, diff = MAE(method) - MAE(reference) with its interval."""
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)
    codes, uniques = pd.factorize(groups)
    g = len(uniques)
    err = abs_err.to_numpy(dtype=float)
    sums = np.zeros((g, err.shape[1]))
    np.add.at(sums, codes, err)
    counts = np.bincount(codes, minlength=g).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, g, size=(n_boot, g))
    weights = np.zeros((n_boot, g))
    np.add.at(weights, (np.repeat(np.arange(n_boot), g), draws.ravel()), 1.0)
    boot = (weights @ sums) / (weights @ counts)[:, None]           # (n_boot, methods)
    mae = err.mean(axis=0)
    rows = []
    for j, m in enumerate(abs_err.columns):
        row = {"method": m, "n": len(err), "n_groups": g, "mae": mae[j],
               "ci_low": np.percentile(boot[:, j], 2.5), "ci_high": np.percentile(boot[:, j], 97.5)}
        if reference is not None:
            r = list(abs_err.columns).index(reference)
            d = boot[:, j] - boot[:, r]
            row.update({"diff": mae[j] - mae[r], "diff_ci_low": np.percentile(d, 2.5),
                        "diff_ci_high": np.percentile(d, 97.5)})
        rows.append(row)
    return pd.DataFrame(rows)
