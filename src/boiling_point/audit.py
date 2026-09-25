"""Label audit: which boiling points are estimates or wrong?

Three checks, none of which uses a model's predictions to decide:
- Joback estimates: Joback & Reid's group-contribution method predicts
  Tb = 198.2 K + a fixed amount per structural group. A label that equals
  this sum to the hundredth of a kelvin was almost certainly computed, not
  measured. (The giant alkanes in the literature data match exactly.)
- Implausible hydrocarbons: an acyclic hydrocarbon boils close to the
  straight-chain alkane with the same carbon count (heavy branching lowers
  it only ~20-30 K), so a label far below that is implausible.
- Independent sources: NIST WebBook values for compounds that every model
  gets wrong by a large margin in the same direction (consistent
  disagreement only *nominates* compounds for checking; the evidence is the
  independent value).

Only the acyclic groups that occur in this dataset (C, H, N, O) are
implemented; joback_tb returns None for anything else (rings, halogens,
carbonyls, boron...).
"""
import numpy as np
import pandas as pd
from rdkit import Chem

JOBACK_INTERCEPT = 198.2
# (name, SMARTS, Tb contribution in K) -- Joback & Reid (1987), non-ring groups
JOBACK_GROUPS = [
    ("-CH3", "[CX4H3]", 23.58),
    ("-CH2-", "[CX4H2;!R]", 22.88),
    (">CH-", "[CX4H1;!R]", 21.74),
    (">C<", "[CX4H0;!R]", 18.25),
    ("=CH2", "[CX3H2]=[C]", 18.18),
    ("=CH-", "[CX3H1;!R]=[C]", 24.96),
    ("=C<", "[CX3H0;!R]=[C]", 24.14),
    ("=C=", "[CX2H0](=[C])=[C]", 26.15),
    ("#CH", "[CX2H1]#[C]", 9.20),
    ("#C-", "[CX2H0]#[C]", 27.38),
    ("-OH (alcohol)", "[OX2H][CX4]", 92.88),
    ("-O- (non-ring)", "[OX2H0;!R]([#6])[#6]", 22.42),
    ("-NH2", "[NX3H2][#6]", 73.23),
    (">NH (non-ring)", "[NX3H1;!R]([#6])[#6]", 50.17),
    (">N- (non-ring)", "[NX3H0;!R]([#6])([#6])[#6]", 11.74),
]
_COMPILED = [(name, Chem.MolFromSmarts(smarts), value) for name, smarts, value in JOBACK_GROUPS]


def joback_tb(smiles: str):
    """Joback boiling-point estimate (K), or None if the molecule has an
    atom not covered by the implemented groups."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(atom.GetIsotope() for atom in mol.GetAtoms()):
        return None
    covered = set()
    total = JOBACK_INTERCEPT
    for _, pattern, value in _COMPILED:
        for match in mol.GetSubstructMatches(pattern):
            covered.add(match[0])
            total += value
    if covered != {atom.GetIdx() for atom in mol.GetAtoms()}:
        return None
    return round(total, 4)


def joback_matches(df: pd.DataFrame, smiles_col: str = "isosmiles", label_col: str = "boiling_point_kelvin",
                   tolerance: float = 0.005) -> pd.DataFrame:
    """Joback estimate per compound and whether the label equals it
    (within `tolerance` K, i.e. identical to the hundredth of a kelvin;
    by chance, fewer than one of this dataset's labels would fall that close)."""
    out = df[["cmpdname", smiles_col, label_col]].copy()
    out["joback_tb"] = out[smiles_col].map(joback_tb)
    out["label_is_joback"] = (out["joback_tb"] - out[label_col]).abs() <= tolerance
    return out


def consistent_disagreement(predictions: pd.DataFrame, min_error: float = 50.0) -> pd.DataFrame:
    """Compounds whose every out-of-fold prediction (all feature sets,
    models and repeats) misses the label by more than `min_error` K in the
    same direction. predictions: rows with 'row', 'y_true', 'y_pred'."""
    residual = predictions["y_pred"] - predictions["y_true"]
    g = predictions.assign(residual=residual).groupby("row")["residual"]
    summary = pd.DataFrame({
        "mean_residual": g.mean(),
        "smallest_miss": g.apply(lambda r: r.abs().min()),
        "all_same_sign": g.apply(lambda r: (r > 0).all() or (r < 0).all()),
        "n_predictions": g.size(),
    })
    flagged = summary[summary["all_same_sign"] & (summary["smallest_miss"] > min_error)]
    return flagged.drop(columns="all_same_sign")


# Measured normal boiling points of n-alkanes C1-C24 (K; CRC Handbook). Longer
# alkanes decompose before boiling at 1 atm, so no measured reference exists.
N_ALKANE_TB = {1: 111.7, 2: 184.6, 3: 231.1, 4: 272.7, 5: 309.2, 6: 341.9, 7: 371.6, 8: 398.8,
               9: 424.0, 10: 447.3, 11: 469.1, 12: 489.5, 13: 508.6, 14: 526.7, 15: 543.8,
               16: 560.0, 17: 575.2, 18: 589.9, 19: 603.1, 20: 616.9, 21: 629.7, 22: 641.8,
               23: 653.4, 24: 664.5}


def hydrocarbon_carbon_count(smiles: str):
    """Number of carbons if the molecule is a plain (non-isotopic)
    hydrocarbon, else None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(a.GetSymbol() != "C" or a.GetIsotope() for a in mol.GetAtoms()):
        return None
    return mol.GetNumAtoms()


def implausible_hydrocarbons(df: pd.DataFrame, margin: float = 80.0, smiles_col: str = "isosmiles",
                             label_col: str = "boiling_point_kelvin") -> pd.DataFrame:
    """Hydrocarbons (C1-C24) whose label is more than `margin` K below the
    n-alkane with the same carbon count."""
    carbons = df[smiles_col].map(hydrocarbon_carbon_count)
    reference = carbons.map(N_ALKANE_TB)
    below = reference - df[label_col]
    out = df.assign(n_carbons=carbons, n_alkane_tb=reference, below_n_alkane=below)
    return out[out["below_n_alkane"] > margin]


def build_label_audit(df: pd.DataFrame, nist_lookups: pd.DataFrame, known_errors: dict,
                      nist_tolerance: float = 30.0) -> pd.DataFrame:
    """One row per audited compound: category, evidence, and whether it is
    excluded from the measured-labels dataset.

    nist_lookups: NIST results for the model-flagged compounds (columns
    cmpdname, nist_tb). known_errors: {name: evidence} found earlier."""
    rows = []
    joback = joback_matches(df)
    for _, r in joback[joback["label_is_joback"]].iterrows():
        rows.append({"cmpdname": r["cmpdname"], "label": r["boiling_point_kelvin"],
                     "category": "Joback estimate",
                     "evidence": f"label equals the Joback group-contribution estimate ({r['joback_tb']:.2f} K)",
                     "exclude": True})
    for _, r in implausible_hydrocarbons(df).iterrows():
        rows.append({"cmpdname": r["cmpdname"], "label": r["boiling_point_kelvin"],
                     "category": "implausible hydrocarbon",
                     "evidence": f"{r['below_n_alkane']:.0f} K below the C{int(r['n_carbons'])} n-alkane "
                                 f"({r['n_alkane_tb']:.1f} K)",
                     "exclude": True})
    labels = df.set_index("cmpdname")["boiling_point_kelvin"]
    for _, r in nist_lookups.iterrows():
        nist = pd.to_numeric(r["nist_tb"], errors="coerce")
        label = labels.get(r["cmpdname"], np.nan)
        if np.isnan(nist):
            category, evidence, exclude = "model-flagged, unverified", "no NIST boiling point", False
        elif abs(nist - label) > nist_tolerance:
            category, evidence, exclude = "contradicted by NIST", f"NIST {nist:.1f} K", True
        else:
            category, evidence, exclude = "NIST agrees", f"NIST {nist:.1f} K", False
        rows.append({"cmpdname": r["cmpdname"], "label": label, "category": category,
                     "evidence": evidence, "exclude": exclude})
    for name, evidence in known_errors.items():
        rows.append({"cmpdname": name, "label": labels.get(name, np.nan),
                     "category": "known data-entry error", "evidence": evidence, "exclude": True})
    audit = pd.DataFrame(rows)
    # A compound can be caught by more than one check: keep one row, excluded if any check excludes it
    audit["exclude"] = audit.groupby("cmpdname")["exclude"].transform("max")
    audit["category"] = audit.groupby("cmpdname")["category"].transform(lambda c: "; ".join(dict.fromkeys(c)))
    audit["evidence"] = audit.groupby("cmpdname")["evidence"].transform(lambda c: "; ".join(dict.fromkeys(c)))
    return audit.drop_duplicates("cmpdname").sort_values(["exclude", "category", "cmpdname"],
                                                         ascending=[False, True, True]).reset_index(drop=True)
