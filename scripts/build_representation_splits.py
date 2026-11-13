"""Build the fixed splits for the representation study (Step 1).

Writes, under results/representation_study/:
- splits.csv: one row per measured literature compound (data/label_audit.csv
  exclusions removed), with its structure group, family, familiar-chemistry
  fold per repeat, held-out family, and learning-curve entry sizes;
- nist_external.csv: the external test set (both NIST rounds), every row kept
  with an exclude flag and reason, plus subgroup and similarity to training;
- splits_config.json: every setting and seed used.

These files are fixed before any model in the study is compared.

Usage:
    python scripts/build_representation_splits.py
"""
import json
import os
import sys

sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from boiling_point import audit, data, descriptors, splits, validation  # noqa: E402
from run_feature_study import load_study_data  # noqa: E402

OUT_DIR = "results/representation_study"
N_SPLITS, N_REPEATS, SPLIT_SEED = 5, 3, 0
LC_SIZES, LC_DRAWS, LC_SEED = [100, 250, 500], 3, 0   # plus the full training fold
MIN_SUBGROUP = 30
NIST_HEURISTIC_CSV = "data/compound_boiling_points_from_nist.csv"
NIST_HEURISTIC_LOOKUPS_CSV = "data/nist_boiling_points_targeted.csv"
NIST_FEASIBILITY_CSV = "data/nist_boiling_points_feasibility_aware.csv"


def build_literature_splits(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["cmpdname", "isosmiles", "family", "boiling_point_kelvin"]].copy()
    out.insert(0, "mol_id", np.arange(len(out)))
    out["stereo_free_smiles"] = out["isosmiles"].map(audit.stereo_free_smiles)
    out["inchikey14"] = out["isosmiles"].map(splits.inchikey_connectivity)
    # Group by InChIKey connectivity: it ignores stereochemistry *and* isotopes, so
    # isotopologues (methane / deuterated methane) share a group as well as stereoisomers
    out["structure_group"] = splits.group_ids(out["inchikey14"])
    n_smiles = out.groupby("structure_group")["stereo_free_smiles"].transform("nunique")
    if (n_smiles > 1).any():
        print("groups joining different stereo-free SMILES (isotopologues):",
              out.loc[n_smiles > 1, "cmpdname"].tolist())

    strat = validation.stratification_labels(out["family"], N_SPLITS)
    folds = splits.familiar_splits(strat, out["structure_group"], N_SPLITS, N_REPEATS, SPLIT_SEED)
    for r in range(N_REPEATS):
        out[f"fold_r{r}"] = folds[:, r]
    out["holdout_family"] = splits.holdout_family(out["family"])

    # Learning curves use repeat 0's folds; one column per (fold, draw)
    rng = np.random.default_rng(LC_SEED)
    for k in range(N_SPLITS):
        train = np.flatnonzero(folds[:, 0] != k)
        for d in range(LC_DRAWS):
            entry = np.full(len(out), -1)
            entry[train] = splits.nested_subset_entry(out["family"].iloc[train], out["structure_group"].iloc[train],
                                                      LC_SIZES, rng)
            out[f"lc_entry_f{k}_d{d}"] = entry   # -1: in the test fold; len(train): full fold only
    return out


def build_nist_external(train: pd.DataFrame) -> pd.DataFrame:
    pubchem = data.load_pubchem_data()
    heuristic = data.load_nist_data(NIST_HEURISTIC_CSV).merge(
        pd.read_csv(NIST_HEURISTIC_LOOKUPS_CSV).query("status == 'found'")[["cmpdname", "n_measurements"]],
        on="cmpdname", how="left").assign(nist_round="heuristic")
    run = pd.read_csv(NIST_FEASIBILITY_CSV)
    feasibility = run[run["status"] == "found"][["cmpdname", "boiling_point_kelvin", "n_measurements"]].assign(
        nist_round="feasibility-aware")
    labels = pd.concat([heuristic, feasibility], ignore_index=True)
    nist = labels.merge(pubchem[["cmpdname", "isosmiles"]], on="cmpdname", how="left")
    if nist["isosmiles"].isna().any():
        raise ValueError(f"no PubChem SMILES for {nist.loc[nist['isosmiles'].isna(), 'cmpdname'].tolist()}")
    nist["stereo_free_smiles"] = nist["isosmiles"].map(audit.stereo_free_smiles)
    nist["subgroup"] = nist["isosmiles"].map(splits.nist_subgroup)

    reasons = pd.Series([[] for _ in range(len(nist))], index=nist.index)
    in_train = nist["stereo_free_smiles"].isin(set(train["stereo_free_smiles"]))
    for i in nist.index[in_train]:
        reasons[i].append("structure is in the training set")
    duplicate = nist["stereo_free_smiles"].duplicated(keep="first")
    for i in nist.index[duplicate & ~in_train]:
        reasons[i].append("duplicate structure within the NIST sets (first kept)")
    joback = audit.joback_matches(nist)
    for i in joback.index[joback["label_is_joback"]]:
        reasons[i].append(f"label equals the Joback estimate ({joback.loc[i, 'joback_tb']:.2f} K)")
    for i, r in audit.implausible_hydrocarbons(nist).iterrows():
        reasons[i].append(f"{r['below_n_alkane']:.0f} K below the C{int(r['n_carbons'])} n-alkane")
    conflicts = audit.stereoisomer_conflicts(nist)
    for _, r in conflicts[conflicts["exclude"]].iterrows():
        reasons[nist.index[nist["cmpdname"] == r["cmpdname"]][0]].append(r["evidence"])

    descriptor_ok = []
    for s in nist["isosmiles"]:
        try:
            descriptors.curated_descriptors(pd.Series([s]))
            descriptor_ok.append(True)
        except ValueError:
            descriptor_ok.append(False)
    nist["descriptors_ok"] = descriptor_ok
    for i in nist.index[~nist["descriptors_ok"]]:
        reasons[i].append("curated descriptors not computable")

    nist["exclude"] = reasons.map(bool)
    nist["exclude_reason"] = reasons.map("; ".join)
    nist["single_measurement"] = nist["n_measurements"] == 1

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train["isosmiles"]]
    nist["nn_tanimoto_train"] = [max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)),
                                                                        train_fps)) for s in nist["isosmiles"]]
    return nist


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, _ = load_study_data(audited=True)
    lit = build_literature_splits(df)
    nist = build_nist_external(lit)

    kept = nist[~nist["exclude"]]
    sizes = kept["subgroup"].value_counts()
    if (sizes < MIN_SUBGROUP).any():
        print(f"subgroups below {MIN_SUBGROUP}: {sizes[sizes < MIN_SUBGROUP].to_dict()} -- merge before modelling")

    lit.to_csv(f"{OUT_DIR}/splits.csv", index=False)
    nist.to_csv(f"{OUT_DIR}/nist_external.csv", index=False)
    config = {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "split_seed": SPLIT_SEED,
              "splitter": "StratifiedGroupKFold(shuffle=True, random_state=split_seed + repeat)",
              "stratify": "family (families with < n_splits members merged into the largest, for splitting only)",
              "group": "InChIKey connectivity block (ignores stereochemistry and isotopes)",
              "holdout_families": list(splits.HOLDOUT_FAMILIES),
              "learning_curve": {"sizes": LC_SIZES + ["full"], "draws": LC_DRAWS, "seed": LC_SEED,
                                 "folds": "repeat 0", "nested": True},
              "nist_min_subgroup": MIN_SUBGROUP,
              "n_literature": len(lit), "n_nist_rows": len(nist), "n_nist_kept": len(kept)}
    with open(f"{OUT_DIR}/splits_config.json", "w") as f:
        json.dump(config, f, indent=1)

    print(f"literature: {len(lit)} compounds, {lit['structure_group'].nunique()} structure groups")
    print("fold sizes, repeat 0:", lit["fold_r0"].value_counts().sort_index().tolist())
    print("held-out families:", lit["holdout_family"].replace("", "(excluded)").value_counts().to_dict())
    print(f"NIST: {len(nist)} rows, {len(kept)} kept; subgroups kept: {sizes.to_dict()}")
    print("NIST exclusions:", nist.loc[nist["exclude"], "exclude_reason"].value_counts().to_dict())


if __name__ == "__main__":
    main()
