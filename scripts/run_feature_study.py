"""Nested, family-stratified CV of the rich-feature study (docs/rich_features_plan.md).

Compares feature sets (the original 12, the curated RDKit set as-is and with
log-transformed size/shape descriptors, and the log version + family one-hot)
across Ridge / XGBoost / MLP / their ensemble, plus per-family XGBoost
models, with hyperparameters tuned inside each outer training fold. Then
fits the final model: the recipe with the lowest mean outer-fold MAE (RMSE
is reported too), tuned and fitted on all compounds.

--audited repeats the study on measured labels only: it leaves out every
compound that data/label_audit.csv (scripts/audit_labels.py) marks as
excluded (Joback estimates, implausible or NIST-contradicted labels, known
data-entry errors) and writes to results/feature_study_audited/.

Usage:
    python scripts/run_feature_study.py [--n-jobs -1]
    python scripts/run_feature_study.py --audited
    python scripts/run_feature_study.py --smoke   # 2 folds x 1 repeat, tiny search, temp dir
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "src")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from boiling_point import data, descriptors, families, features, validation  # noqa: E402

OUT_DIR = "results/feature_study"
AUDITED_OUT_DIR = "results/feature_study_audited"
LABEL_AUDIT_CSV = "data/label_audit.csv"
# Likely data-entry errors found by cross-referencing recurring outliers
# (main notebook, Conclusions point 4): both ~90-100 K too low.
EXCLUDED = ["2,6-Nonadien-1-ol", "N-Methyldodecylamine"]
PER_FAMILY_MIN = 100


def load_study_data(audited: bool = False):
    """Literature compounds (minus EXCLUDED, or minus every label the audit
    excludes if audited) with family, hard-region flag, and the feature sets."""
    pubchem = data.load_pubchem_data("compound_property_from_PubChem.csv")
    literature = data.build_modelling_table(
        data.load_literature_data("compound_boiling_points_from_literature.xlsx"), pubchem)
    missing = set(EXCLUDED) - set(literature["cmpdname"])
    assert not missing, f"excluded compounds not found: {missing}"
    excluded = set(EXCLUDED)
    if audited:
        audit = pd.read_csv(LABEL_AUDIT_CSV)
        excluded |= set(audit.loc[audit["exclude"], "cmpdname"])
    df = literature[~literature["cmpdname"].isin(excluded)].reset_index(drop=True)
    df["family"] = families.assign_families(df["isosmiles"]).to_numpy()
    df["hard_region"] = features.in_hard_region(df).to_numpy()

    curated = descriptors.curated_descriptors(df["isosmiles"])
    curated_log = descriptors.log_size_shape(curated)
    family_onehot = pd.get_dummies(df["family"], prefix="family", dtype=float)
    feature_sets = {
        "old 12": df[features.MODEL_FEATURE_COLUMNS].astype(float),
        "curated": curated,
        "curated (log)": curated_log,
        "curated (log) + family": pd.concat([curated_log, family_onehot], axis=1),
    }
    return df, feature_sets


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audited", action="store_true", help="measured labels only (see data/label_audit.csv)")
    args = parser.parse_args()
    n_splits, n_repeats, n_iter = (2, 1, 2) if args.smoke else (5, 3, args.n_iter)
    out_dir = tempfile.mkdtemp() if args.smoke else (AUDITED_OUT_DIR if args.audited else OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    df, feature_sets = load_study_data(audited=args.audited)
    strat = validation.stratification_labels(df["family"], n_splits)
    print(f"{len(df)} compounds; families: {df['family'].value_counts().to_dict()}")
    splits = validation.outer_splits(strat, n_splits, n_repeats)

    start = time.time()
    jobs = [(name, split) for split in splits for name in feature_sets]
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(validation.run_outer_fold)(
            feature_sets[name].to_numpy(dtype=float), df["boiling_point_kelvin"].to_numpy(),
            strat, df["family"].to_numpy(), train_idx, test_idx, name, repeat, fold,
            n_iter=n_iter, per_family_min=PER_FAMILY_MIN if name == "curated (log)" else None)
        for name, (repeat, fold, train_idx, test_idx) in jobs)
    print(f"nested CV: {time.time() - start:.0f}s")

    predictions = pd.concat([p for p, _ in results], ignore_index=True)
    fold_rows = pd.concat([r for _, r in results], ignore_index=True)
    metrics = validation.fold_metrics(predictions, hard_mask=df["hard_region"].to_numpy())
    predictions.to_csv(f"{out_dir}/predictions.csv", index=False)
    fold_rows.to_csv(f"{out_dir}/fold_summary.csv", index=False)
    metrics.to_csv(f"{out_dir}/fold_metrics.csv", index=False)
    df[["cmpdname", "isosmiles", "family", "hard_region", "boiling_point_kelvin"]].to_csv(
        f"{out_dir}/compounds.csv", index_label="row")

    summary = (metrics.groupby(["feature_set", "model"])[["rmse", "mae", "r2", "rmse_hard"]]
               .agg(["mean", "std"]).round(2))
    print(summary.to_string())

    global_recipes = metrics[metrics["model"].isin(validation.BASE_MODELS + (validation.ENSEMBLE,))]
    means = global_recipes.groupby(["feature_set", "model"])[["mae", "rmse"]].mean()
    best = means["mae"].idxmin()
    rmse_best = means["rmse"].idxmin()
    print(f"winning recipe by MAE: {best}; by RMSE it would be {rmse_best}")
    print(f"fitting the final model on all {len(df)} compounds")
    final, params = validation.fit_final_model(
        best[1], feature_sets[best[0]].to_numpy(dtype=float), df["boiling_point_kelvin"].to_numpy(),
        strat, n_inner=n_splits, n_iter=n_iter)
    joblib.dump({"model": final, "feature_set": best[0], "model_type": best[1],
                 "feature_columns": list(feature_sets[best[0]].columns)}, f"{out_dir}/final_model.joblib")
    with open(f"{out_dir}/final_model.json", "w") as f:
        json.dump({"feature_set": best[0], "model": best[1], "selected_by": "mean outer-fold MAE",
                   "rmse_winner": list(rmse_best), "params": params}, f, indent=1, default=str)
    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
