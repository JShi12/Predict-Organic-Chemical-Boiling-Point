"""How much did stereoisomer leakage flatter the feature-study numbers?

Runs the feature study's nested CV for the curated descriptors twice on the
same 1,250 measured labels: on the feature study's ungrouped folds
(validation.outer_splits) and on the representation study's
structure-grouped folds (results/representation_study/splits.csv). Same
models, tuning and data; only the outer folds differ. (The committed
results/feature_study_audited/ are on 1,251 labels, so they aren't reused.)

Usage:
    python scripts/compare_grouped_splits.py
"""
import sys

sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from boiling_point import validation  # noqa: E402
from run_feature_study import load_study_data  # noqa: E402

SPLITS_CSV = "results/representation_study/splits.csv"
OUT_CSV = "results/representation_study/grouped_vs_ungrouped.csv"
FEATURE_SETS = ["curated (log)", "curated (log) + family"]
N_REPEATS, N_SPLITS = 3, 5


def main():
    df, feature_sets = load_study_data(audited=True)
    splits = pd.read_csv(SPLITS_CSV)
    if not (splits["cmpdname"].to_numpy() == df["cmpdname"].to_numpy()).all():
        raise ValueError("splits.csv rows are not aligned with the study data")
    strat = validation.stratification_labels(df["family"], N_SPLITS)
    y = df["boiling_point_kelvin"].to_numpy()

    folds = {"grouped": [(r, k, np.flatnonzero(splits[f"fold_r{r}"] != k), np.flatnonzero(splits[f"fold_r{r}"] == k))
                         for r in range(N_REPEATS) for k in range(N_SPLITS)],
             "ungrouped": validation.outer_splits(strat, N_SPLITS, N_REPEATS)}
    jobs = [(kind, name, split) for kind in folds for name in FEATURE_SETS for split in folds[kind]]
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(validation.run_outer_fold)(
            feature_sets[name].to_numpy(dtype=float), y, strat, df["family"].to_numpy(), train, test, name, r, k)
        for _, name, (r, k, train, test) in jobs)
    kinds = pd.Series([kind for kind, _, _ in jobs])
    both = pd.concat([
        validation.fold_metrics(pd.concat([results[i][0] for i in kinds.index[kinds == kind]], ignore_index=True),
                                hard_mask=df["hard_region"].to_numpy()).assign(splits=kind)
        for kind in folds])
    both.to_csv(OUT_CSV, index=False)
    print(both.groupby(["feature_set", "model", "splits"])[["mae", "rmse"]].agg(["mean", "std"]).round(2)
          .to_string())


if __name__ == "__main__":
    main()
