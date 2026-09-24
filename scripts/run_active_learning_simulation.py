"""Retrospective active-learning simulation (docs/active_learning_plan.md, Part A).

Pretends most of the labelled compounds (literature + NIST) are unknown, then
reveals them in rounds according to each selection strategy. At checkpoints,
the production ensemble is retrained on the labels so far and scored on a
fixed test set that is stratified on the hard region (high polar area /
rotatable bonds). Seeds x strategies run in parallel processes.

Usage:
    python scripts/run_active_learning_simulation.py [--seeds 10] [--n-jobs -1]
    python scripts/run_active_learning_simulation.py --smoke   # 2 seeds, up to 200 labels, temp dir
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from boiling_point import active_learning, data, features, viz  # noqa: E402

RESULTS_CSV = "results/active_learning_simulation.csv"
PICKS_CSV = "results/active_learning_picks.csv"
FIGURE = "results/images/active_learning_curves.png"
TEST_SPLIT_SEED = 42
PICKS_SAVED = 550  # labelling order saved up to this many labels (the small-batch phase)


def load_dataset():
    labels = pd.concat([
        data.load_literature_data("compound_boiling_points_from_literature.xlsx"),
        data.load_nist_data("compound_boiling_points_from_nist.csv"),
    ], ignore_index=True)
    pubchem = data.load_pubchem_data("compound_property_from_PubChem.csv")
    return data.build_modelling_table(labels, pubchem)


def _run_one(X, y, hard, test_idx, pool_idx, strategy, seed, max_labels):
    start = time.time()
    results = active_learning.run_strategy(X, y, hard, test_idx, pool_idx, strategy, seed,
                                           max_labels=max_labels)
    print(f"  {strategy:>15} seed {seed}: {time.time() - start:6.0f}s", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true", help="2 seeds, stop at 200 labels, write to a temp dir")
    args = parser.parse_args()
    seeds = range(2 if args.smoke else args.seeds)
    max_labels = 200 if args.smoke else None
    out = (lambda path: os.path.join(tempfile.mkdtemp(), os.path.basename(path))) if args.smoke else (lambda path: path)

    df = load_dataset()
    X = features.select_model_features(df).to_numpy(dtype=float)
    y = features.get_target(df).to_numpy(dtype=float)
    hard = features.in_hard_region(df).to_numpy()
    pool_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=hard,
                                          random_state=TEST_SPLIT_SEED)
    print(f"{len(df)} labelled compounds ({hard.sum()} in hard region); "
          f"test {len(test_idx)} ({hard[test_idx].sum()} hard), pool {len(pool_idx)}")

    jobs = [(strategy, seed) for seed in seeds for strategy in active_learning.STRATEGIES]
    runs = Parallel(n_jobs=args.n_jobs)(
        delayed(_run_one)(X, y, hard, test_idx, pool_idx, strategy, seed, max_labels)
        for strategy, seed in jobs
    )

    results = pd.concat(runs, ignore_index=True)
    results_path = out(RESULTS_CSV)
    results.to_csv(results_path, index=False)

    picks = pd.concat([
        pd.DataFrame({"seed": run["seed"].iloc[0], "strategy": run["strategy"].iloc[0],
                      "order": range(len(order)), "row": order,
                      "cmpdname": df["cmpdname"].to_numpy()[order],
                      "hard_region": hard[order]})
        for run in runs
        for order in [run.attrs["labelled_order"][:PICKS_SAVED]]
    ], ignore_index=True)
    picks.to_csv(out(PICKS_CSV), index=False)

    import matplotlib
    matplotlib.use("Agg")  # here, not at import: the notebook imports load_dataset from this module
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    viz.plot_active_learning_curves(results, "rmse", ax=axes[0], title="All test compounds")
    viz.plot_active_learning_curves(results, "rmse_hard", ax=axes[1],
                                    title="Hard region (polar area ≥ 40 or rotatable bonds ≥ 14)")
    fig.savefig(out(FIGURE), dpi=150, bbox_inches="tight")
    print(f"Results written to {results_path}")

    print(results.groupby(["strategy", "n_labels"])[["rmse", "rmse_hard"]].mean().round(1)
          .unstack("strategy").to_string())


if __name__ == "__main__":
    main()
