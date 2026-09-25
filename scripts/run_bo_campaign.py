"""Bayesian-optimisation campaigns for a boiling-point specification window.

Each 'experiment' reveals one measured boiling point from the 1,251
audited compounds (data/label_audit.csv). Every campaign starts from the
same random experiments for a given seed, then picks batches; success is
how many compounds inside the window it finds per experiment. The GP uses
the curated (log) RDKit descriptors, with constant and near-duplicate
columns dropped.

Usage:
    python scripts/run_bo_campaign.py [--lo 453 --hi 473] [--seeds 20] [--n-jobs -1]
    python scripts/run_bo_campaign.py --smoke     # 2 seeds, 40 experiments, temp dir
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.feature_selection import VarianceThreshold  # noqa: E402

from boiling_point import bayes_opt, descriptors  # noqa: E402
from run_feature_study import load_study_data  # noqa: E402

RESULTS_CSV = "results/bo_campaign.csv"
FIGURE = "results/images/bo_spec_window.png"


def gp_features(feature_frame: pd.DataFrame) -> np.ndarray:
    """Curated (log) descriptors minus constant and near-duplicate columns.
    Both filters look only at the features, never the boiling points, so
    fitting them on the whole pool leaks nothing."""
    X = VarianceThreshold().fit_transform(feature_frame.to_numpy(dtype=float))
    return descriptors.DropCorrelated().fit_transform(X)


def _run(X, y, mw, strategy, seed, lo, hi, budget, batch_size):
    start = time.time()
    r = bayes_opt.run_campaign(X, y, mw, strategy, seed, lo, hi, batch_size=batch_size, budget=budget)
    print(f"  {strategy:>14} seed {seed:2d}: {r['hits'].iloc[-1]:3d} hits  ({time.time() - start:.0f}s)", flush=True)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lo", type=float, default=453.0)
    parser.add_argument("--hi", type=float, default=473.0)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = range(2 if args.smoke else args.seeds)
    budget = 40 if args.smoke else args.budget

    df, feature_sets = load_study_data(audited=True)
    X = gp_features(feature_sets["curated (log)"])
    y = df["boiling_point_kelvin"].to_numpy(dtype=float)
    mw = df["mw"].to_numpy(dtype=float)
    n_in_spec = int(((y >= args.lo) & (y <= args.hi)).sum())
    print(f"{len(df)} measured compounds, {X.shape[1]} GP features; "
          f"{n_in_spec} in {args.lo:.0f}-{args.hi:.0f} K ({n_in_spec / len(df):.1%})")

    runs = Parallel(n_jobs=args.n_jobs)(
        delayed(_run)(X, y, mw, strategy, seed, args.lo, args.hi, budget, args.batch_size)
        for seed in seeds for strategy in bayes_opt.STRATEGIES)
    results = pd.concat(runs, ignore_index=True)
    results["cmpdname"] = df["cmpdname"].to_numpy()[results["row"]]

    out_dir = tempfile.mkdtemp() if args.smoke else "."
    results_path = os.path.join(out_dir, os.path.basename(RESULTS_CSV) if args.smoke else RESULTS_CSV)
    results.to_csv(results_path, index=False)

    import matplotlib
    matplotlib.use("Agg")  # here, not at import: notebooks import from this module
    from boiling_point import viz
    fig = viz.plot_bo_campaign(results, n_in_spec,
                               title=f"Finding compounds with a boiling point of {args.lo:.0f}-{args.hi:.0f} K")
    fig.savefig(os.path.join(out_dir, os.path.basename(FIGURE)) if args.smoke else FIGURE,
                dpi=150, bbox_inches="tight")

    print(bayes_opt.experiments_to_reach(results).round(0).to_string())
    print(results[results["experiment"].isin([50, 100, budget])]
          .groupby(["experiment", "strategy"])["hits"].agg(["mean", "std"]).round(1).unstack("experiment").to_string())
    print(f"results written to {results_path}")


if __name__ == "__main__":
    main()
