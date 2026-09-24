"""Replay the heuristic NIST run in different orders (no NIST requests).

The heuristic run (scripts/scrape_nist_boiling_points.py) looked up all
5,400 hard-region candidates and found boiling points for 105. Because
every outcome is recorded in nist_boiling_points_targeted.csv, we can ask
how many of those lookups each ordering would have needed to collect the
same hits:

- rule_order: the order the heuristic run actually used;
- random: uniformly shuffled (the expected curve for any fixed-order rule);
- uncertainty: bootstrap XGBoost spread only (the first model-driven attempt);
- feasibility: predicted chance that a lookup succeeds only;
- feasibility_weighted: spread x chance of success.

Model-driven orders work in rounds of 100 and only learn from outcomes
revealed so far; the first round is random, to give the feasibility
classifier something to learn from. The uncertainty model is refit on the
literature compounds plus the hits revealed so far.

Usage:
    python scripts/replay_nist_lookups.py [--seeds 5] [--n-jobs -1]
"""
import argparse
import sys

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from boiling_point import active_learning, data, features, nist_scraper  # noqa: E402

OUTCOMES_CSV = "nist_boiling_points_targeted.csv"
RESULTS_CSV = "results/nist_lookup_replay.csv"
FIGURE = "results/images/nist_lookup_replay.png"
BATCH = 100
MODEL_ORDERS = ("uncertainty", "feasibility", "feasibility_weighted")


def load():
    literature = data.load_literature_data("compound_boiling_points_from_literature.xlsx")
    pubchem = data.load_pubchem_data("compound_property_from_PubChem.csv")
    labelled = data.build_modelling_table(literature, pubchem)
    outcomes = pd.read_csv(OUTCOMES_CSV)
    candidates = nist_scraper.select_nist_likely_candidates(pubchem, set(literature["cmpdname"]))
    pool = features.add_smiles_features(candidates.reset_index(drop=True))
    pool = outcomes.merge(pool, on="cmpdname", how="left")  # keeps the heuristic run's order
    assert pool["mw"].notna().all() and len(pool) == len(outcomes)
    return labelled, pool


def curve(order, found, strategy, seed):
    """Cumulative hits after each lookup, sampled every BATCH lookups."""
    hits = np.cumsum(found[order])
    steps = np.r_[np.arange(BATCH, len(order), BATCH), len(order)]
    return pd.DataFrame({"strategy": strategy, "seed": seed, "lookups": steps, "hits": hits[steps - 1]})


def model_order(strategy, seed, X_lab, y_lab, X_pool, y_pool, found):
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(X_pool))
    first = rng.choice(remaining, BATCH, replace=False)
    order = list(first)
    remaining = np.setdiff1d(remaining, first)
    round_no = 0
    while len(remaining):
        seen = np.array(order)
        hits = seen[found[seen]]
        batch = min(BATCH, len(remaining))
        if strategy in ("uncertainty", "feasibility_weighted"):
            models = active_learning.bootstrap_xgb_models(
                np.vstack([X_lab, X_pool[hits]]), np.concatenate([y_lab, y_pool[hits]]),
                random_state=seed * 1000 + round_no)
        if strategy in ("feasibility", "feasibility_weighted") and found[seen].any():
            classifier = active_learning.fit_feasibility_classifier(X_pool[seen], found[seen])
        else:
            classifier = None
        if strategy == "uncertainty":
            picks = active_learning.bootstrap_select_batch(models, X_pool[remaining], batch)
        elif classifier is None:  # nothing found yet: nothing to learn feasibility from
            picks = list(rng.choice(len(remaining), batch, replace=False))
        elif strategy == "feasibility":
            p = classifier.predict_proba(X_pool[remaining])[:, 1]
            picks = list(np.argsort(-p, kind="stable")[:batch])
        else:
            picks = active_learning.feasibility_weighted_batch(models, classifier, X_pool[remaining], batch)
        order.extend(remaining[picks])
        remaining = np.delete(remaining, picks)
        round_no += 1
    return curve(np.array(order), found, strategy, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    labelled, pool = load()
    X_lab = features.select_model_features(labelled).to_numpy(dtype=float)
    y_lab = features.get_target(labelled).to_numpy(dtype=float)
    X_pool = features.select_model_features(pool).to_numpy(dtype=float)
    found = (pool["status"] == "found").to_numpy()
    y_pool = pool["boiling_point_kelvin"].to_numpy(dtype=float)
    print(f"replaying {len(pool)} lookups with {found.sum()} hits")

    curves = [curve(np.arange(len(pool)), found, "rule_order", 0)]
    rng = np.random.default_rng(0)
    curves += [curve(rng.permutation(len(pool)), found, "random", s) for s in range(20)]
    curves += Parallel(n_jobs=args.n_jobs)(
        delayed(model_order)(strategy, seed, X_lab, y_lab, X_pool, y_pool, found)
        for strategy in MODEL_ORDERS for seed in range(args.seeds))
    results = pd.concat(curves, ignore_index=True)
    results.to_csv(RESULTS_CSV, index=False)

    targets = (25, 50, 75, 100, int(found.sum()))
    rows = []
    for (strategy, seed), c in results.groupby(["strategy", "seed"]):
        rows.append({"strategy": strategy, "seed": seed,
                     **{f"lookups_to_{t}": c.loc[c["hits"] >= t, "lookups"].min() for t in targets}})
    print(pd.DataFrame(rows).groupby("strategy").mean().drop(columns="seed").round(0).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from boiling_point import viz
    fig, ax = plt.subplots(figsize=(9, 4.2))
    viz.plot_lookup_replay(results, ax=ax)
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
