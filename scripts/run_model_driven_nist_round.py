"""Real model-driven NIST collection round (docs/active_learning_plan.md, Part B).

The counterpart to scripts/scrape_nist_boiling_points.py: instead of the
hand-written rule (polar area >= 40 or rotatable bonds >= 14), a selector
model trained on the literature compounds picks which PubChem candidates to
look up next. The default selector is the bootstrap XGBoost ensemble, which
beat the GP in the retrospective simulation and tracked the production
ensemble's errors better (see Active_Learning.ipynb); --selector
gp_uncertainty uses the GP instead. The candidate pool uses only the NIST-likely-name filter, with
no chemical-region filter, so the model is free to look outside the region
the rule chose.

Works in rounds: fit the selector on literature + hits so far, pick the
next batch it is most uncertain about, look them up (reusing cached outcomes
from the earlier heuristic run where available, otherwise scraping NIST),
repeat. Compounds NIST doesn't have leave the pool without a label, so their
neighbourhood stays uncertain and the next round can try a nearby compound.

Stops at --target-hits (default 105, matching the heuristic run) or
--max-lookups (default 3000). Progress is appended to the output CSV one row
at a time; reruns resume from it.

Usage:
    python scripts/run_model_driven_nist_round.py [--output nist_boiling_points_model_driven.csv]
    python scripts/run_model_driven_nist_round.py --max-lookups 5 --output /tmp/dry_run.csv  # dry run
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from boiling_point import active_learning, data, features, nist_scraper  # noqa: E402

CACHE_CSV = "nist_boiling_points_targeted.csv"
FIELDS = nist_scraper.CSV_FIELDS + ["round", "rank_in_round", "source"]


def load_pools():
    literature = data.load_literature_data("compound_boiling_points_from_literature.xlsx")
    pubchem = data.load_pubchem_data("compound_property_from_PubChem.csv")
    labelled = data.build_modelling_table(literature, pubchem)
    candidates = nist_scraper.select_nist_likely_candidates(pubchem, set(literature["cmpdname"]))
    candidates = features.add_smiles_features(candidates.reset_index(drop=True))
    return labelled, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="nist_boiling_points_model_driven.csv")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--target-hits", type=int, default=105)
    parser.add_argument("--max-lookups", type=int, default=3000)
    parser.add_argument("--selector", choices=["bootstrap_xgb", "gp_uncertainty"], default="bootstrap_xgb")
    args = parser.parse_args()

    labelled, candidates = load_pools()
    cache = pd.read_csv(CACHE_CSV).set_index("cmpdname")
    cache = cache[~cache["status"].astype(str).str.startswith("error")]

    done = pd.read_csv(args.output) if os.path.exists(args.output) else pd.DataFrame(columns=FIELDS)
    hits = done[done["status"] == "found"]
    round_no = int(done["round"].max()) + 1 if len(done) else 0
    print(f"{len(labelled)} literature compounds; {len(candidates)} candidates; "
          f"resuming with {len(done)} lookups / {len(hits)} hits")

    X_cand = features.select_model_features(candidates).to_numpy(dtype=float)
    X_lab = features.select_model_features(labelled).to_numpy(dtype=float)
    y_lab = features.get_target(labelled).to_numpy(dtype=float)
    X_gp_cand = active_learning.gp_inputs(X_cand)
    X_gp_lab = active_learning.gp_inputs(X_lab)
    all_gp = np.vstack([X_gp_lab, X_gp_cand])
    bounds = np.stack([all_gp.min(axis=0), all_gp.max(axis=0)])
    name_to_pos = {name: i for i, name in enumerate(candidates["cmpdname"])}

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FIELDS)

        gp, last_optimised_n = None, 0
        while len(hits) < args.target_hits and len(done) < args.max_lookups:
            hit_pos = [name_to_pos[n] for n in hits["cmpdname"]]
            y_train = np.concatenate([y_lab, hits["boiling_point_kelvin"].to_numpy(dtype=float)])
            looked_up = set(done["cmpdname"])
            remaining = np.array([i for i, n in enumerate(candidates["cmpdname"]) if n not in looked_up])
            batch = min(args.batch_size, args.max_lookups - len(done), len(remaining))

            start = time.time()
            if args.selector == "bootstrap_xgb":
                X_train = np.vstack([X_lab, X_cand[hit_pos]])
                models = active_learning.bootstrap_xgb_models(X_train, y_train, random_state=round_no)
                positions = active_learning.bootstrap_select_batch(models, X_cand[remaining], batch)
            else:
                X_train = np.vstack([X_gp_lab, X_gp_cand[hit_pos]])
                optimise = len(y_train) >= active_learning.REOPTIMISE_GROWTH * last_optimised_n
                gp = active_learning.fit_gp(X_train, y_train, bounds, hyperparameters_from=gp,
                                            optimise=optimise)
                if optimise:
                    last_optimised_n = len(y_train)
                positions = active_learning.gp_select_batch(gp, X_gp_cand[remaining], batch)
            picks = remaining[positions]
            print(f"round {round_no}: {args.selector} on {len(y_train)} labels, picked {batch} "
                  f"in {time.time() - start:.0f}s", flush=True)

            new_rows = []
            for rank, pos in enumerate(picks):
                name = candidates["cmpdname"].iloc[pos]
                if name in cache.index:
                    c = cache.loc[name]
                    row = [name, c["nist_id"], c["boiling_point_kelvin"], c["n_measurements"], c["status"]]
                    source = "cache"
                else:
                    row = nist_scraper.lookup_boiling_point(name)
                    source = "scraped"
                row = [("" if pd.isna(v) else v) for v in row] + [round_no, rank, source]
                writer.writerow(row)
                f.flush()
                new_rows.append(dict(zip(FIELDS, row)))

            done = pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True)
            hits = done[done["status"] == "found"]
            round_hits = sum(r["status"] == "found" for r in new_rows)
            print(f"  {round_hits}/{batch} found; total {len(hits)} hits / {len(done)} lookups "
                  f"({len(hits) / len(done):.1%})", flush=True)
            round_no += 1

    print("Done.")


if __name__ == "__main__":
    main()
