"""Do the NIST additions improve the production ensemble? (Part B evaluation)

Compares the ensemble trained on the literature train+val split alone with
it trained on that plus (a) the heuristic run's 105 compounds, (b) the
feasibility-aware run's first 105 hits, (c) both. All are scored on the
same literature test split (the main notebook's), with paired bootstrap
intervals.

The model-chosen compounds are mostly halogenated, and the literature
dataset has almost none, so the literature test set can't reward them. A
second check therefore splits the model-chosen 105 in half, adds one half,
and tests on the other (3 random splits x both halves).

Usage:
    python scripts/evaluate_nist_additions.py
"""
import sys
import warnings

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from boiling_point import active_learning, data, features, preprocessing  # noqa: E402

COMPARISON_CSV = "results/nist_additions_comparison.csv"
HELDOUT_CSV = "results/nist_model_chosen_heldout.csv"
N_ADDED = 105


def to_xy(table):
    return (features.select_model_features(table).to_numpy(dtype=float),
            features.get_target(table).to_numpy(dtype=float))


def heldout_run(X_base, y_base, X_new, y_new, split_seed, half):
    idx = np.random.default_rng(split_seed).permutation(len(y_new))
    cut = len(idx) // 2
    add, held = (idx[:cut], idx[cut:]) if half == 0 else (idx[cut:], idx[:cut])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = active_learning.make_production_ensemble().fit(X_base, y_base)
        augmented = active_learning.make_production_ensemble().fit(
            np.vstack([X_base, X_new[add]]), np.concatenate([y_base, y_new[add]]))
    rmse = lambda m: float(np.sqrt(np.mean((m.predict(X_new[held]) - y_new[held]) ** 2)))
    return {"split": split_seed, "half": half, "literature_only": rmse(base),
            "with_other_half": rmse(augmented)}


def main():
    pubchem = data.load_pubchem_data("compound_property_from_PubChem.csv")
    literature = data.build_modelling_table(
        data.load_literature_data("compound_boiling_points_from_literature.xlsx"), pubchem)
    X = features.select_model_features(literature)
    y = features.get_target(literature)
    hard = features.in_hard_region(literature)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.train_val_test_split(X, y, random_state=42)
    X_base = pd.concat([X_train, X_val]).to_numpy(dtype=float)
    y_base = pd.concat([y_train, y_val]).to_numpy(dtype=float)

    heuristic = data.build_modelling_table(data.load_nist_data("compound_boiling_points_from_nist.csv"), pubchem)
    run = pd.read_csv("nist_boiling_points_feasibility_aware.csv")
    model_hits = run[run["status"] == "found"].head(N_ADDED)[["cmpdname", "boiling_point_kelvin"]]
    model_chosen = data.build_modelling_table(model_hits, pubchem)
    X_h, y_h = to_xy(heuristic)
    X_m, y_m = to_xy(model_chosen)

    comparison = active_learning.compare_additions(
        X_base, y_base, X_test.to_numpy(dtype=float), y_test.to_numpy(dtype=float),
        hard.loc[X_test.index].to_numpy(),
        {"+ heuristic 105": (X_h, y_h), "+ model-chosen 105": (X_m, y_m),
         "+ both (210)": (np.vstack([X_h, X_m]), np.concatenate([y_h, y_m]))})
    comparison.to_csv(COMPARISON_CSV)
    print(comparison.round(2).to_string())

    heldout = pd.DataFrame(Parallel(n_jobs=-1)(
        delayed(heldout_run)(X_base, y_base, X_m, y_m, seed, half) for seed in range(3) for half in (0, 1)))
    heldout.to_csv(HELDOUT_CSV, index=False)
    print(heldout.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
