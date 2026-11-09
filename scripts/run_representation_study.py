"""Frozen-representation comparison (representation study, Step 3).

Stage A (Ridge): curated (log) descriptors, Morgan counts and frozen
MoLFormer embeddings, in all three regimes (familiar chemistry, family
holdout, NIST external). The MoLFormer layer is chosen by Ridge in the inner
CV. Also writes the layer probe (diagnostic) and the reference models.

Stage B (XGBoost): the same tasks and inner folds, MoLFormer on the layer
Stage A chose; then the best pipeline per representation (lower inner-CV
MAE) and the provisional confirmatory comparisons C1-C3. They are
provisional because Holm's correction needs C4 (Step 6) and C5 (Step 4).

Reads results/representation_study/{splits,nist_external}.csv and the
MoLFormer cache (scripts/build_representations.py). No torch is imported:
xgboost and the encoder can't share a process (see representations.py).

Usage:
    python scripts/run_representation_study.py --stage A [--smoke]
    python scripts/run_representation_study.py --stage B [--smoke]
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from boiling_point import descriptors, representations as rep, study, study_stats, validation  # noqa: E402

SPLITS_CSV = "results/representation_study/splits.csv"
NIST_CSV = "results/representation_study/nist_external.csv"
CACHE_DIR = "data/cache/molformer"
OUT_DIR = "results/representation_study/frozen"
SMOKE_ALPHAS = [1, 100, 1e4]
SMOKE_XGB = {"max_depth": [3], "min_child_weight": [3], "learning_rate": [0.1], "n_estimators": [50]}


def load_data() -> dict:
    lit = pd.read_csv(SPLITS_CSV)
    nist_all = pd.read_csv(NIST_CSV)
    nist = nist_all[~nist_all["exclude"]]
    keys = list("literature:" + lit["mol_id"].astype(str)) + list("nist:" + nist.index.astype(str))
    smiles = pd.Series(list(lit["isosmiles"]) + list(nist["isosmiles"])).map(rep.model_smiles)
    n_lit = len(lit)
    families = np.array(list(lit["family"]) + ["(nist)"] * len(nist), dtype=object)
    groups = np.concatenate([lit["structure_group"].to_numpy(), -1 - np.arange(len(nist))])
    y = np.concatenate([lit["boiling_point_kelvin"].to_numpy(), nist["boiling_point_kelvin"].to_numpy()])
    curated_raw = descriptors.curated_descriptors(smiles)
    curated = descriptors.log_size_shape(curated_raw)
    X = {"curated": curated.to_numpy(dtype=float),
         "morgan": rep.morgan_counts(smiles),
         "molformer": np.asarray(rep.load_cache(CACHE_DIR, keys, smiles, "canonical", rep.checked_config()),
                                 dtype=np.float64)}
    family_onehot = pd.get_dummies(pd.Series(lit["family"]), prefix="family", dtype=float)
    return {"lit": lit, "nist": nist, "n_lit": n_lit, "keys": np.array(keys), "smiles": smiles, "y": y,
            "families": families, "groups": groups, "X": X, "molwt": curated_raw[["MolWt"]].to_numpy(dtype=float),
            "family_features": pd.concat([curated.iloc[:n_lit].reset_index(drop=True), family_onehot], axis=1)
            .to_numpy(dtype=float)}


def make_tasks(d: dict, smoke: bool) -> list:
    tasks = (study.familiar_tasks(d["lit"]) + study.holdout_tasks(d["lit"])
             + [study.nist_task(d["n_lit"], np.arange(len(d["nist"])))])
    if smoke:
        tasks = [t for t in tasks if t["task"] in ("r0f0", "amine", "nist")]
    return tasks


def records(results: list, keys: np.ndarray) -> dict:
    """Split run_task results into tidy tables."""
    preds, selection, inner_test, inner_resid, probe = [], [], [], [], []
    for r in results:
        base = {k: r[k] for k in ("regime", "task", "repeat", "fold", "representation", "regressor")}
        preds.append(pd.DataFrame({**base, "key": keys[r["test"]], "y_true": r["y_true"], "y_pred": r["pred"]}))
        selection.append({**base, "layer": r["layer"], "params": json.dumps(r["params"]), "inner_mae": r["inner_mae"],
                          "seconds": r["seconds"], "fit_seconds": r["fit_seconds"]})
        for i, p in enumerate(r["inner_test"]):
            inner_test.append(pd.DataFrame({**base, "inner_fold": i, "key": keys[r["test"]], "y_pred": p}))
        for i, va, resid in r["inner_resid"]:
            inner_resid.append(pd.DataFrame({**base, "inner_fold": i, "key": keys[r["train"][va]], "residual": resid}))
        for row in r["probe"]:
            probe.append({**base, **row})
    return {"predictions": pd.concat(preds, ignore_index=True), "selection": pd.DataFrame(selection),
            "cvplus_test": pd.concat(inner_test, ignore_index=True),
            "cvplus_residuals": pd.concat(inner_resid, ignore_index=True), "layer_probe": pd.DataFrame(probe)}


def reference_rows(d: dict, tasks: list, alphas) -> pd.DataFrame:
    """Training mean, molecular-weight-only Ridge, and (familiar only) the
    feature study's curated (log) + family ensemble on the same folds."""
    rows = []
    y = d["y"]
    for t in tasks:
        base = {k: t[k] for k in ("regime", "task", "repeat", "fold")}
        keys, test = d["keys"][t["test"]], t["test"]
        rows.append(pd.DataFrame({**base, "reference": "training mean", "key": keys, "y_true": y[test],
                                  "y_pred": np.full(len(test), y[t["train"]].mean())}))
        mw = study.run_task(t, "curated", "ridge", d["molwt"], y, d["families"], d["groups"], alphas=alphas)
        rows.append(pd.DataFrame({**base, "reference": "molecular weight (Ridge)", "key": keys, "y_true": y[test],
                                  "y_pred": mw["pred"]}))
        if t["regime"] == "familiar":
            strat = validation.stratification_labels(pd.Series(d["lit"]["family"]), 5)
            pred, _ = validation.run_outer_fold(d["family_features"], y[:d["n_lit"]], strat,
                                                d["lit"]["family"].to_numpy(), t["train"], t["test"],
                                                "curated (log) + family", t["repeat"], t["fold"])
            ens = pred[pred["model"] == validation.ENSEMBLE].set_index("row").loc[test, "y_pred"]
            rows.append(pd.DataFrame({**base, "reference": "curated (log) + family ensemble (feature study, "
                                                            "ungrouped inner CV)",
                                      "key": keys, "y_true": y[test], "y_pred": ens.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def save(tables: dict, out: str, prefix: str):
    os.makedirs(out, exist_ok=True)
    for name, table in tables.items():
        if len(table):
            # the CV+ tables (for Step 5's intervals) are large, so they're gzipped
            suffix = ".csv.gz" if name.startswith("cvplus") else ".csv"
            table.to_csv(f"{out}/{prefix}{name}{suffix}", index=False)


def stage_a(d, tasks, out, n_jobs, alphas):
    jobs = [(t, r) for t in tasks for r in study.REPRESENTATIONS]
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(study.run_task)(t, r, "ridge", d["X"][r], d["y"], d["families"], d["groups"], alphas=alphas,
                                layer_probe=(r == "molformer" and t["regime"] == "familiar"))
        for t, r in jobs)
    tables = records(results, d["keys"])
    probe = tables.pop("layer_probe")
    save(tables, out, "ridge_")
    probe.to_csv(f"{out}/layer_probe.csv", index=False)
    references = Parallel(n_jobs=n_jobs)(delayed(reference_rows)(d, [t], alphas) for t in tasks)
    pd.concat(references, ignore_index=True).to_csv(f"{out}/references.csv", index=False)
    return tables


def stage_b(d, tasks, out, n_jobs, xgb_grid):
    chosen = pd.read_csv(f"{out}/ridge_selection.csv")
    layer = chosen[chosen["representation"] == "molformer"].set_index("task")["layer"].astype(int).to_dict()
    jobs = [(t, r) for t in tasks for r in study.REPRESENTATIONS]
    # the slow MoLFormer jobs first, so the parallel pool stays busy
    jobs.sort(key=lambda j: j[1] != "molformer")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(study.run_task)(t, r, "xgboost", d["X"][r], d["y"], d["families"], d["groups"],
                                layer=layer.get(t["task"]) if r == "molformer" else None, xgb_param_grid=xgb_grid)
        for t, r in jobs)
    tables = records(results, d["keys"])
    tables.pop("layer_probe")
    save(tables, out, "xgboost_")
    return tables


def best_pipelines(out: str) -> pd.DataFrame:
    """Per (task, representation): the regressor with the lower inner-CV MAE."""
    sel = pd.concat([pd.read_csv(f"{out}/{r}_selection.csv") for r in ("ridge", "xgboost")], ignore_index=True)
    key = ["regime", "task", "representation"]
    winner = sel.loc[sel.groupby(key)["inner_mae"].idxmin(), key + ["regressor", "inner_mae"]]
    if winner.duplicated(key).any():
        raise ValueError("more than one best pipeline per task and representation")
    winner = winner.drop(columns="inner_mae")
    preds = pd.concat([pd.read_csv(f"{out}/{r}_predictions.csv") for r in ("ridge", "xgboost")], ignore_index=True)
    best = preds.merge(winner, on=key + ["regressor"])
    best.to_csv(f"{out}/best_predictions.csv", index=False)
    winner.to_csv(f"{out}/best_selection.csv", index=False)
    return best


def provisional_confirmatory(d: dict, best: pd.DataFrame) -> pd.DataFrame:
    """C1-C3: MAE(MoLFormer) - MAE(curated), both best pipelines. Negative =
    MoLFormer better. Raw p-values; Holm and the decision wait for C4 and C5."""
    wide = best.pivot_table(index=["regime", "task", "key"], columns="representation", values="y_pred").reset_index()
    wide = wide.merge(best[["key", "y_true"]].drop_duplicates(), on="key")
    rows = []
    fam = wide[wide["regime"] == "familiar"].assign(
        a=lambda w: (w["molformer"] - w["y_true"]).abs(), b=lambda w: (w["curated"] - w["y_true"]).abs())
    if len(fam):
        per_fold = fam.groupby("task")[["a", "b"]].mean()
        n_test = fam.groupby("task").size().mean()
        r = study_stats.corrected_resampled_ttest(per_fold["a"] - per_fold["b"], len(d["lit"]) - n_test, n_test)
        rows.append({"comparison": "C1", "regime": "familiar", "estimate": r["estimate"], "se": r["se"],
                     "ci_low": np.nan, "ci_high": np.nan, "p_raw": r["p"], "n_units": len(per_fold)})
    strata = {"holdout": lambda w: w["task"],
              "nist": lambda w: w["key"].map(dict(zip("nist:" + d["nist"].index.astype(str), d["nist"]["subgroup"])))}
    for comparison, regime in (("C2", "holdout"), ("C3", "nist")):
        w = wide[wide["regime"] == regime]
        if not len(w):
            continue
        r = study_stats.bootstrap_macro_difference((w["molformer"] - w["y_true"]).abs(),
                                                   (w["curated"] - w["y_true"]).abs(), strata[regime](w))
        rows.append({"comparison": comparison, "regime": regime, "estimate": r["estimate"], "se": np.nan,
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_raw": r["p"], "n_units": len(w),
                     "per_stratum": json.dumps(r["per_stratum"])})
    return pd.DataFrame(rows).assign(status="provisional: Holm and the 1.0 K rule applied once C4 and C5 exist")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    parser.add_argument("--smoke", action="store_true", help="1 task per regime, reduced grids, temporary output")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or (os.path.join(tempfile.gettempdir(), "representation_study_smoke") if args.smoke else OUT_DIR)
    start = time.time()
    d = load_data()
    tasks = make_tasks(d, args.smoke)
    print(f"{len(tasks)} tasks; {len(d['keys'])} molecules; output: {out}")
    if args.stage == "A":
        tables = stage_a(d, tasks, out, args.n_jobs, SMOKE_ALPHAS if args.smoke else study.ALPHAS)
        metrics = study_stats.fold_mae_table(tables["predictions"])
    else:
        stage_b(d, tasks, out, args.n_jobs, SMOKE_XGB if args.smoke else study.XGB_PARAM_GRID)
        best = best_pipelines(out)
        conf = provisional_confirmatory(d, best)
        conf.to_csv(f"{out}/confirmatory_provisional.csv", index=False)
        metrics = study_stats.fold_mae_table(best)
        print(conf.to_string(index=False))
    summary = metrics.groupby(["regime", "representation", "regressor"])[["mae", "rmse"]].agg(["mean", "std"]).round(2)
    print(summary.to_string())
    print(f"stage {args.stage} done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
