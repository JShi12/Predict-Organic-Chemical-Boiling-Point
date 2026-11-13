"""Combining representations (representation study, Step 8, exploratory).

Stage concat: Ridge on curated (log) descriptors + one frozen MoLFormer layer,
each block weighted equally (stacking.BlockScaler); layer and alpha tuned
jointly in the inner CV, on the same tasks and inner folds as Step 3 (~20 s).

Stage stack: convex stacking weights per task, fitted on the saved
out-of-fold training predictions of the Step 3 pipelines (no refits):
  S0     curated Ridge + curated XGBoost (control: one representation)
  S1     best curated + best MoLFormer
  S1m    best curated + best Morgan (control: another second view)
  S2     all six frozen pipelines
  S1_abs S1 with absolute-error weights (sensitivity)
  S0_abs S0 with absolute-error weights (added after seeing S0: curated
         Ridge and XGBoost have about the same out-of-fold RMSE but very
         different MAE, so squared-error weights split them evenly and raise
         MAE; this keeps the control on the same loss as S1_abs)
  average  S1's two inputs with equal weights (reference; seen before planning)
then MAE per regime and paired contrasts with the C1-C3 tests (raw p only,
no Holm: exploratory). For the familiar regime, the 95% interval is
estimate +/- t(0.975, df) x SE from the corrected resampled t-test.

Writes results/representation_study/combined/. No torch.

    python scripts/run_combined.py --stage concat [--smoke]
    python scripts/run_combined.py --stage stack
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402

from boiling_point import stacking, study  # noqa: E402
from run_representation_study import (NIST_CSV, SMOKE_ALPHAS, SPLITS_CSV, load_data, make_tasks,  # noqa: E402
                                      provisional_confirmatory, records)

BASE = "results/representation_study"
FROZEN = f"{BASE}/frozen"
OUT = f"{BASE}/combined"
PIPELINES = [f"{r}_{g}" for r in study.REPRESENTATIONS for g in ("ridge", "xgboost")]
STACKS = ["S0", "S1", "S1m", "S2", "S1_abs", "S0_abs", "average"]
CONTRASTS = [("S0", "curated_best"), ("S1", "curated_best"), ("S1m", "curated_best"), ("S2", "curated_best"),
             ("S1_abs", "curated_best"), ("S0_abs", "curated_best"), ("average", "curated_best"), ("S1", "S0"), ("S1_abs", "S0_abs"),
             ("concat_ridge", "molformer_best"), ("concat_ridge", "curated_best")]


# --- stage concat -------------------------------------------------------------------

def stage_concat(smoke: bool, n_jobs: int, out: str):
    d = load_data()
    cur, mol = d["X"]["curated"], d["X"]["molformer"]
    X = np.stack([np.hstack([cur, mol[layer]]) for layer in range(mol.shape[0])])
    blocks = (cur.shape[1], mol.shape[2])
    tasks = make_tasks(d, smoke)
    alphas = SMOKE_ALPHAS if smoke else study.ALPHAS
    print(f"concat: {X.shape} (layers, molecules, columns), blocks {blocks}, {len(tasks)} tasks", flush=True)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(study.run_task)(t, "concat", "ridge", X, d["y"], d["families"], d["groups"], alphas=alphas,
                                blocks=blocks) for t in tasks)
    tables = records(results, d["keys"])
    os.makedirs(out, exist_ok=True)
    tables["predictions"].to_csv(f"{out}/concat_ridge_predictions.csv", index=False)
    tables["selection"].to_csv(f"{out}/concat_ridge_selection.csv", index=False)
    print(tables["selection"][["regime", "task", "layer", "params", "inner_mae"]].to_string(index=False))


# --- stage stack ----------------------------------------------------------------------

def load_frozen():
    preds = pd.concat([pd.read_csv(f"{FROZEN}/{g}_predictions.csv") for g in ("ridge", "xgboost")])
    resid = pd.concat([pd.read_csv(f"{FROZEN}/{g}_cvplus_residuals.csv.gz") for g in ("ridge", "xgboost")])
    for t in (preds, resid):
        t["pipeline"] = t["representation"] + "_" + t["regressor"]
    best = pd.read_csv(f"{FROZEN}/best_selection.csv")
    best["pipeline"] = best["representation"] + "_" + best["regressor"]
    return preds, resid, best.set_index(["regime", "task", "representation"])["pipeline"]


def training_keys(tasks_by_name: dict, keys: np.ndarray, task: str) -> set:
    return set(keys[tasks_by_name[task]["train"]])


def stack_task(regime, task, preds, resid, best, y_by_key, train_keys):
    """Stacked test predictions and weights for one task."""
    p = preds[(preds["regime"] == regime) & (preds["task"] == task)]
    test = p.pivot(index="key", columns="pipeline", values="y_pred")[PIPELINES]
    y_test = p.drop_duplicates("key").set_index("key").loc[test.index, "y_true"]
    oof = stacking.oof_matrix(resid[(resid["regime"] == regime) & (resid["task"] == task)], y_by_key, PIPELINES)
    if set(oof.index) != train_keys:
        raise ValueError(f"{task}: out-of-fold keys differ from the task's training rows")
    y_oof = y_by_key.loc[oof.index].to_numpy()
    b = {r: best.loc[(regime, task, r)] for r in study.REPRESENTATIONS}
    inputs = {"S0": (["curated_ridge", "curated_xgboost"], "squared"),
              "S1": ([b["curated"], b["molformer"]], "squared"),
              "S1m": ([b["curated"], b["morgan"]], "squared"),
              "S2": (PIPELINES, "squared"),
              "S1_abs": ([b["curated"], b["molformer"]], "absolute"),
              "S0_abs": (["curated_ridge", "curated_xgboost"], "absolute"),
              "average": ([b["curated"], b["molformer"]], None)}
    rows, weights = [], []
    base = {"regime": regime, "task": task}
    for method, (cols, loss) in inputs.items():
        w = np.full(len(cols), 1 / len(cols)) if loss is None else stacking.convex_weights(oof[cols], y_oof, loss)
        rows.append(pd.DataFrame({**base, "method": method, "key": test.index, "y_true": y_test.to_numpy(),
                                  "y_pred": test[cols].to_numpy() @ w}))
        oof_mae = float(np.mean(np.abs(oof[cols].to_numpy() @ w - y_oof)))
        weights += [{**base, "method": method, "pipeline": c, "weight": float(wi), "oof_mae": oof_mae}
                    for c, wi in zip(cols, w)]
    for r in study.REPRESENTATIONS:                 # the best single pipelines, for comparison
        rows.append(pd.DataFrame({**base, "method": f"{r}_best", "key": test.index, "y_true": y_test.to_numpy(),
                                  "y_pred": test[b[r]].to_numpy()}))
    return rows, weights


def metrics(all_preds: pd.DataFrame, nist_subgroup: dict) -> pd.DataFrame:
    """Per regime and method: MAE mean and SD over tasks (familiar folds,
    held-out families), and for NIST the pooled MAE and the macro-average over
    subgroups."""
    e = all_preds.assign(abs_err=(all_preds["y_pred"] - all_preds["y_true"]).abs())
    per_task = e.groupby(["regime", "task", "method"])["abs_err"].mean().reset_index()
    out = per_task.groupby(["regime", "method"])["abs_err"].agg(mae="mean", sd="std", n_tasks="size").reset_index()
    nist = e[e["regime"] == "nist"].assign(subgroup=lambda x: x["key"].map(nist_subgroup))
    macro = nist.groupby(["method", "subgroup"])["abs_err"].mean().groupby("method").mean()
    out["nist_macro_mae"] = out["method"].map(macro).where(out["regime"] == "nist")
    return out


def contrasts(all_preds, d) -> pd.DataFrame:
    frame = all_preds.rename(columns={"method": "representation"})
    rows = []
    for first, second in CONTRASTS:
        sub = frame[frame["representation"].isin([first, second])]
        r = provisional_confirmatory(d, sub, first, second).drop(columns="status")
        fam = r["comparison"] == "C1"
        df = len(sub[sub["regime"] == "familiar"]["task"].unique()) - 1
        half = t_dist.ppf(0.975, df) * r.loc[fam, "se"]
        r.loc[fam, "ci_low"] = r.loc[fam, "estimate"] - half
        r.loc[fam, "ci_high"] = r.loc[fam, "estimate"] + half
        r.insert(0, "contrast", f"{first} - {second}")
        r["test"] = np.where(fam, "corrected t", "bootstrap")
        rows.append(r.rename(columns={"comparison": "setting_as_in"}))
    out = pd.concat(rows, ignore_index=True)
    out["at_least_1K_better"] = (out["ci_high"] < 0) & (out["estimate"] <= -1.0)
    return out


def stage_stack(out: str):
    lit = pd.read_csv(SPLITS_CSV)
    nist_all = pd.read_csv(NIST_CSV)
    nist = nist_all[~nist_all["exclude"]]
    d = {"lit": lit, "nist": nist}
    keys = np.array(list("literature:" + lit["mol_id"].astype(str)) + list("nist:" + nist.index.astype(str)))
    tasks = {t["task"]: t for t in make_tasks({"lit": lit, "n_lit": len(lit), "nist": nist}, smoke=False)}
    preds, resid, best = load_frozen()
    y_by_key = preds.drop_duplicates("key").set_index("key")["y_true"]

    rows, weights = [], []
    for (regime, task), _ in preds.groupby(["regime", "task"]):
        r, w = stack_task(regime, task, preds, resid, best, y_by_key, training_keys(tasks, keys, task))
        rows += r
        weights += w
    stacked = pd.concat(rows, ignore_index=True)
    concat = pd.read_csv(f"{out}/concat_ridge_predictions.csv").assign(method="concat_ridge")
    all_preds = pd.concat([stacked, concat[["regime", "task", "method", "key", "y_true", "y_pred"]]],
                          ignore_index=True)

    # guard: the best single pipelines reproduce the Step 3 MAEs
    saved = pd.read_csv(f"{FROZEN}/best_predictions.csv")
    for r in study.REPRESENTATIONS:
        a = all_preds[all_preds["method"] == f"{r}_best"].sort_values(["regime", "task", "key"])
        s = saved[saved["representation"] == r].sort_values(["regime", "task", "key"])
        if not np.allclose(a["y_pred"].to_numpy(), s["y_pred"].to_numpy(), rtol=0, atol=1e-9):
            raise ValueError(f"{r}: best predictions differ from frozen/best_predictions.csv")

    subgroup = dict(zip("nist:" + nist.index.astype(str), nist["subgroup"]))
    m = metrics(all_preds, subgroup)
    c = contrasts(all_preds, d)
    os.makedirs(out, exist_ok=True)
    stacked.to_csv(f"{out}/stacked_predictions.csv", index=False)
    pd.DataFrame(weights).to_csv(f"{out}/weights.csv", index=False)
    m.to_csv(f"{out}/metrics.csv", index=False)
    c.to_csv(f"{out}/contrasts.csv", index=False)

    pd.set_option("display.width", 220)
    print("best single pipelines reproduce frozen/best_predictions.csv")
    print(m.pivot(index="method", columns="regime", values="mae").round(2).to_string())
    print(m[m["regime"] == "nist"][["method", "nist_macro_mae"]].round(2).to_string(index=False))
    w = pd.DataFrame(weights)
    print(w.groupby(["regime", "method", "pipeline"])["weight"].agg(["mean", "min", "max"]).round(3).to_string())
    print(c.drop(columns=["per_stratum"]).round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=["concat", "stack"], required=True)
    parser.add_argument("--smoke", action="store_true", help="concat: 1 task per regime, temporary output")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "concat":
        stage_concat(args.smoke, args.n_jobs, tempfile.mkdtemp() if args.smoke else OUT)
    else:
        stage_stack(OUT)


if __name__ == "__main__":
    main()
