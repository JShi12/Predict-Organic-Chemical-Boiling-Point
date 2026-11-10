"""Learning curves and C4 (representation study, Step 6).

45 tasks: the 5 repeat-0 folds x 3 draws x sizes 100 / 250 / 500, each
training on its saved nested subset and testing on the unchanged fold. Step
3's pipelines: for curated (log), Morgan counts and frozen MoLFormer, Ridge
(MoLFormer's layer chosen by Ridge in the inner CV) then XGBoost on that
layer; the best pipeline per task and representation has the lower inner-CV
MAE. The full training fold is Step 3's repeat-0 result.

Also: C4 (best MoLFormer minus best curated at n = 100, corrected resampled
t-test over the 15 fold x draw runs, raw p) and the n = 250 half of H5
(fine-tuned minus frozen MoLFormer, matched and unmatched).

Each (task, representation, regressor) result is saved as it finishes, so the
script can be restarted. No torch (see representations.py).

    python scripts/run_learning_curves.py [--smoke] [--n-jobs 8]
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
from scipy.stats import t as t_dist  # noqa: E402

from boiling_point import study, study_stats  # noqa: E402
import run_finetuning  # noqa: E402  (n = 250 fitting rows; imports no torch)
import run_representation_study as rrs  # noqa: E402

OUT_DIR = "results/representation_study/learning_curve"
FROZEN = "results/representation_study/frozen"
FINETUNE = "results/representation_study/finetune"
SIZES = (100, 250, 500)


def run_path(out, task, representation, regressor):
    return f"{out}/runs/{task}__{representation}__{regressor}.json"


def run_job(d, t, representation, regressor, out, layer=None, alphas=study.ALPHAS, xgb_grid=study.XGB_PARAM_GRID):
    path = run_path(out, t["task"], representation, regressor)
    if os.path.exists(path):
        return path
    r = study.run_task(t, representation, regressor, d["X"][representation], d["y"], d["families"], d["groups"],
                       layer=layer, alphas=alphas, xgb_param_grid=xgb_grid)
    record = {"regime": "learning_curve", "task": t["task"], "fold": t["fold"], "size": t["size"], "draw": t["draw"],
              "n_train": len(t["train"]), "representation": representation, "regressor": regressor,
              "layer": r["layer"], "params": r["params"], "inner_mae": r["inner_mae"], "seconds": r["seconds"],
              "key": d["keys"][t["test"]].tolist(), "y_true": r["y_true"].tolist(), "y_pred": r["pred"].tolist()}
    with open(path + ".tmp", "w") as f:
        json.dump(record, f)
    os.replace(path + ".tmp", path)
    return path


def train(d, tasks, out, n_jobs, smoke):
    os.makedirs(f"{out}/runs", exist_ok=True)
    alphas = rrs.SMOKE_ALPHAS if smoke else study.ALPHAS
    grid = rrs.SMOKE_XGB if smoke else study.XGB_PARAM_GRID
    Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_job)(d, t, r, "ridge", out, alphas=alphas)
                                       for t in tasks for r in study.REPRESENTATIONS)
    layer = {t["task"]: json.load(open(run_path(out, t["task"], "molformer", "ridge")))["layer"] for t in tasks}
    jobs = sorted([(t, r) for t in tasks for r in study.REPRESENTATIONS], key=lambda j: j[1] != "molformer")
    Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_job)(d, t, r, "xgboost", out, layer=layer[t["task"]] if r == "molformer" else None,
                         xgb_grid=grid) for t, r in jobs)


def fold_mae(pred, by):
    return pred.assign(ae=(pred["y_pred"] - pred["y_true"]).abs()).groupby(by)["ae"].mean()


def collect(d, tasks, out):
    runs = [json.load(open(f"{out}/runs/{f}")) for f in sorted(os.listdir(f"{out}/runs")) if f.endswith(".json")]
    meta = ["regime", "task", "fold", "size", "draw", "n_train", "representation", "regressor"]
    selection = pd.DataFrame([{**{k: r[k] for k in meta}, "layer": r["layer"], "params": json.dumps(r["params"]),
                               "inner_mae": r["inner_mae"], "seconds": r["seconds"]} for r in runs])
    predictions = pd.concat([pd.DataFrame({**{k: r[k] for k in meta}, "key": r["key"], "y_true": r["y_true"],
                                           "y_pred": r["y_pred"]}) for r in runs], ignore_index=True)
    key = ["task", "representation"]
    best_sel = selection.loc[selection.groupby(key)["inner_mae"].idxmin()]
    if best_sel.duplicated(key).any():
        raise ValueError("more than one best pipeline per task and representation")
    best = predictions.merge(best_sel[key + ["regressor"]], on=key + ["regressor"])
    selection.to_csv(f"{out}/selection.csv", index=False)
    predictions.to_csv(f"{out}/predictions.csv", index=False)
    best_sel[key + ["regressor", "layer", "inner_mae"]].to_csv(f"{out}/best_selection.csv", index=False)
    best.to_csv(f"{out}/best_predictions.csv", index=False)

    # training-mean reference at every size
    ref = pd.concat([pd.DataFrame({"task": t["task"], "fold": t["fold"], "size": t["size"], "draw": t["draw"],
                                   "key": d["keys"][t["test"]], "y_true": d["y"][t["test"]],
                                   "y_pred": d["y"][t["train"]].mean()}) for t in tasks], ignore_index=True)
    ref.to_csv(f"{out}/references.csv", index=False)

    # curves: per run MAE, for the best pipeline and each regressor; the full size from Step 3 (repeat 0)
    rows = []
    for regressor, table in [("best", best)] + [(r, predictions[predictions["regressor"] == r])
                                                for r in ("ridge", "xgboost")]:
        m = fold_mae(table, ["size", "fold", "draw", "representation"]).rename("mae").reset_index()
        rows.append(m.assign(regressor=regressor))
    full = {"best": pd.read_csv(f"{FROZEN}/best_predictions.csv"),
            "ridge": pd.read_csv(f"{FROZEN}/ridge_predictions.csv"),
            "xgboost": pd.read_csv(f"{FROZEN}/xgboost_predictions.csv")}
    for regressor, table in full.items():
        table = table[(table["regime"] == "familiar") & (table["repeat"] == 0)]
        m = fold_mae(table, ["fold", "representation"]).rename("mae").reset_index()
        rows.append(m.assign(size="full", draw=0, regressor=regressor))
    ref_mae = fold_mae(ref, ["size", "fold", "draw"]).rename("mae").reset_index()
    full_ref = pd.read_csv(f"{FROZEN}/references.csv").query("regime == 'familiar' and repeat == 0 and "
                                                            "reference == 'training mean'")
    rows.append(ref_mae.assign(representation="training mean", regressor="reference"))
    rows.append(fold_mae(full_ref, ["fold"]).rename("mae").reset_index()
                .assign(size="full", draw=0, representation="training mean", regressor="reference"))
    curves = pd.concat(rows, ignore_index=True)
    curves["size"] = curves["size"].astype(str)
    curves.to_csv(f"{out}/curves.csv", index=False)

    # paired differences against curated (best pipelines), per run
    b = curves[curves["regressor"] == "best"].pivot_table(index=["size", "fold", "draw"], columns="representation",
                                                          values="mae")
    diffs = pd.DataFrame({f"{r}_minus_curated": b[r] - b["curated"] for r in ("molformer", "morgan")}).reset_index()
    diffs.to_csv(f"{out}/differences.csv", index=False)

    # C4: n = 100
    c4 = diffs[diffs["size"] == "100"]
    if len(c4) >= 2:
        n_train = selection.loc[selection["size"] == 100, "n_train"].mean()
        n_test = best[best["size"] == 100].groupby("task").size().mean() / len(study.REPRESENTATIONS)
        res = study_stats.corrected_resampled_ttest(c4["molformer_minus_curated"], n_train, n_test)
        pd.DataFrame([{"comparison": "C4", "regime": "learning_curve (n = 100)", "estimate": res["estimate"],
                       "se": res["se"], "t": res["t"], "df": res["df"], "p_raw": res["p"], "n_units": len(c4),
                       "n_train": n_train, "n_test": n_test,
                       "min_detectable": float(res["se"] * t_dist.ppf(0.975, res["df"])),
                       "status": "provisional: Holm and the 1.0 K rule in confirmatory.csv"}]).to_csv(
            f"{out}/c4_provisional.csv", index=False)
        print(f"C4 (MoLFormer - curated, n = 100, {len(c4)} runs): {res['estimate']:+.2f} K, SE {res['se']:.2f}, "
              f"p = {res['p']:.3f}")

    h5_n250(d, best, out)
    s = curves.groupby(["regressor", "representation", "size"])["mae"].agg(["mean", "std", "size"]).round(2)
    print(s.to_string())


def h5_n250(d, best, out):
    """Fine-tuned minus frozen MoLFormer at n = 250 (draw 0) and at full size,
    per repeat-0 fold: against the frozen best pipeline, and matched (frozen
    MoLFormer + Ridge refitted on the fine-tuned model's fitting rows)."""
    if not os.path.exists(f"{FINETUNE}/predictions.csv"):
        return
    ft = pd.read_csv(f"{FINETUNE}/predictions.csv").query("inference == 'canonical'")
    ft_mae = fold_mae(ft, ["task", "seed"]).groupby("task").mean()             # single-seed MAE, mean over seeds
    rf = run_finetuning.load()
    tasks = {t["task"]: t for t in run_finetuning.tasks(rf)}
    rows = []
    for k in range(5):
        lc, full = f"n250_f{k}", f"r0f{k}"
        t = tasks[lc]
        frozen = best[(best["task"] == f"n250_f{k}_d0") & (best["representation"] == "molformer")]
        if lc not in ft_mae.index or frozen.empty:      # (e.g. a smoke run)
            continue
        m = study.run_task({**t, "train": t["fit"]}, "molformer", "ridge", d["X"]["molformer"], d["y"],
                           d["families"], d["groups"])
        matched_250 = float(np.mean(np.abs(m["pred"] - m["y_true"])))
        frozen_250 = fold_mae(frozen, ["task"]).iloc[0]
        rows.append({"size": "250", "fold": k, "finetuned": ft_mae[lc], "frozen_best": frozen_250,
                     "frozen_matched": matched_250})
    fb = pd.read_csv(f"{FROZEN}/best_predictions.csv").query("representation == 'molformer' and regime == 'familiar'")
    fm = pd.read_csv(f"{FINETUNE}/frozen_matched_predictions.csv")
    for k in range(5):
        full = f"r0f{k}"
        rows.append({"size": "full", "fold": k, "finetuned": ft_mae[full],
                     "frozen_best": fold_mae(fb[fb["task"] == full], ["task"]).iloc[0],
                     "frozen_matched": fold_mae(fm[fm["task"] == full], ["task"]).iloc[0]})
    h5 = pd.DataFrame(rows).assign(minus_best=lambda x: x["finetuned"] - x["frozen_best"],
                                   minus_matched=lambda x: x["finetuned"] - x["frozen_matched"])
    h5.to_csv(f"{out}/h5_finetuned_vs_frozen.csv", index=False)
    print(h5.groupby("size")[["finetuned", "frozen_best", "frozen_matched", "minus_best", "minus_matched"]]
          .agg(["mean", "std"]).round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smoke", action="store_true", help="1 task, reduced grids, temporary output")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    out = os.path.join(tempfile.gettempdir(), "learning_curve_smoke") if args.smoke else OUT_DIR
    start = time.time()
    d = rrs.load_data()
    tasks = study.learning_curve_tasks(d["lit"], SIZES)
    if args.smoke:
        tasks = [t for t in tasks if t["task"] == "n100_f0_d0"]
    print(f"{len(tasks)} tasks; output: {out}", flush=True)
    if not args.collect_only:
        train(d, tasks, out, args.n_jobs, args.smoke)
    collect(d, tasks, out)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
