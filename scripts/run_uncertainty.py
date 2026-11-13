"""Prediction intervals and uncertainty ranking (representation study, Step 5).

No model is retrained. Frozen models (the best pipeline per task for curated,
Morgan and MoLFormer): 90% K-fold CV+ intervals from the inner-fold models
saved in Step 3, and, in the familiar regime, Mondrian CV+ by family.
Fine-tuned MoLFormer: 90% split-conformal intervals from the calibration half
saved in Step 4, around the 3-seed ensemble mean (constant and normalised
width) and around seed 0 alone. Per-molecule uncertainty: the SD of the inner
models' (frozen) or seeds' (fine-tuned) predictions; baseline score: distance
to the training set (1 - highest count-Tanimoto similarity, Morgan counts).

    python scripts/run_uncertainty.py
writes results/representation_study/uncertainty/. Descriptive only.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, "scripts")
from boiling_point import conformal, representations as rep  # noqa: E402
import run_finetuning  # noqa: E402  (shared loader and tasks; imports no torch)

ALPHA = 0.1
FROZEN = "results/representation_study/frozen"
FINETUNE = "results/representation_study/finetune"
OUT = "results/representation_study/uncertainty"
FRACTIONS = np.round(np.arange(0.1, 1.01, 0.1), 2)
MIN_GROUP = 10        # smallest family/subgroup for a within-group Spearman


def frozen_intervals(d, tasks, group_of, distance):
    best = pd.read_csv(f"{FROZEN}/best_selection.csv")
    preds = pd.read_csv(f"{FROZEN}/best_predictions.csv")
    cvt = pd.concat([pd.read_csv(f"{FROZEN}/{r}_cvplus_test.csv.gz") for r in ("ridge", "xgboost")])
    cvr = pd.concat([pd.read_csv(f"{FROZEN}/{r}_cvplus_residuals.csv.gz") for r in ("ridge", "xgboost")])
    cvt, cvr = (x.set_index(["task", "representation", "regressor"]).sort_index() for x in (cvt, cvr))
    preds = preds.set_index(["task", "representation"]).sort_index()
    rows = []
    for b in best.itertuples():
        t = tasks[b.task]
        keys = d["keys"][t["test"]]
        p = preds.loc[(b.task, b.representation)].set_index("key").loc[keys]
        inner = cvt.loc[(b.task, b.representation, b.regressor)].pivot(index="inner_fold", columns="key",
                                                                       values="y_pred")[keys].to_numpy()
        res = cvr.loc[(b.task, b.representation, b.regressor)]
        fold, r = res["inner_fold"].to_numpy(), res["residual"].abs().to_numpy()
        base = {"regime": b.regime, "task": b.task, "method": f"{b.representation}_frozen",
                "regressor": b.regressor, "key": keys, "group": [group_of[k] for k in keys],
                "y_true": p["y_true"].to_numpy(), "y_pred": p["y_pred"].to_numpy(),
                "uncertainty": inner.std(axis=0, ddof=1), "train_distance": distance[b.task]}
        lo, hi = conformal.cv_plus_interval(inner, fold, r, ALPHA)
        rows.append(pd.DataFrame({**base, "variant": "cvplus", "lower": lo, "upper": hi}))
        if b.regime == "familiar":
            res_group = np.array([group_of[k] for k in res["key"]])
            test_group = np.array(base["group"])
            mlo, mhi = lo.copy(), hi.copy()
            for g in np.unique(test_group):
                sel, train_sel = test_group == g, res_group == g
                glo, ghi = conformal.cv_plus_interval(inner[:, sel], fold[train_sel], r[train_sel], ALPHA)
                if np.isfinite(glo).all() and np.isfinite(ghi).all():   # else keep the pooled interval
                    mlo[sel], mhi[sel] = glo, ghi
            rows.append(pd.DataFrame({**base, "variant": "cvplus_mondrian", "lower": mlo, "upper": mhi}))
    return rows


def finetuned_intervals(d, tasks, group_of, distance):
    pred = pd.read_csv(f"{FINETUNE}/predictions.csv").query("inference == 'canonical'")
    cal = pd.read_csv(f"{FINETUNE}/calibration.csv")
    rows = []
    for task, tp in pred.groupby("task"):
        t = tasks[task]
        if t["regime"] == "learning_curve":
            continue
        keys = d["keys"][t["test"]]
        test = tp.pivot(index="key", columns="seed", values="y_pred").loc[keys]
        c = cal[cal["task"] == task].pivot(index="key", columns="seed", values="y_pred")
        c_true = cal[cal["task"] == task].drop_duplicates("key").set_index("key").loc[c.index, "y_true"].to_numpy()
        y_true = tp.drop_duplicates("key").set_index("key").loc[keys, "y_true"].to_numpy()
        base = {"regime": t["regime"], "task": task, "regressor": "head", "key": keys,
                "group": [group_of[k] for k in keys], "y_true": y_true, "train_distance": distance[task]}
        m, s = test.mean(axis=1).to_numpy(), test.std(axis=1, ddof=1).to_numpy()
        cm, cs = c.mean(axis=1).to_numpy(), c.std(axis=1, ddof=1).to_numpy()
        w = conformal.split_conformal_width(np.abs(c_true - cm), ALPHA)
        wn = conformal.normalised_split_conformal(np.abs(c_true - cm), cs, s, ALPHA)
        ens = {**base, "method": "molformer_finetuned_ensemble", "y_pred": m, "uncertainty": s}
        rows.append(pd.DataFrame({**ens, "variant": "split", "lower": m - w, "upper": m + w}))
        rows.append(pd.DataFrame({**ens, "variant": "split_normalised", "lower": m - wn, "upper": m + wn}))
        m0 = test[0].to_numpy()
        w0 = conformal.split_conformal_width(np.abs(c_true - c[0].to_numpy()), ALPHA)
        rows.append(pd.DataFrame({**base, "method": "molformer_finetuned_seed0", "y_pred": m0,
                                  "uncertainty": np.nan, "variant": "split", "lower": m0 - w0, "upper": m0 + w0}))
    return rows


def coverage_table(iv):
    iv = iv.assign(covered=(iv["y_true"] >= iv["lower"]) & (iv["y_true"] <= iv["upper"]),
                   width=iv["upper"] - iv["lower"])
    keys = ["regime", "task", "method", "variant"]
    overall = iv.groupby(keys).agg(n=("covered", "size"), coverage=("covered", "mean"), width=("width", "mean"))
    by_group = iv.groupby(keys + ["group"]).agg(n=("covered", "size"), coverage=("covered", "mean"),
                                                width=("width", "mean"))
    return pd.concat([overall.assign(group="all").set_index("group", append=True), by_group]).reset_index()


def spearman_table(iv):
    iv = iv[iv["variant"].isin(["cvplus", "split"])].assign(abs_error=lambda x: (x["y_pred"] - x["y_true"]).abs())
    rows = []
    for (regime, task, method), g in iv.groupby(["regime", "task", "method"]):
        scopes = [("all", g)] + [(name, gg) for name, gg in g.groupby("group") if len(gg) >= MIN_GROUP]
        for scope, gg in scopes:
            for score in ("uncertainty", "train_distance"):
                if gg[score].isna().all():
                    continue
                rho = spearmanr(gg[score], gg["abs_error"])[0]
                rows.append({"regime": regime, "task": task, "method": method, "scope": scope,
                             "score": "model" if score == "uncertainty" else "distance", "n": len(gg),
                             "spearman": rho})
    return pd.DataFrame(rows)


def retention_table(iv, seed: int = 0):
    iv = iv[(iv["regime"] == "familiar") & iv["variant"].isin(["cvplus", "split"])
            & (iv["method"] != "molformer_finetuned_seed0")]
    rng = np.random.default_rng(seed)
    rows = []
    for (task, method), g in iv.groupby(["task", "method"]):
        e = (g["y_pred"] - g["y_true"]).abs().to_numpy()
        scores = {"model": g["uncertainty"].to_numpy(), "distance": g["train_distance"].to_numpy(),
                  "oracle": e, "random": rng.permutation(len(e)).astype(float)}
        for name, u in scores.items():
            for f, mae in zip(FRACTIONS, conformal.retention_curve(u, e, FRACTIONS)):
                rows.append({"task": task, "method": method, "score": name, "fraction": f, "mae": mae})
    return pd.DataFrame(rows)


def main():
    d = run_finetuning.load()
    tasks = {t["task"]: t for t in run_finetuning.tasks(d)}
    group_of = dict(zip(d["keys"], list(d["lit"]["family"]) + list(d["nist"]["subgroup"])))
    fp = rep.morgan_counts(d["smiles"])
    distance = {name: conformal.count_tanimoto_distance(fp[t["test"]], fp[t["train"]]) for name, t in tasks.items()
                if t["regime"] != "learning_curve"}
    iv = pd.concat(frozen_intervals(d, tasks, group_of, distance) + finetuned_intervals(d, tasks, group_of, distance),
                   ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    iv.to_csv(f"{OUT}/intervals.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})  # byte-reproducible
    cov = coverage_table(iv)
    cov.to_csv(f"{OUT}/coverage.csv", index=False)
    sp = spearman_table(iv)
    sp.to_csv(f"{OUT}/spearman.csv", index=False)
    retention_table(iv).to_csv(f"{OUT}/retention.csv", index=False)

    pd.set_option("display.width", 200)
    allg = cov[cov["group"] == "all"]
    print(allg.groupby(["regime", "method", "variant"])[["coverage", "width"]].agg(["mean", "std"]).round(3))
    print(sp[sp["scope"] == "all"].groupby(["regime", "method", "score"])["spearman"].agg(["mean", "std"]).round(2))


if __name__ == "__main__":
    main()
