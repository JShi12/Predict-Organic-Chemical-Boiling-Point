"""Descriptive Morgan contrasts for the representation study: Morgan - curated
and Morgan - MoLFormer (frozen, best pipelines) in the four settings of
C1-C4, with the same tests (corrected resampled t-test; stratified bootstrap).

Morgan counts were pre-registered as a descriptive baseline, so these rows
are outside the confirmatory family: raw p-values only, no Holm, no
"finding". Before writing, the script recomputes C1-C4 through the same code
and stops if they differ from the saved confirmatory values.

Writes results/representation_study/morgan_descriptive.csv.

    python scripts/morgan_descriptive.py
"""
import sys

sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402

from boiling_point import study_stats  # noqa: E402
from run_representation_study import NIST_CSV, SPLITS_CSV, provisional_confirmatory  # noqa: E402

BASE = "results/representation_study"
SETTING = {"C1": "familiar, 15 folds", "C2": "held-out families, macro-average of 4",
           "C3": "NIST external, macro-average of 3 subgroups", "C4": "familiar, n = 100, 5 folds x 3 draws"}
NAME = {"curated": "curated (log)", "morgan": "Morgan", "molformer": "frozen MoLFormer"}
PAIRS = [("morgan", "curated"), ("morgan", "molformer")]


def frozen_rows(d, best, first, second):
    rows = provisional_confirmatory(d, best, first, second)
    rows["test"] = np.where(rows["comparison"] == "C1", "corrected t", "bootstrap")
    return rows.drop(columns="status")


def n100_row(diffs, c4, first, second):
    d = diffs[diffs["size"] == "100"]
    a = d[f"{first}_minus_curated"] if first != "curated" else 0.0
    b = d[f"{second}_minus_curated"] if second != "curated" else 0.0
    r = study_stats.corrected_resampled_ttest(a - b, float(c4["n_train"].iloc[0]), float(c4["n_test"].iloc[0]))
    return pd.DataFrame([{"comparison": "C4", "regime": "learning_curve (n = 100)", "estimate": r["estimate"],
                          "se": r["se"], "ci_low": np.nan, "ci_high": np.nan, "p_raw": r["p"], "n_units": len(d),
                          "test": "corrected t", "min_detectable": float(r["se"] * t_dist.ppf(0.975, r["df"]))}])


def main():
    lit = pd.read_csv(SPLITS_CSV)
    nist = pd.read_csv(NIST_CSV)
    d = {"lit": lit, "nist": nist[~nist["exclude"]]}
    best = pd.read_csv(f"{BASE}/frozen/best_predictions.csv")
    diffs = pd.read_csv(f"{BASE}/learning_curve/differences.csv", dtype={"size": str})
    c4 = pd.read_csv(f"{BASE}/learning_curve/c4_provisional.csv")

    # guard: the shared code must reproduce C1-C4 exactly
    saved = pd.read_csv(f"{BASE}/confirmatory.csv").set_index("comparison")
    check = pd.concat([frozen_rows(d, best, "molformer", "curated"), n100_row(diffs, c4, "molformer", "curated")])
    for _, r in check.iterrows():
        for col in ("estimate", "p_raw"):
            if not np.isclose(r[col], saved.loc[r["comparison"], col], rtol=1e-9, atol=1e-12):
                raise ValueError(f"{r['comparison']} {col}: {r[col]} != saved {saved.loc[r['comparison'], col]}")

    parts = []
    for first, second in PAIRS:
        rows = pd.concat([frozen_rows(d, best, first, second), n100_row(diffs, c4, first, second)],
                         ignore_index=True)
        rows.insert(0, "contrast", f"{NAME[first]} - {NAME[second]}")
        parts.append(rows)
    out = pd.concat(parts, ignore_index=True)
    out = out.rename(columns={"comparison": "as_in"})
    out.insert(2, "setting", out["as_in"].map(SETTING))
    out["at_least_1K"] = out["estimate"].abs() >= 1.0
    out["direction"] = np.where(out["estimate"] > 0, "Morgan worse", "Morgan better")
    out = out[["contrast", "as_in", "setting", "test", "estimate", "se", "ci_low", "ci_high", "p_raw",
               "min_detectable", "n_units", "per_stratum", "at_least_1K", "direction"]]
    out.to_csv(f"{BASE}/morgan_descriptive.csv", index=False)
    pd.set_option("display.width", 220)
    print("C1-C4 reproduced from the shared code.")
    print(out.drop(columns=["setting", "per_stratum"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
