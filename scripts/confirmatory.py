"""Final confirmatory table for the representation study: C1-C5 with Holm
(1979) at alpha = 0.05 and the 1.0 K minimum meaningful difference.

Reads the provisional results of Step 3 (C1-C3), Step 6 (C4) and Step 4 (C5);
writes results/representation_study/confirmatory.csv.

    python scripts/confirmatory.py
"""
import numpy as np
import pandas as pd

from boiling_point import study_stats

BASE = "results/representation_study"
HYPOTHESIS = {"C1": "H1", "C2": "H2", "C3": "H3", "C4": "H4", "C5": "H5"}
CONTRAST = {"C1": "frozen MoLFormer - curated (log), familiar, 15 folds",
            "C2": "frozen MoLFormer - curated (log), held-out families, macro-average of 4",
            "C3": "frozen MoLFormer - curated (log), NIST external, macro-average of 3 subgroups",
            "C4": "frozen MoLFormer - curated (log), familiar, n = 100, 5 folds x 3 draws",
            "C5": "fine-tuned - frozen MoLFormer, familiar, 15 folds"}


def main():
    frozen = pd.read_csv(f"{BASE}/frozen/confirmatory_provisional.csv")
    c4 = pd.read_csv(f"{BASE}/learning_curve/c4_provisional.csv")
    c5 = pd.read_csv(f"{BASE}/finetune/c5_provisional.csv")
    cols = ["comparison", "estimate", "se", "ci_low", "ci_high", "p_raw"]
    table = pd.concat([frozen.reindex(columns=cols), c4.reindex(columns=cols), c5.reindex(columns=cols)],
                      ignore_index=True)
    table["test"] = np.where(table["comparison"].isin(["C2", "C3"]), "bootstrap", "corrected t")
    out = study_stats.confirmatory_table(table)
    out.insert(1, "hypothesis", out["comparison"].map(HYPOTHESIS))
    out.insert(2, "contrast", out["comparison"].map(CONTRAST))
    out["direction"] = np.where(out["estimate"] > 0, "first worse", "first better")
    out.to_csv(f"{BASE}/confirmatory.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns="contrast").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
