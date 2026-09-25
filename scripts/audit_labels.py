"""Audit the literature boiling-point labels (writes data/label_audit.csv).

Checks every label for being a Joback group-contribution estimate and every
hydrocarbon for physical plausibility, and records the NIST lookups made
for the compounds every model mispredicts in the same direction
(data/nist_audit_lookups.csv, produced once with
nist_scraper.lookup_boiling_point). Compounds marked exclude=True are left
out of the measured-labels version of the feature study
(scripts/run_feature_study.py --audited).

Usage:
    python scripts/audit_labels.py
"""
import sys

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from boiling_point import audit, data  # noqa: E402

AUDIT_CSV = "data/label_audit.csv"
NIST_LOOKUPS_CSV = "data/nist_audit_lookups.csv"
# Found earlier by cross-referencing recurring outliers (Boiling_Point_Predictor.ipynb, Conclusions 4)
KNOWN_ERRORS = {
    "2,6-Nonadien-1-ol": "PubChem (WHO/FAO JECFA) gives 469.15 K",
    "N-Methyldodecylamine": "a supplier gives 473.15 K",
}


def main():
    literature = data.build_modelling_table(
        data.load_literature_data("compound_boiling_points_from_literature.xlsx"), data.load_pubchem_data())
    table = audit.build_label_audit(literature, pd.read_csv(NIST_LOOKUPS_CSV), KNOWN_ERRORS)
    table.to_csv(AUDIT_CSV, index=False)
    excluded = table["exclude"].sum()
    print(table.groupby(["category", "exclude"]).size().to_string())
    print(f"{excluded} of {len(literature)} labels excluded -> {len(literature) - excluded} measured labels; "
          f"written to {AUDIT_CSV}")


if __name__ == "__main__":
    main()
