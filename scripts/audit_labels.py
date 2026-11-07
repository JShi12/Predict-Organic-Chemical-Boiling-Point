"""Audit the literature boiling-point labels (writes data/label_audit.csv).

Checks every label for being a Joback group-contribution estimate, every
hydrocarbon for physical plausibility, and stereoisomers of the same
structure for consistent labels, and records the NIST lookups made
for the compounds every model mispredicts in the same direction
(data/nist_audit_lookups.csv, produced once with
nist_scraper.lookup_boiling_point). Compounds marked exclude=True are left
out of the measured-labels version of the feature study
(scripts/run_feature_study.py --audited).

Usage:
    python scripts/audit_labels.py
"""

import pandas as pd

from boiling_point import audit, data

AUDIT_CSV = "data/label_audit.csv"
NIST_LOOKUPS_CSV = "data/nist_audit_lookups.csv"
# Found earlier by cross-referencing recurring outliers (Boiling_Point_Predictor.ipynb, Conclusions 4)
KNOWN_ERRORS = {
    "2,6-Nonadien-1-ol": "PubChem (WHO/FAO JECFA) gives 469.15 K",
    "N-Methyldodecylamine": "a supplier gives 473.15 K",
}
# Independent evidence for flagged labels that isn't decisive enough to exclude them
# (PubChem CID 8065 has no stereochemistry, so it doesn't say which isomer was measured)
PUBCHEM_8065 = ("PubChem CID 8065 (2-butene-1,4-diol, stereo unspecified): 234 °C at 760 mmHg = 507.15 K "
                "(DOE PAC database); 141-149 °C at 20 mmHg (HSDB)")
NOTES = {
    "cis-2-Butene-1,4-Diol": f"{PUBCHEM_8065}, which agrees with this label",
    "2-Butene-1,4-diol": f"{PUBCHEM_8065}; this label is 103 K lower, near the reduced-pressure range",
}


def main():
    literature = data.build_modelling_table(
        data.load_literature_data("data/compound_boiling_points_from_literature.xlsx"), data.load_pubchem_data())
    table = audit.build_label_audit(literature, pd.read_csv(NIST_LOOKUPS_CSV), KNOWN_ERRORS, notes=NOTES)
    table.to_csv(AUDIT_CSV, index=False)
    excluded = table["exclude"].sum()
    print(table.groupby(["category", "exclude"]).size().to_string())
    print(f"{excluded} of {len(literature)} labels excluded -> {len(literature) - excluded} measured labels; "
          f"written to {AUDIT_CSV}")


if __name__ == "__main__":
    main()
