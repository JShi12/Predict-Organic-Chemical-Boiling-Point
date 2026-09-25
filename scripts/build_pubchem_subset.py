"""Build data/pubchem_subset.csv: every PubChem row this project uses.

The full PubChem download (data/compound_property_from_PubChem.csv, ~77 MB) is
too large to commit. Every notebook and script only needs:
- rows for compounds with a boiling point (literature + all NIST files), and
- rows passing the NIST-likely-name filter (the candidate pools of the
  heuristic and model-driven NIST rounds).
Rows keep their original order (the heuristic run's lookup order came from
it) and duplicate names are kept, so merges behave exactly as with the full
file. data.load_pubchem_data falls back to this subset when the full file
is missing.

Usage (needs the full file at the repo root):
    python scripts/build_pubchem_subset.py
"""
import sys

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from boiling_point import data, nist_scraper  # noqa: E402

LABEL_FILES = [
    "data/compound_boiling_points_from_nist.csv",
    "data/nist_boiling_points_targeted.csv",
    "data/nist_boiling_points_uncertainty_only.csv",
    "data/nist_boiling_points_feasibility_aware.csv",
]


def main():
    full = pd.read_csv(data.PUBCHEM_FULL_PATH)
    names = set(data.load_literature_data("data/compound_boiling_points_from_literature.xlsx")["cmpdname"])
    for path in LABEL_FILES:
        names |= set(pd.read_csv(path)["cmpdname"])
    candidates = nist_scraper.select_nist_likely_candidates(full, already_have_names=set())
    keep = full["cmpdname"].isin(names) | full.index.isin(candidates.index)
    subset = full.loc[keep, ["cid"] + data.PUBCHEM_COLUMNS]
    subset.to_csv(data.PUBCHEM_SUBSET_PATH, index=False)
    print(f"{len(subset)} of {len(full)} rows written to {data.PUBCHEM_SUBSET_PATH}")


if __name__ == "__main__":
    main()
