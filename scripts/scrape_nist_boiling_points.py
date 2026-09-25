"""Scrape NIST WebBook for boiling points of candidate compounds (from our
PubChem CSV) with high polar area and/or rotatable bonds -- the outlier
region flagged in the notebook's error analysis as underrepresented in our
current 1,588-compound dataset.

This is a long-running, network-bound script (NIST's crawl-delay is 5s, and
most candidates need 1-2 requests each), expected to take several hours for
the full targeted candidate pool. Progress is written to the output CSV
incrementally, and reruns automatically resume by skipping names already
recorded there.

Usage:
    python scripts/scrape_nist_boiling_points.py [output_csv]
"""
import sys

sys.path.insert(0, "src")

from boiling_point import data, nist_scraper  # noqa: E402

DEFAULT_OUTPUT = "data/nist_boiling_points_targeted.csv"


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

    df1 = data.load_literature_data("data/compound_boiling_points_from_literature.xlsx")
    df2 = data.load_pubchem_data("data/compound_property_from_PubChem.csv")
    already_have = set(df1["cmpdname"])

    candidates = nist_scraper.select_targeted_candidates(df2, already_have)
    print(f"{len(candidates)} targeted candidates (high polar area / rotatable bonds, simple names)")
    print(f"Writing results to {output_path} (resumable if interrupted)")

    nist_scraper.scrape_boiling_points(candidates.tolist(), output_path)

    print("Done.")


if __name__ == "__main__":
    main()
