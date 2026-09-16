"""Scrapes normal boiling point data from the NIST Chemistry WebBook
(https://webbook.nist.gov) for candidate compounds not already in our
literature-sourced dataset, to help address the error analysis finding that
high polar-area/rotatable-bond compounds are underrepresented.

Not affiliated with, maintained by, or endorsed by NIST. Respects the
5-second crawl-delay from NIST's robots.txt (see CRAWL_DELAY_SECONDS).
"""
import csv
import time

import nistchempy as nist
import pandas as pd
import requests
from bs4 import BeautifulSoup

CRAWL_DELAY_SECONDS = 5.0
NIST_TBOIL_URL = "https://webbook.nist.gov/cgi/cbook.cgi?ID={compound_id}&Units=SI&Type=TBOIL"
CSV_FIELDS = ["cmpdname", "nist_id", "boiling_point_kelvin", "n_measurements", "status"]

# Suffixes typical of the simple, common organic compounds NIST WebBook's
# curated ~7000-compound database actually covers well (verified empirically:
# random PubChem names had a 0% hit rate, this filter got 25% on a general
# sample and ~4% on the harder high-polararea/rotbonds subset).
_SIMPLE_NAME_SUFFIXES = (
    "ol", "olamine", "amine", "diol", "triol", "glycol", "ane", "ene", "yne",
    "ether", "acid", "one", "al", "amide", "imine",
)


def select_targeted_candidates(pubchem_df: pd.DataFrame, already_have_names,
                                polararea_threshold: float = 40, rotbonds_threshold: float = 14,
                                max_name_length: int = 25, max_hyphens: int = 1) -> pd.Series:
    """Candidate compound names for scraping: not already in our dataset,
    high polar area and/or rotatable bonds (the underrepresented outlier
    region from the error analysis), and simple-looking names (biases
    toward names NIST WebBook is likely to actually have)."""
    candidates = pubchem_df[~pubchem_df["cmpdname"].isin(already_have_names)]
    suffix_pattern = "(?:" + "|".join(_SIMPLE_NAME_SUFFIXES) + ")$"
    targeted = candidates[
        ((candidates["polararea"] >= polararea_threshold) | (candidates["rotbonds"] >= rotbonds_threshold))
        & (~candidates["cmpdname"].str.contains(r"[,()\[\]]", regex=True, na=True))
        & (candidates["cmpdname"].str.len() <= max_name_length)
        & (candidates["cmpdname"].str.count("-") <= max_hyphens)
        & (~candidates["cmpdname"].str.match(r"^CID ", na=False))
        & (candidates["cmpdname"].str.contains(suffix_pattern, regex=True, case=False, na=False))
    ]
    return targeted["cmpdname"]


def parse_boiling_point_table(html: str) -> list:
    """Extract normal boiling point measurements (K) from a NIST WebBook
    Type=TBOIL page's HTML. Returns an empty list if the compound has no
    boiling point section."""
    soup = BeautifulSoup(html, "html.parser")
    header = soup.find(id="TBOIL")
    if header is None:
        return []
    table = header.find_next("table", class_="data")
    values = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if cells:
            try:
                values.append(float(cells[0].get_text(strip=True)))
            except ValueError:
                pass
    return values


def resolve_compound_id(name: str):
    """Look up a compound's NIST WebBook ID by name. Returns None if not found."""
    search = nist.run_search(name, "name", use_SI=True)
    if not search.compounds:
        return None
    compound = search.compounds[0]
    return getattr(compound, "ID", None) or getattr(compound, "compound_id", None)


def fetch_boiling_point(compound_id: str) -> list:
    """Fetch normal boiling point measurements (K) for a NIST compound ID."""
    response = requests.get(NIST_TBOIL_URL.format(compound_id=compound_id), timeout=15)
    return parse_boiling_point_table(response.text)


def scrape_boiling_points(names, output_path: str, crawl_delay: float = CRAWL_DELAY_SECONDS,
                           resume: bool = True) -> None:
    """Look up each name on NIST WebBook and append results to output_path
    one row at a time (flushed immediately), so progress survives
    interruption. If resume=True and output_path already has rows, names
    already recorded there are skipped."""
    already_done = set()
    if resume:
        try:
            already_done = set(pd.read_csv(output_path)["cmpdname"])
        except FileNotFoundError:
            pass

    write_header = not already_done
    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_FIELDS)

        for name in names:
            if name in already_done:
                continue
            try:
                compound_id = resolve_compound_id(name)
                time.sleep(crawl_delay)
                if compound_id is None:
                    row = [name, "", "", 0, "not_found"]
                else:
                    values = fetch_boiling_point(compound_id)
                    time.sleep(crawl_delay)
                    if values:
                        median = sorted(values)[len(values) // 2]
                        row = [name, compound_id, round(median, 2), len(values), "found"]
                    else:
                        row = [name, compound_id, "", 0, "no_boiling_point"]
            except Exception as e:
                row = [name, "", "", 0, f"error:{e}"]

            writer.writerow(row)
            f.flush()
