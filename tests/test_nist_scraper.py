import pandas as pd

from boiling_point.nist_scraper import parse_boiling_point_table, select_targeted_candidates

# Trimmed excerpt of a real NIST WebBook Type=TBOIL page's structure
# (verified against Thianthrene, https://webbook.nist.gov/cgi/cbook.cgi?ID=C92853&Units=SI&Type=TBOIL).
_SAMPLE_TBOIL_HTML = """
<html><body>
<h2 id="TBOIL">Normal boiling point</h2>
<table class="data" aria-label="Normal boiling point"><tr>
<th scope="col">Tboil (K)</th><th scope="col">Reference</th><th scope="col">Comment</th></tr>
<tr class="exp"><td class="right-nowrap">639.2</td><td>Weast and Grasselli, 1989</td><td>BS</td></tr>
</table>
</body></html>
"""

_SAMPLE_NO_TBOIL_HTML = """
<html><body>
<h2 id="Refs">References</h2>
</body></html>
"""


def test_parse_boiling_point_table_extracts_values():
    values = parse_boiling_point_table(_SAMPLE_TBOIL_HTML)

    assert values == [639.2]


def test_parse_boiling_point_table_returns_empty_when_no_section():
    values = parse_boiling_point_table(_SAMPLE_NO_TBOIL_HTML)

    assert values == []


def _synthetic_pubchem_df():
    return pd.DataFrame({
        "cmpdname": [
            "Simple Diol",           # already have this one
            "Undecaethylene glycol", # high polararea, simple name -> candidate
            "Low Polarity Alkane",   # low polararea/rotbonds -> excluded
            "CID 12345",             # placeholder name -> excluded
            "Complex, Ester (name)", # punctuation -> excluded
            "Very-Long-Hyphenated-Chemical-Name-ol",  # too many hyphens -> excluded
        ],
        "polararea": [50.0, 60.0, 5.0, 60.0, 60.0, 60.0],
        "rotbonds": [3, 20, 1, 20, 20, 20],
    })


def test_select_targeted_candidates_applies_all_filters():
    df = _synthetic_pubchem_df()

    result = select_targeted_candidates(df, already_have_names={"Simple Diol"})

    assert list(result) == ["Undecaethylene glycol"]
