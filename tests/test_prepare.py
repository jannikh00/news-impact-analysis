import re

import pandas as pd

from prepare_data import IRRELEVANT_HEADLINE, excluded_rows, load_cik_overrides

EXCLUSIONS = pd.DataFrame({"news_symbol": ["P", "COR", "DOW"], "start": [None, None, "2019-01-01"],
                           "end": [None, "2023-08-29", "2019-04-01"], "reason": ["", "", ""]})


def test_excluded_rows_respects_open_and_closed_ranges():
    chunk = pd.DataFrame({
        "symbol": ["P", "COR", "COR", "DOW", "DOW", "DOW", "KR"],
        "date": ["2014-01-02 00:00:00", "2016-05-04 00:00:00", "2023-08-30 00:00:00",
                 "2018-12-31 00:00:00", "2019-02-01 10:00:00-04:00", "2019-04-02 00:00:00",
                 "2014-01-02 00:00:00"],
    })
    assert excluded_rows(chunk, EXCLUSIONS).tolist() == [True, True, False, False, True, False, False]


def test_market_research_spam_pattern():
    assert re.search(IRRELEVANT_HEADLINE, "Bottle Jack Market Size, Share —…—… Industry Analysis")
    assert re.search(IRRELEVANT_HEADLINE, "Endoscope Disinfectors Market Share,Size Global Statistics")
    assert not re.search(IRRELEVANT_HEADLINE, "Kroger Gains Market Share In Grocery")


def test_cik_overrides(tmp_path):
    (tmp_path / "sec_cik_overrides.csv").write_text(
        "symbol,cik,filed_from,filed_to,note\nVTRS,69499,,2015-03-31,a\nVTRS,1792044,2020-11-16,,b\n")
    assert load_cik_overrides(tmp_path) == {"VTRS": [(69499, None, "2015-03-31"), (1792044, "2020-11-16", None)]}


def _filings(rows):
    return pd.DataFrame(rows, columns=["filed", "accn", "end", "val"])


def test_sec_filing_outliers_dropped_but_step_changes_kept():
    from prepare_data import sec_shares
    days = pd.bdate_range("2012-01-02", "2014-12-31")
    filings = _filings([
        ("2012-02-01", "a1", "2012-01-20", 100e6),
        ("2012-05-01", "a2", "2012-04-20", 101e6),
        ("2012-08-01", "a3", "2012-07-20", 707e12),   # typo (AMD 2012)
        ("2012-11-01", "a4", "2012-10-20", 102e6),
        ("2013-02-01", "a5", "2013-01-20", 500e6),    # genuine issuance (PG&E-style step)
        ("2013-05-01", "a6", "2013-04-20", 505e6),
        ("2013-08-01", "a7", "2013-07-20", 12345),    # placeholder (PG&E 2010)
        ("2013-11-01", "a8", "2013-10-20", 510e6),
    ])
    s, dropped = sec_shares(filings, pd.Series(dtype=float), days)
    assert dropped == 2
    assert s.loc["2012-09-03"] == 101e6          # typo filing ignored, previous count carried
    assert s.loc["2013-03-01"] == 500e6          # step change kept
    assert s.loc["2013-09-03"] == 505e6          # placeholder ignored


def test_sec_split_restatement_by_as_of_date():
    from prepare_data import sec_shares
    days = pd.bdate_range("2014-01-02", "2014-12-31")
    splits = pd.Series([7.0], index=pd.DatetimeIndex(["2014-06-09"], tz="America/New_York"))
    filings = _filings([("2014-04-24", "a", "2014-04-11", 861e6), ("2014-07-23", "b", "2014-07-11", 5.987e9)])
    s, _ = sec_shares(filings, splits, days)
    # Pre-split count restated x7; the post-split filing is already on today's basis.
    assert s.loc["2014-06-10"] == 861e6 * 7 and s.loc["2014-08-01"] == 5.987e9


def test_structural_check_rejects_class_only_counts():
    from prepare_data import check_sec_against_reference
    idx = pd.bdate_range("2016-01-01", periods=300)
    caps = pd.DataFrame({"sec_shares": 0.15e9, "yf_shares": 400e9, "ratio": 380e9}, index=idx)
    accepted, info = check_sec_against_reference(caps)
    assert not accepted and set(info) == {"yf_shares", "ratio"}       # BRK-style: fails both
    caps["sec_shares"] = 395e9
    assert check_sec_against_reference(caps)[0]
    # A contaminated Yahoo series (reused ticker) alone does not reject SEC when the fallback agrees.
    assert check_sec_against_reference(caps.assign(yf_shares=80e9))[0]
    # Without Yahoo overlap the ratio fallback over the last year is the only reference.
    accepted, info = check_sec_against_reference(caps.assign(yf_shares=float("nan"), ratio=1500e9))
    assert accepted is False and set(info) == {"ratio"}
