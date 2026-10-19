"""Milestone C: flag ticker-years whose headlines rarely name the current company.

A ticker can be reused: before 2021 ``COR`` was CoreSite Realty, while Yahoo's
``COR`` history (and today's S&P 500 member) is Cencora. Such news would be
matched with the wrong company's returns. For every universe symbol (after the
alias mapping) and calendar year, this script measures the share of headlines
that mention the current company's name, or a predecessor name for aliased
tickers, and flags symbol-years far below the symbol's own typical rate.

Writes ``results/full_sample/ticker_identity_audit.csv`` (all symbol-years) and
prints the flagged ones for manual review.

Usage:
    python scripts/audit_ticker_identity.py
"""

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline" if (ROOT / "pipeline").exists() else ROOT  # repo layout, or the course's Assignment_3/
sys.path.insert(0, str(PIPELINE))
from prepare_data import NEWS_FILES, load_universe  # noqa: E402

DATASETS = PIPELINE / "datasets"
OUT = ROOT / "results" / "full_sample" / "ticker_identity_audit.csv"
GENERIC = {"inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.", "the", "group", "holdings",
           "holding", "plc", "ltd", "ltd.", "trust", "international", "technologies", "&", "and", "of"}
# Manual review of flags that are not reused tickers (reused ones are in ticker_exclusions.csv).
REVIEWED = {
    "AVGO": "ok: headlines use the predecessor name Avago Technologies (same listing)",
    "CRH": "ok: 31 rows in 2012, mostly sector round-ups",
    "A": "minor: 2009 rows are unrelated feed items; outside the 2012-2018 sample",
    "KEY": "removed: 2020 rows are market-research spam (IRRELEVANT_HEADLINE filter)",
}
MIN_ROWS = 30          # symbol-years with fewer headlines are not judged
FLAG_RATIO = 0.25      # flag when a year's mention rate is below 25% of the symbol's median year


def name_keywords(name):
    """Distinctive lowercase tokens of a company name, e.g. 'Cencora' or 'Home Depot' -> 'home', 'depot'."""
    words = [re.sub(r"[^\w&']", "", w).lower() for w in re.split(r"[\s/\-–|]+", name)]
    words = [w for w in words if w and w not in GENERIC and len(w) > 1]
    return words[:2]


def main():
    sp500, alias_map, _ = load_universe(DATASETS)
    names = pd.read_csv(DATASETS / "sp500.csv").set_index("symbol")["company_name"]
    aliases = pd.read_csv(DATASETS / "ticker_aliases.csv")
    # Predecessor names, from "Facebook renamed Meta Platforms (2022)" -> "Facebook".
    predecessor = {r.symbol: r.change.split(" renamed ")[0].split(" merged ")[0].split(" -> ")[0].split(" ticker ")[0]
                   for r in aliases.itertuples()}

    frames = []
    with zipfile.ZipFile(DATASETS / "news_datasets.zip") as zf:
        for name in NEWS_FILES:
            df = pd.read_csv(zf.open(name), dtype=str, usecols=["headline", "date", "stock"])
            df["symbol"] = df["stock"].map(alias_map).fillna(df["stock"])
            frames.append(df[df["symbol"].isin(sp500)])
    news = pd.concat(frames, ignore_index=True)
    news["year"] = news["date"].str.slice(0, 4)
    head = news["headline"].str.lower()

    rows = []
    for symbol, idx in news.groupby("symbol").groups.items():
        keywords = name_keywords(names[symbol]) + name_keywords(predecessor.get(symbol, ""))
        pattern = "|".join(r"\b" + re.escape(k) for k in keywords)
        hit = head.loc[idx].str.contains(pattern, regex=True) if pattern else pd.Series(False, index=idx)
        per_year = pd.DataFrame({"year": news.loc[idx, "year"], "hit": hit}).groupby("year")["hit"].agg(["size", "mean"])
        for year, r in per_year.iterrows():
            rows.append({"symbol": symbol, "company_name": names[symbol], "keywords": " ".join(keywords),
                         "year": year, "rows": int(r["size"]), "name_mention_rate": round(r["mean"], 3)})

    audit = pd.DataFrame(rows)
    judged = audit[audit["rows"] >= MIN_ROWS]
    typical = judged.groupby("symbol")["name_mention_rate"].median().rename("symbol_median_rate")
    audit = audit.merge(typical, on="symbol", how="left")
    audit["flag"] = (audit["rows"] >= MIN_ROWS) & (audit["name_mention_rate"] < FLAG_RATIO * audit["symbol_median_rate"])
    # Whole-symbol flag: the name almost never appears in any year.
    audit["flag_symbol"] = audit["symbol_median_rate"] < 0.05
    excluded = pd.read_csv(DATASETS / "ticker_exclusions.csv").groupby("news_symbol")["reason"].first()
    flagged = audit["flag"] | audit["flag_symbol"]
    audit["resolution"] = ""
    def resolve(row):
        if row["symbol"] in excluded:
            return f"excluded: {excluded[row['symbol']]}"
        if row["symbol"] in REVIEWED:
            return REVIEWED[row["symbol"]]
        return "outside the 2012-2018 sample" if not 2012 <= int(row["year"]) <= 2018 else "unreviewed"
    audit.loc[flagged, "resolution"] = audit[flagged].apply(resolve, axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT, index=False)

    flagged = audit[flagged]
    print(f"{audit['symbol'].nunique()} symbols, {len(audit)} symbol-years, {len(flagged)} flagged")
    print(flagged.sort_values(["symbol", "year"]).to_string(index=False))


if __name__ == "__main__":
    main()
