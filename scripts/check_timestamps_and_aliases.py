"""Milestone A: timezone-offset evidence and ticker-alias coverage.

1. Tests whether the fixed ``-04:00`` offsets in ``analyst_ratings.csv`` are
   true instants or mislabeled New York wall-clock times, by comparing the
   wall-clock hour of recurring scheduled Benzinga pieces in EDT vs EST months.
2. Counts news rows recovered by mapping predecessor tickers
   (``pipeline/datasets/ticker_aliases.csv``) onto current constituents.

Writes ``results/feasibility/timestamps_aliases.json``.

Usage:
    python scripts/check_timestamps_and_aliases.py
"""

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline" if (ROOT / "pipeline").exists() else ROOT  # repo layout, or the course's Assignment_3/
DATASETS = PIPELINE / "datasets"
OUT = ROOT / "results" / "feasibility" / "timestamps_aliases.json"
RECURRING = {
    "Biggest Movers From": r"Biggest Movers From",
    "Stocks Moving In ... Pre-Market": r"Stocks Moving In .* Pre-Market",
    "Mid-Day Session": r"Mid-Day Session",
    "Stocks That Hit 52-Week Highs": r"Stocks That Hit 52-Week Highs",
    "After-Hours": r"After-Hours",
}


def offset_evidence(a: pd.DataFrame) -> dict:
    timed = a[a["date"].str.contains(r"[+-]\d\d:\d\d$")].copy()
    timed["wall"] = pd.to_datetime(timed["date"].str.slice(0, 19))
    local = timed["wall"].dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    timed["dst"] = local.map(lambda t: bool(t.dst()) if pd.notna(t) else None)
    timed["hour"] = timed["wall"].dt.hour + timed["wall"].dt.minute / 60
    out = {}
    for label, pat in RECURRING.items():
        s = timed[timed["headline"].str.contains(pat, regex=True)]
        med = s.groupby("dst")["hour"].median().round(2)
        out[label] = {"n": len(s), "median_wall_hour_edt": med.get(True), "median_wall_hour_est": med.get(False)}
    return {
        "timed_rows": len(timed),
        "timed_rows_per_year": timed["wall"].dt.year.value_counts().sort_index().to_dict(),
        "recurring_pieces": out,
    }


def alias_coverage(news: pd.DataFrame) -> dict:
    sp500 = set(pd.read_csv(DATASETS / "sp500.csv")["symbol"])
    aliases = pd.read_csv(DATASETS / "ticker_aliases.csv")
    counts = news["stock"].value_counts()
    aliases["target_in_snapshot"] = aliases["symbol"].isin(sp500)
    aliases["news_rows"] = aliases["news_symbol"].map(counts).fillna(0).astype(int)

    mapping = dict(zip(aliases.loc[aliases["target_in_snapshot"], "news_symbol"],
                       aliases.loc[aliases["target_in_snapshot"], "symbol"]))
    news = news.assign(symbol=news["stock"].replace(mapping))
    universe = news[news["symbol"].isin(sp500)]
    exact = universe["date"].str.contains(r"[+-]\d\d:\d\d$")
    return {
        "aliases": aliases[["news_symbol", "symbol", "target_in_snapshot", "news_rows"]].to_dict("records"),
        "universe_rows_raw": int(news["stock"].isin(sp500).sum()),
        "universe_rows_with_aliases": len(universe),
        "universe_tickers_with_news": int(universe["symbol"].nunique()),
        "universe_tickers_without_news": sorted(sp500 - set(universe["symbol"])),
        "exact_time_share": round(float(exact.mean()), 4),
        "rows_per_year": universe["date"].str.slice(0, 4).value_counts().sort_index().to_dict(),
        "company_days": int(universe.assign(d=universe["date"].str.slice(0, 10))
                            .drop_duplicates(["symbol", "d"]).shape[0]),
    }


def main():
    cols = ["headline", "date", "stock"]
    with zipfile.ZipFile(DATASETS / "news_datasets.zip") as zf:
        a = pd.read_csv(zf.open("analyst_ratings.csv"), dtype=str, usecols=cols)
        h = pd.read_csv(zf.open("headlines.csv"), dtype=str, usecols=cols)
    report = {"offset_evidence": offset_evidence(a), "alias_coverage": alias_coverage(pd.concat([a, h]))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
