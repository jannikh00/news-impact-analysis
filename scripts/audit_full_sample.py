"""Milestone C: coverage, exclusion and split audit of the full event dataset.

Reads the manifests and tables written by prepare_data.py and pipeline scripts
1-2 and writes to ``results/full_sample/``:

- exclusion_waterfall.csv  raw news rows -> articles -> events -> final split
- coverage_by_symbol.csv   per company: articles, price coverage, events per partition,
                           market-cap source of its sample events
- cap_source_in_sample.csv which method produced the pre-event cap behind each MIV
- split_manifest.json      sampling_manifest.json plus the audit totals

Usage:
    python scripts/audit_full_sample.py [--data-dir DIR] [--out-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from utils import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, write_json  # noqa: E402

RESULTS = ROOT / "results" / "full_sample"
SAMPLE = ["inner_train", "validation", "test"]


def waterfall(prep, impacts, sampling):
    news, ev = prep["news"], sampling["events"]
    raw = sum(f["rows_read"] for f in news["files"].values())
    reused = sum(news["dropped_reused_ticker"].values())
    parts = sampling["partition_counts"]
    no_target = ev["events_without_valid_targets"]
    steps = [
        ("raw news rows (both files)", raw, ""),
        ("after reused-ticker exclusions", raw - reused, "ticker_exclusions.csv"),
        ("in S&P 500 universe (after aliases)", news["rows_in_universe"], "snapshot + ticker_aliases.csv"),
        ("valid date", news["rows_in_universe"] - news["dropped_invalid_date"], "drops 1969-12-31 placeholder"),
        ("not market-research spam",
         news["rows_in_universe"] - news["dropped_invalid_date"] - news["dropped_irrelevant_headlines"], ""),
        ("after exact duplicates",
         news["rows_written"] + news["dropped_same_day_headline_duplicates"], ""),
        ("articles in news.csv", news["rows_written"], "after same-day headline duplicates"),
        ("articles mapped to a trading day", ev["articles"] - ev["articles_beyond_calendar"], ""),
        ("calendar-day bundles", ev["calendar_day_bundles"], "same company, same New York date"),
        ("company-day events", ev["events"], "weekend/holiday bundles merged into next session"),
        ("events with a price row", ev["events"] - no_target["no_price_row_for_event_date"],
         "no Yahoo price on the event date (pre-listing, delisted dates)"),
        ("events with IIS", ev["events"] - no_target["no_price_row_for_event_date"] - no_target["iis_undefined"],
         "insufficient history / incomplete window / zero variability"),
        ("events with IIS and MIV", ev["events_with_targets"], "common sample"),
        ("inside the sampled period", ev["events_with_targets"] - parts.get("excluded_outside_period", 0),
         f"{sampling['period']['calendar'][0]} .. {sampling['period']['calendar'][1]}"),
        ("CAR window inside period", sum(parts.get(p, 0) for p in SAMPLE)
         + parts.get("excluded_window_crosses_test", 0) + parts.get("excluded_window_crosses_validation", 0), ""),
        ("final sample (train + test)", sum(parts.get(p, 0) for p in SAMPLE),
         "boundary-crossing training events removed"),
    ]
    df = pd.DataFrame(steps, columns=["stage", "rows", "note"])
    df.insert(2, "dropped", (-df["rows"].diff()).fillna(0).astype(int))
    return df


def event_cap_sources(events, sources, prices):
    """Cap source of the trading row before each event date (the cap MIV uses)."""
    rows = prices[["date", "symbol"]].merge(sources, on=["date", "symbol"], how="left")
    rows = rows.sort_values(["symbol", "date"])
    rows["next_date"] = rows.groupby("symbol")["date"].shift(-1)
    lookup = rows.dropna(subset=["next_date"]).rename(columns={"next_date": "event_date"})
    lookup["event_date"] = pd.to_datetime(lookup["event_date"])
    return events.merge(lookup[["symbol", "event_date", "cap_source"]], on=["symbol", "event_date"], how="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    load = lambda name: json.loads((args.data_dir / name).read_text())
    prep, impacts, sampling = load("prepare_manifest.json"), load("impacts_manifest.json"), load("sampling_manifest.json")

    events = pd.read_csv(args.out_dir / "artifacts" / "events.csv", parse_dates=["event_date"],
                         usecols=["symbol", "event_date", "partition", "n_headlines", "mixed_session"])
    prices = pd.read_csv(args.data_dir / "historical_prices.csv", usecols=["date", "symbol"])
    sources = pd.read_csv(args.data_dir / "market_cap_sources.csv")
    news = pd.read_csv(args.data_dir / "news.csv", usecols=["symbol"])
    names = pd.read_csv(args.data_dir / "sp500.csv").set_index("symbol")["company_name"]
    events = event_cap_sources(events, sources, prices)

    RESULTS.mkdir(parents=True, exist_ok=True)
    wf = waterfall(prep, impacts, sampling)
    wf.to_csv(RESULTS / "exclusion_waterfall.csv", index=False)

    sample = events[events["partition"].isin(SAMPLE)]
    cap = sample.pivot_table(index="partition", columns="cap_source", values="symbol", aggfunc="size", fill_value=0)
    cap.to_csv(RESULTS / "cap_source_in_sample.csv")

    price_info = pd.DataFrame.from_dict(prep["prices"]["symbols"], orient="index")
    coverage = pd.DataFrame({"articles": news["symbol"].value_counts()})
    coverage["company_name"] = names.reindex(coverage.index)
    coverage = coverage.join(price_info[["first_date", "last_date"]].rename(
        columns={"first_date": "price_first_date", "last_date": "price_last_date"}), how="left")
    if "error" in price_info:
        coverage = coverage.join(price_info["error"].rename("price_error"), how="left")
    coverage = coverage.join(events.groupby("symbol").size().rename("events_with_targets"), how="left")
    coverage = coverage.join(events.pivot_table(index="symbol", columns="partition", values="event_date",
                                                aggfunc="size", fill_value=0), how="left")
    sample_caps = sample.pivot_table(index="symbol", columns="cap_source", values="event_date", aggfunc="size",
                                     fill_value=0).add_prefix("cap_")
    coverage = coverage.join(sample_caps, how="left").fillna({c: 0 for c in sample_caps.columns})
    for p in SAMPLE:
        coverage[p] = coverage.get(p, 0)
    coverage["sample_events"] = coverage[SAMPLE].fillna(0).sum(axis=1).astype(int)
    coverage = coverage.rename_axis("symbol").sort_values("sample_events", ascending=False)
    coverage.to_csv(RESULTS / "coverage_by_symbol.csv")

    summary = {
        "symbols_in_snapshot": int(len(names)),
        "symbols_with_articles": int(len(coverage)),
        "symbols_with_sample_events": int((coverage["sample_events"] > 0).sum()),
        "symbols_with_test_events": int((coverage["test"].fillna(0) > 0).sum()),
        "symbols_without_price_history": coverage.index[coverage["price_first_date"].isna()].tolist(),
        "sample_events": int(len(sample)),
        "sample_cap_source_share": sample["cap_source"].value_counts(normalize=True).round(4).to_dict(),
        "sample_mixed_session_events": int(sample["mixed_session"].sum()),
        "sample_median_headlines_per_event": float(sample["n_headlines"].median()),
        "sample_iis_distribution": None,
    }
    test = pd.read_csv(args.data_dir / "news_dtm_test.csv", usecols=["symbol", "IIS"])
    train = pd.read_csv(args.data_dir / "news_dtm_train.csv", usecols=["symbol", "IIS"])
    summary["sample_iis_distribution"] = {
        "train": train["IIS"].value_counts(normalize=True).sort_index().round(4).to_dict(),
        "test": test["IIS"].value_counts(normalize=True).sort_index().round(4).to_dict()}
    summary["top10_test_companies"] = test["symbol"].value_counts().head(10).to_dict()
    write_json({**sampling, "audit": summary}, RESULTS / "split_manifest.json")

    print(wf.to_string(index=False))
    print(cap.to_string())
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
