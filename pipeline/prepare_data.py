"""Prepare news.csv and historical_prices.csv from the supplied news and Yahoo/SEC data.

News: streams both files out of ``news_datasets.zip``, normalizes the headers
(``stock -> symbol``, ``url -> URL``, drops the unnamed index), maps
predecessor tickers via ``ticker_aliases.csv``, keeps the S&P 500 snapshot
universe, drops invalid dates and duplicates, and writes article-level rows
(timestamps are kept as published; company-day aggregation happens in
2_sampling_vectorize.py).

Prices: daily OHLC, adjusted close and volume from Yahoo Finance for every
company left in news.csv plus the S&P 500 index (``^GSPC`` stored as
``^sp500``, market_cap blank). Market capitalization uses, in order:
SEC EDGAR cover-page share counts, Yahoo share counts, then the current-cap
price-ratio fallback; each row's method goes to ``market_cap_sources.csv``.
SEC requests need ``SEC_USER_AGENT="Name email"`` in the environment.

Usage:
    python prepare_data.py                      # full universe
    python prepare_data.py --tickers KR HD      # prototype subset
"""

import argparse
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from utils import (DEFAULT_DATA_DIR, MARKET_SYMBOL, NEWS_COLUMNS, NEWS_SCHEMA, PRICE_COLUMNS,
                   PRICES_SCHEMA, write_json, write_validated_csv)

NEWS_FILES = ["analyst_ratings.csv", "headlines.csv"]  # analyst_ratings first: it carries exact timestamps
MIN_NEWS_DATE = "2009-01-01"                            # drops the 1969-12-31 epoch placeholder
PRICE_START, PRICE_END = "2008-06-01", "2020-07-15"     # estimation history before 2009, event tail after 2020-06
INDEX_YAHOO = "^GSPC"
# Accepted sec_shares / ratio-fallback cap range. Wide on purpose: the fallback itself
# drifts by up to ~2x with buybacks (AAPL 2009), while single-class counts are off ~1000x.
SEC_RATIO_BOUNDS = (0.2, 5.0)
CAP_METHODS = ["sec_shares", "yf_shares", "ratio"]


# --- Universe ---------------------------------------------------------------------

def load_universe(data_dir):
    sp500 = set(pd.read_csv(data_dir / "sp500.csv")["symbol"])
    aliases = pd.read_csv(data_dir / "ticker_aliases.csv")
    usable = aliases[aliases["symbol"].isin(sp500) & (aliases["status"] != "excluded")]
    return sp500, dict(zip(usable["news_symbol"], usable["symbol"]))


# --- News -------------------------------------------------------------------------

def prepare_news(zip_path, sp500, alias_map, tickers, chunksize=200_000):
    keep = set(tickers) if tickers else sp500
    frames, stats = [], {"files": {}}
    with zipfile.ZipFile(zip_path) as zf:
        for name in NEWS_FILES:
            read = kept = aliased = 0
            with zf.open(name) as fh:
                for chunk in pd.read_csv(fh, chunksize=chunksize, dtype=str, keep_default_na=False, na_values=[""]):
                    chunk = chunk.drop(columns=[c for c in chunk.columns if c.startswith("Unnamed")])
                    chunk = chunk.rename(columns={"stock": "symbol", "url": "URL"})
                    read += len(chunk)
                    mapped = chunk["symbol"].map(alias_map)
                    aliased += int(mapped.notna().sum())
                    chunk["symbol"] = mapped.fillna(chunk["symbol"])
                    chunk = chunk[chunk["symbol"].isin(keep)]
                    kept += len(chunk)
                    frames.append(chunk.assign(_source=name))
            stats["files"][name] = {"rows_read": read, "rows_alias_mapped": aliased, "rows_in_universe": kept}

    news = pd.concat(frames, ignore_index=True)
    stats["rows_in_universe"] = len(news)
    news = news[news["headline"].notna() & news["date"].notna()]
    valid_date = news["date"].str.slice(0, 10) >= MIN_NEWS_DATE
    stats["dropped_invalid_date"] = int((~valid_date).sum())
    news = news[valid_date]

    before = len(news)
    news = news.drop_duplicates(subset=["date", "symbol", "headline", "URL", "publisher"])
    stats["dropped_exact_duplicates"] = before - len(news)

    # Same company, calendar date and headline (case/whitespace-insensitive), e.g. the same
    # story in both files: keep the first copy, which prefers analyst_ratings' exact timestamp.
    key = news["headline"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    before = len(news)
    news = news[~pd.DataFrame({"s": news["symbol"], "d": news["date"].str.slice(0, 10), "h": key}).duplicated()]
    stats["dropped_same_day_headline_duplicates"] = before - len(news)
    stats["rows_by_source"] = news["_source"].value_counts().to_dict()

    news = (news.assign(article=np.nan)
            .sort_values(["symbol", "date"], kind="stable")[NEWS_COLUMNS]
            .reset_index(drop=True))
    stats["rows_written"] = len(news)
    stats["symbols_written"] = int(news["symbol"].nunique())
    return news, stats


# --- Prices and market capitalization -------------------------------------------

def yahoo_symbol(symbol):
    return INDEX_YAHOO if symbol == MARKET_SYMBOL else symbol.replace(".", "-")


def future_split_factor(dates, splits):
    """Product of every split ratio *after* each date (including splits after PRICE_END)."""
    dates = pd.DatetimeIndex(dates)
    splits = splits[splits > 0]
    factor = np.ones(len(dates))
    for d, ratio in splits.items():
        factor[dates < d.tz_localize(None).normalize()] *= ratio
    return factor


def yf_shares(ticker, index):
    """Yahoo share counts on today's split basis, forward-filled to trading days."""
    s = ticker.get_shares_full(start="2008-01-01", end=PRICE_END)
    if s is None or s.empty:
        return pd.Series(np.nan, index=index)
    s.index = s.index.tz_localize(None).normalize()
    s = s.groupby(level=0).last().sort_index()
    s = s * future_split_factor(s.index, ticker.splits)
    return s.reindex(index, method="ffill")


_SEC_CIKS = None


def sec_shares(symbol, index, user_agent, splits):
    """SEC cover-page share counts on today's split basis, effective from their filing date.

    Yahoo's Close is split-adjusted to today's basis, while each count is as of its own
    date ("end"), so it is restated with the splits after that date. This also covers
    the gap between a split and the next filing, when the latest count is still pre-split.
    """
    global _SEC_CIKS
    headers = {"User-Agent": user_agent}
    if _SEC_CIKS is None:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30)
        r.raise_for_status()
        _SEC_CIKS = {v["ticker"]: v["cik_str"] for v in r.json().values()}
    cik = _SEC_CIKS.get(symbol.replace(".", "-"))
    if cik is None:
        return pd.Series(np.nan, index=index)
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=headers, timeout=30)
    r.raise_for_status()
    fact = r.json()["facts"].get("dei", {}).get("EntityCommonStockSharesOutstanding")
    if not fact:
        return pd.Series(np.nan, index=index)
    df = pd.DataFrame(fact["units"]["shares"])
    df = df[df["val"] > 0]  # some filings report 0 (e.g. BRK 2010-04-30 10-Q)
    # Multi-class filers report one value per class on the same date: sum them.
    df = df.drop_duplicates(["accn", "end", "val"]).groupby(["filed", "accn", "end"])["val"].sum().reset_index()
    df["val"] = df["val"] * future_split_factor(pd.to_datetime(df["end"]), splits)
    s = df.assign(filed=pd.to_datetime(df["filed"])).groupby("filed")["val"].last().sort_index()
    time.sleep(0.15)  # stay well under the SEC's 10 requests/second limit
    return s.reindex(index, method="ffill")


def market_cap(symbol, ticker, hist, sec_user_agent):
    """Daily market cap and the method used for each day (see module docstring)."""
    caps = pd.DataFrame(index=hist.index)
    if sec_user_agent:
        caps["sec_shares"] = hist["Close"] * sec_shares(symbol, hist.index, sec_user_agent, ticker.splits)
    caps["yf_shares"] = hist["Close"] * yf_shares(ticker, hist.index)
    current_cap = ticker.fast_info.get("marketCap")
    last_close = ticker.history(period="5d", auto_adjust=False)["Close"].iloc[-1]
    caps["ratio"] = current_cap * hist["Close"] / last_close if current_cap else np.nan

    stats = {}
    if "sec_shares" in caps:
        implausible = caps["sec_shares"].notna() & ~(caps["sec_shares"] / caps["ratio"]).between(*SEC_RATIO_BOUNDS)
        stats["sec_shares_rejected_implausible"] = int(implausible.sum())
        caps.loc[implausible, "sec_shares"] = np.nan
    order = [c for c in CAP_METHODS if c in caps]
    available = caps[order].notna()
    chosen = caps[order].bfill(axis=1).iloc[:, 0]
    source = available.idxmax(axis=1).where(available.any(axis=1))
    stats["cap_source_share"] = {k: round(v, 3) for k, v in source.value_counts(normalize=True).items()}
    return chosen, source, stats


def download_prices(symbol, sec_user_agent):
    ticker = yf.Ticker(yahoo_symbol(symbol))
    hist = ticker.history(start=PRICE_START, end=PRICE_END, auto_adjust=False, actions=True)
    if hist.empty:
        return None, None, {"error": "no Yahoo price history"}
    hist.index = hist.index.tz_localize(None).normalize()
    prices = pd.DataFrame({
        "date": hist.index.strftime("%Y-%m-%d"), "symbol": symbol,
        "open": hist["Open"].values, "high": hist["High"].values, "low": hist["Low"].values,
        "close": hist["Close"].values, "adj_close": hist["Adj Close"].values,
        "volume": hist["Volume"].values,
    })
    stats = {"first_date": prices["date"].iloc[0], "last_date": prices["date"].iloc[-1], "rows": len(prices)}
    if symbol == MARKET_SYMBOL:
        prices["market_cap"] = np.nan
        return prices, None, stats
    cap, source, cap_stats = market_cap(symbol, ticker, hist, sec_user_agent)
    prices["market_cap"] = cap.values
    sources = pd.DataFrame({"date": prices["date"], "symbol": symbol, "cap_source": source.values}).dropna()
    return prices, sources, {**stats, **cap_stats}


def prepare_prices(symbols, sec_user_agent):
    frames, source_frames, stats = [], [], {}
    for symbol in [MARKET_SYMBOL] + sorted(symbols):
        prices, sources, info = download_prices(symbol, sec_user_agent)
        stats[symbol] = info
        print(f"  {symbol}: {info}", flush=True)
        if prices is not None:
            frames.append(prices)
        if sources is not None:
            source_frames.append(sources)
    prices = pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]
    prices = prices[prices["adj_close"].notna()]
    return prices, pd.concat(source_frames, ignore_index=True), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", nargs="+", help="restrict to these universe symbols (default: whole snapshot)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="where outputs are written")
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="folder with news_datasets.zip, sp500.csv and ticker_aliases.csv")
    ap.add_argument("--skip-prices", action="store_true")
    args = ap.parse_args()

    sec_user_agent = os.environ.get("SEC_USER_AGENT")
    sp500, alias_map = load_universe(args.source_dir)
    if args.tickers:
        unknown = sorted(set(args.tickers) - sp500)
        if unknown:
            raise SystemExit(f"not in sp500.csv: {unknown}")

    print("preparing news.csv ...", flush=True)
    news, news_stats = prepare_news(args.source_dir / "news_datasets.zip", sp500, alias_map, args.tickers)
    write_validated_csv(news, args.data_dir / "news.csv", NEWS_SCHEMA)
    manifest = {"tickers": args.tickers or "all", "aliases_applied": alias_map, "news": news_stats,
                "sec_shares_enabled": bool(sec_user_agent)}

    if not args.skip_prices:
        print("preparing historical_prices.csv ...", flush=True)
        prices, sources, price_stats = prepare_prices(news["symbol"].unique(), sec_user_agent)
        write_validated_csv(prices, args.data_dir / "historical_prices.csv", PRICES_SCHEMA)
        sources.to_csv(args.data_dir / "market_cap_sources.csv", index=False)
        manifest["prices"] = {"window": [PRICE_START, PRICE_END], "rows_written": len(prices), "symbols": price_stats}
    write_json(manifest, args.data_dir / "prepare_manifest.json")
    print(f"news rows: {news_stats['rows_written']}, symbols: {news_stats['symbols_written']}")


if __name__ == "__main__":
    main()
