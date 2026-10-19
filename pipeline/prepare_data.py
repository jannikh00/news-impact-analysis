"""Prepare news.csv and historical_prices.csv from the supplied news and Yahoo/SEC data.

News: streams both files out of ``news_datasets.zip``, normalizes the headers
(``stock -> symbol``, ``url -> URL``, drops the unnamed index), drops rows whose
ticker belonged to a different company at the time (``ticker_exclusions.csv``),
maps predecessor and share-class tickers via ``ticker_aliases.csv``, keeps the
S&P 500 snapshot universe, drops invalid dates, syndicated market-research spam
and duplicates, and writes article-level rows
(timestamps are kept as published; company-day aggregation happens in
2_sampling_vectorize.py).

Prices: daily OHLC, adjusted close and volume from Yahoo Finance for every
company left in news.csv plus the S&P 500 index (``^GSPC`` stored as
``^sp500``, market_cap blank). Market capitalization uses, in order:
SEC EDGAR cover-page share counts, Yahoo share counts, then the current-cap
price-ratio fallback; each row's method goes to ``market_cap_sources.csv``.
``sec_cik_overrides.csv`` adds predecessor SEC filers for merged companies.
SEC requests need ``SEC_USER_AGENT="Name email"`` in the environment. Raw
Yahoo and SEC downloads are cached in ``datasets/cache`` (``--refresh`` re-downloads),
so the market-cap rules can be changed and rerun offline.

Usage:
    python prepare_data.py                      # full universe
    python prepare_data.py --tickers KR HD      # prototype subset
"""

import argparse
import json
import os
import time
import zipfile
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from utils import (DEFAULT_DATA_DIR, MARKET_SYMBOL, NEWS_COLUMNS, NEWS_SCHEMA, PRICE_COLUMNS,
                   PRICES_SCHEMA, load_and_validate_csv, write_json, write_validated_csv)

NEWS_FILES = ["analyst_ratings.csv", "headlines.csv"]  # analyst_ratings first: it carries exact timestamps
MIN_NEWS_DATE = "2009-01-01"                            # drops the 1969-12-31 epoch placeholder
PRICE_START, PRICE_END = "2008-06-01", "2020-07-15"     # estimation history before 2009, event tail after 2020-06
INDEX_YAHOO = "^GSPC"
# SEC quality checks. Per filing: drop isolated spikes, i.e. counts more than 2x away from both
# neighbouring filings (typos, counts in millions, placeholders, split-timing errors), worst
# first; genuine step changes survive because the following filings confirm the new level.
SEC_FILING_OUTLIER = 2.0
# Before that, drop counts more than 100x from the company's median (unit errors that repeat over
# several filings). In the cached filings, flagged counts are all >= ~480x or <= 1/1000 of the
# median, while every other count stays within ~20x of it.
SEC_UNIT_ERROR = 100.0
# Per company: reject SEC entirely only when its cap disagrees structurally with BOTH references,
# Yahoo's share counts and the ratio fallback over the window's last year, where that fallback
# drifts least (e.g. BRK reports only Class A). Either reference alone can be wrong: Yahoo's share
# history under a reused ticker can belong to the old company (COR), and the fallback drifts with
# large issuance (PG&E 2020).
SEC_VS_YAHOO_BOUNDS = (0.5, 2.0)
SEC_VS_RATIO_BOUNDS = (1 / 3, 3.0)
MIN_CHECK_OVERLAP = 60
CAP_METHODS = ["sec_shares", "yf_shares", "ratio"]
# Syndicated market-research press releases ("... Market Size, Share —…—… Report"), 2019-2020.
IRRELEVANT_HEADLINE = r"Market\s+(?:Size|Share)\s*,\s*(?:Share|Size)|—…—…"
YAHOO_RETRIES = 3


# --- Universe ---------------------------------------------------------------------

def load_universe(data_dir):
    """S&P 500 symbols, the news-ticker alias map, and the reused-ticker exclusions."""
    sp500 = set(pd.read_csv(data_dir / "sp500.csv")["symbol"])
    aliases = pd.read_csv(data_dir / "ticker_aliases.csv")
    usable = aliases[aliases["symbol"].isin(sp500) & (aliases["status"] != "excluded")]
    exclusions = pd.read_csv(data_dir / "ticker_exclusions.csv", dtype=str)
    return sp500, dict(zip(usable["news_symbol"], usable["symbol"])), exclusions


def excluded_rows(chunk, exclusions):
    """Rows whose raw news ticker meant another company on that date (empty start/end = open)."""
    day = chunk["date"].str.slice(0, 10)
    mask = pd.Series(False, index=chunk.index)
    for r in exclusions.itertuples():
        hit = chunk["symbol"].eq(r.news_symbol)
        if isinstance(r.start, str):
            hit &= day >= r.start
        if isinstance(r.end, str):
            hit &= day <= r.end
        mask |= hit
    return mask


# --- News -------------------------------------------------------------------------

def prepare_news(zip_path, sp500, alias_map, exclusions, tickers, chunksize=200_000):
    keep = set(tickers) if tickers else sp500
    frames, stats, reused = [], {"files": {}}, pd.Series(dtype=int)
    with zipfile.ZipFile(zip_path) as zf:
        for name in NEWS_FILES:
            read = kept = aliased = 0
            with zf.open(name) as fh:
                for chunk in pd.read_csv(fh, chunksize=chunksize, dtype=str, keep_default_na=False, na_values=[""]):
                    chunk = chunk.drop(columns=[c for c in chunk.columns if c.startswith("Unnamed")])
                    chunk = chunk.rename(columns={"stock": "symbol", "url": "URL"})
                    read += len(chunk)
                    drop = excluded_rows(chunk, exclusions)
                    reused = reused.add(chunk.loc[drop, "symbol"].value_counts(), fill_value=0)
                    chunk = chunk[~drop]
                    mapped = chunk["symbol"].map(alias_map)
                    aliased += int(mapped.notna().sum())
                    chunk["symbol"] = mapped.fillna(chunk["symbol"])
                    chunk = chunk[chunk["symbol"].isin(keep)]
                    kept += len(chunk)
                    frames.append(chunk.assign(_source=name))
            stats["files"][name] = {"rows_read": read, "rows_alias_mapped": aliased, "rows_in_universe": kept}

    news = pd.concat(frames, ignore_index=True)
    stats["dropped_reused_ticker"] = {k: int(v) for k, v in reused.sort_index().items()}
    stats["rows_in_universe"] = len(news)
    news = news[news["headline"].notna() & news["date"].notna()]
    valid_date = news["date"].str.slice(0, 10) >= MIN_NEWS_DATE
    stats["dropped_invalid_date"] = int((~valid_date).sum())
    news = news[valid_date]
    spam = news["headline"].str.contains(IRRELEVANT_HEADLINE, regex=True)
    stats["dropped_irrelevant_headlines"] = int(spam.sum())
    news = news[~spam]

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


def cached(path, fetch, refresh=False):
    """Load a pickled download from `path`, or fetch it and store it there."""
    if path.exists() and not refresh:
        return joblib.load(path)
    obj = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return obj


def fetch_yahoo(symbol):
    ticker = yf.Ticker(yahoo_symbol(symbol))
    data = {"hist": ticker.history(start=PRICE_START, end=PRICE_END, auto_adjust=False, actions=True),
            "splits": ticker.splits, "retrieved": date.today().isoformat()}
    if symbol != MARKET_SYMBOL and not data["hist"].empty:
        data["shares"] = ticker.get_shares_full(start="2008-01-01", end=PRICE_END)
        data["current_cap"] = ticker.fast_info.get("marketCap")
        data["last_close"] = ticker.history(period="5d", auto_adjust=False)["Close"].iloc[-1]
    return data


def fetch_sec_tickers(headers):
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30)
    r.raise_for_status()
    return {v["ticker"]: v["cik_str"] for v in r.json().values()}


def fetch_sec_filings(cik, headers):
    """Raw cover-page share counts (filed, accn, end, val) of one SEC filer."""
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=headers, timeout=30)
    time.sleep(0.15)  # stay well under the SEC's 10 requests/second limit
    empty = pd.DataFrame(columns=["filed", "accn", "end", "val"])
    if r.status_code == 404:
        return empty
    r.raise_for_status()
    fact = r.json()["facts"].get("dei", {}).get("EntityCommonStockSharesOutstanding")
    return pd.DataFrame(fact["units"]["shares"])[["filed", "accn", "end", "val"]] if fact else empty


def future_split_factor(dates, splits):
    """Product of every split ratio *after* each date (including splits after PRICE_END)."""
    dates = pd.DatetimeIndex(dates)
    splits = splits[splits > 0]
    factor = np.ones(len(dates))
    for d, ratio in splits.items():
        factor[dates < d.tz_localize(None).normalize()] *= ratio
    return factor


def yf_shares(shares, splits, index):
    """Yahoo share counts on today's split basis, forward-filled to trading days."""
    if shares is None or shares.empty:
        return pd.Series(np.nan, index=index)
    shares = shares.copy()
    shares.index = shares.index.tz_localize(None).normalize()
    shares = shares.groupby(level=0).last().sort_index()
    return (shares * future_split_factor(shares.index, splits)).reindex(index, method="ffill")


def spike_positions(values, factor=SEC_FILING_OUTLIER):
    """Positions of isolated spikes, removed one at a time from the most extreme.

    A value is a spike when it is more than `factor` away from both neighbours (for the
    first/last value: from the two nearest values on its only side).
    """
    v = np.asarray(values, dtype=float)
    alive, dropped = list(range(len(v))), []
    while len(alive) >= 3:
        devs = []
        for k, i in enumerate(alive):
            if k == 0:
                nb = alive[1:3]
            elif k == len(alive) - 1:
                nb = alive[-3:-1]
            else:
                nb = [alive[k - 1], alive[k + 1]]
            devs.append(min(max(v[i] / v[j], v[j] / v[i]) for j in nb))
        worst = int(np.argmax(devs))
        if devs[worst] <= factor:
            break
        dropped.append(alive.pop(worst))
    return dropped


def unit_error_positions(values, factor=SEC_UNIT_ERROR):
    """Positions of counts more than `factor` away from the company's median count.

    Catches runs of consecutive bad filings that confirm each other and so pass the
    spike check, e.g. counts reported in thousands of shares (GRMN 2016-2018, FTNT 2014),
    millions (AJG 2020) or a placeholder of 1 (ICE 2013).
    """
    v = np.asarray(values, dtype=float)
    ratio = v / np.median(v)
    return list(np.flatnonzero((ratio > factor) | (ratio < 1 / factor)))


def sec_shares(filings, splits, index):
    """SEC cover-page share counts on today's split basis, effective from their filing date.

    Yahoo's Close is split-adjusted to today's basis, while each count is as of its own
    date ("end"), so it is restated with the splits after that date. This also covers
    the gap between a split and the next filing, when the latest count is still pre-split.
    Returns the daily series and the number of filings dropped as outliers.
    """
    df = filings[filings["val"] > 0]
    if df.empty:
        return pd.Series(np.nan, index=index), 0
    # Multi-class filers report one value per class on the same date: sum them.
    df = df.drop_duplicates(["accn", "end", "val"]).groupby(["filed", "accn", "end"])["val"].sum().reset_index()
    df["val"] = df["val"] * future_split_factor(pd.to_datetime(df["end"]), splits)
    df = df.sort_values("filed").reset_index(drop=True)
    unit_errors = unit_error_positions(df["val"])
    df = df.drop(index=unit_errors).reset_index(drop=True)
    spikes = spike_positions(df["val"])
    df = df.drop(index=spikes)
    s = df.assign(filed=pd.to_datetime(df["filed"])).groupby("filed")["val"].last()
    return s.sort_index().reindex(index, method="ffill"), len(unit_errors) + len(spikes)


def check_sec_against_reference(caps):
    """Company-level structural check of the SEC cap; returns (accepted, details)."""
    checks = {}
    both = caps[["sec_shares", "yf_shares"]].dropna()
    if len(both) >= MIN_CHECK_OVERLAP:
        checks["yf_shares"] = (float((both["sec_shares"] / both["yf_shares"]).median()), SEC_VS_YAHOO_BOUNDS)
    recent = caps[["sec_shares", "ratio"]].dropna().tail(250)
    if len(recent) >= MIN_CHECK_OVERLAP:
        checks["ratio"] = (float((recent["sec_shares"] / recent["ratio"]).median()), SEC_VS_RATIO_BOUNDS)
    details = {ref: round(rel, 3) for ref, (rel, _) in checks.items()}
    if not checks:
        return True, details
    return any(lo <= rel <= hi for rel, (lo, hi) in checks.values()), details


def market_cap(data, sec_filings):
    """Daily market cap and the method used for each day (see module docstring)."""
    hist = data["hist"]
    caps, stats = pd.DataFrame(index=hist.index), {}
    if sec_filings is not None:
        sec, dropped = sec_shares(sec_filings, data["splits"], hist.index)
        caps["sec_shares"] = hist["Close"] * sec
        stats["sec_filings_dropped_outliers"] = dropped
    caps["yf_shares"] = hist["Close"] * yf_shares(data.get("shares"), data["splits"], hist.index)
    current_cap = data.get("current_cap")
    caps["ratio"] = current_cap * hist["Close"] / data["last_close"] if current_cap else np.nan

    if "sec_shares" in caps and caps["sec_shares"].notna().any():
        accepted, stats["sec_check"] = check_sec_against_reference(caps)
        if not accepted:
            caps["sec_shares"] = np.nan
            stats["sec_rejected_structural"] = True
    order = [c for c in CAP_METHODS if c in caps]
    available = caps[order].notna()
    chosen = caps[order].bfill(axis=1).iloc[:, 0]
    source = available.idxmax(axis=1).where(available.any(axis=1))
    stats["cap_source_share"] = {k: round(v, 3) for k, v in source.value_counts(normalize=True).items()}
    return chosen, source, stats


def sec_filings_for(symbol, filers, headers, cache_dir, refresh):
    """Combined filings of the symbol's SEC filers (overrides, else its current CIK)."""
    if filers is None:
        ciks = cached(cache_dir / "sec" / "company_tickers.joblib", lambda: fetch_sec_tickers(headers), refresh)
        cik = ciks.get(symbol.replace(".", "-"))
        if cik is None:
            return None
        filers = [(cik, None, None)]
    parts = []
    for cik, filed_from, filed_to in filers:
        df = cached(cache_dir / "sec" / f"{cik}.joblib", lambda: fetch_sec_filings(cik, headers), refresh)
        if filed_from:
            df = df[df["filed"] >= filed_from]
        if filed_to:
            df = df[df["filed"] <= filed_to]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def download_prices(symbol, sec_user_agent, filers, cache_dir, refresh):
    data = cached(cache_dir / "yahoo" / f"{yahoo_symbol(symbol)}.joblib", lambda: fetch_yahoo(symbol), refresh)
    hist = data["hist"].copy()
    if hist.empty:
        return None, None, {"error": "no Yahoo price history"}
    hist.index = hist.index.tz_localize(None).normalize()
    data = {**data, "hist": hist}
    prices = pd.DataFrame({
        "date": hist.index.strftime("%Y-%m-%d"), "symbol": symbol,
        "open": hist["Open"].values, "high": hist["High"].values, "low": hist["Low"].values,
        "close": hist["Close"].values, "adj_close": hist["Adj Close"].values,
        "volume": hist["Volume"].values,
    })
    stats = {"first_date": prices["date"].iloc[0], "last_date": prices["date"].iloc[-1], "rows": len(prices),
             "retrieved": data["retrieved"]}
    if symbol == MARKET_SYMBOL:
        prices["market_cap"] = np.nan
        return prices, None, stats
    sec_filings = None
    if sec_user_agent:
        sec_filings = sec_filings_for(symbol, filers, {"User-Agent": sec_user_agent}, cache_dir, refresh)
    cap, source, cap_stats = market_cap(data, sec_filings)
    prices["market_cap"] = cap.values
    sources = pd.DataFrame({"date": prices["date"], "symbol": symbol, "cap_source": source.values}).dropna()
    return prices, sources, {**stats, **cap_stats}


def load_cik_overrides(data_dir):
    path = data_dir / "sec_cik_overrides.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"filed_from": str, "filed_to": str})
    return {sym: [(int(r.cik), r.filed_from if isinstance(r.filed_from, str) else None,
                   r.filed_to if isinstance(r.filed_to, str) else None) for r in g.itertuples()]
            for sym, g in df.groupby("symbol")}


def prepare_prices(symbols, sec_user_agent, cik_overrides, cache_dir, refresh=False):
    frames, source_frames, stats = [], [], {}
    for symbol in [MARKET_SYMBOL] + sorted(symbols):
        for attempt in range(1, YAHOO_RETRIES + 1):
            try:
                prices, sources, info = download_prices(symbol, sec_user_agent, cik_overrides.get(symbol),
                                                        cache_dir, refresh)
                break
            except Exception as exc:  # network errors and provider rate limits: back off, then give up
                prices, sources, info = None, None, {"error": f"{type(exc).__name__}: {exc}"}
                time.sleep(10 * attempt)
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
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_DATA_DIR / "cache",
                    help="raw Yahoo/SEC downloads (reused on later runs)")
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    ap.add_argument("--skip-news", action="store_true", help="reuse the existing news.csv")
    ap.add_argument("--skip-prices", action="store_true")
    args = ap.parse_args()

    sec_user_agent = os.environ.get("SEC_USER_AGENT")
    sp500, alias_map, exclusions = load_universe(args.source_dir)
    if args.tickers:
        unknown = sorted(set(args.tickers) - sp500)
        if unknown:
            raise SystemExit(f"not in sp500.csv: {unknown}")

    manifest_path = args.data_dir / "prepare_manifest.json"
    if args.skip_news:
        news = load_and_validate_csv(args.data_dir / "news.csv", NEWS_SCHEMA)
        manifest = json.loads(manifest_path.read_text())
        news_stats = manifest["news"]
    else:
        print("preparing news.csv ...", flush=True)
        news, news_stats = prepare_news(args.source_dir / "news_datasets.zip", sp500, alias_map, exclusions,
                                        args.tickers)
        write_validated_csv(news, args.data_dir / "news.csv", NEWS_SCHEMA)
        manifest = {"tickers": args.tickers or "all", "aliases_applied": alias_map, "news": news_stats}
    manifest["sec_shares_enabled"] = bool(sec_user_agent)

    if not args.skip_prices:
        print("preparing historical_prices.csv ...", flush=True)
        prices, sources, price_stats = prepare_prices(news["symbol"].unique(), sec_user_agent,
                                                      load_cik_overrides(args.source_dir), args.cache_dir,
                                                      args.refresh)
        write_validated_csv(prices, args.data_dir / "historical_prices.csv", PRICES_SCHEMA)
        sources.to_csv(args.data_dir / "market_cap_sources.csv", index=False)
        manifest["prices"] = {"window": [PRICE_START, PRICE_END], "rows_written": len(prices), "symbols": price_stats}
    write_json(manifest, manifest_path)
    print(f"news rows: {news_stats['rows_written']}, symbols: {news_stats['symbols_written']}")


if __name__ == "__main__":
    main()
