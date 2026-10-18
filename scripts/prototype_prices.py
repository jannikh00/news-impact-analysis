"""Milestone A: prototype prices and market capitalization for a few companies.

Downloads daily OHLC/adjusted prices from Yahoo Finance for a small ticker
sample plus the S&P 500 index (stored as ``^sp500``), then compares three
market-capitalization methods:

1. ``yf_shares``  - Yahoo shares-outstanding history (as reported, starts ~2015)
2. ``sec_shares`` - SEC EDGAR XBRL ``dei:EntityCommonStockSharesOutstanding``
                    (as reported, ~2009+). Only runs when the environment
                    variable SEC_USER_AGENT is set ("Name email@example.com"),
                    because the SEC rejects requests without a contact.
3. ``ratio``      - the assignment's fallback: current market cap scaled by
                    split-adjusted close_t / close_today (constant share count).

Yahoo's ``Close`` is split-adjusted to today's share basis, while reported
share counts are as of their own date. Methods 1-2 therefore restate each
count on today's basis (count x splits after its as-of date) before
multiplying by ``Close``. This also covers the gap between a split and the
next filing, when the latest reported count is still pre-split.

Usage:
    python scripts/prototype_prices.py [--tickers AAPL JPM ...]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "feasibility"
START, END = "2008-06-01", "2020-07-15"  # estimation history before 2009 + event-window tail after 2020-06
DEFAULT_TICKERS = ["AAPL", "JPM", "KO", "NVDA", "BRK.B", "META", "VTRS"]
INDEX_YAHOO, INDEX_SYMBOL = "^GSPC", "^sp500"
# Accepted sec_shares / ratio-fallback cap range. Wide on purpose: the fallback itself
# drifts by up to ~2x with buybacks (AAPL 2009), while single-class counts are off ~1000x.
SEC_RATIO_BOUNDS = (0.2, 5.0)
PRICE_COLS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "market_cap"]


def yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def download(symbol: str) -> tuple[pd.DataFrame, yf.Ticker]:
    t = yf.Ticker(INDEX_YAHOO if symbol == INDEX_SYMBOL else yahoo_symbol(symbol))
    h = t.history(start=START, end=END, auto_adjust=False, actions=True)
    h.index = h.index.tz_localize(None).normalize()
    return h, t


def future_split_factor(dates, splits: pd.Series) -> np.ndarray:
    """Product of every split ratio *after* each date (including splits after END)."""
    dates = pd.DatetimeIndex(dates)
    splits = splits[splits > 0]
    factor = np.ones(len(dates))
    for d, ratio in splits.items():
        factor[dates < d.tz_localize(None).normalize()] *= ratio
    return factor


def yf_shares(t: yf.Ticker, index: pd.DatetimeIndex) -> pd.Series:
    """Yahoo share counts on today's split basis, forward-filled to trading days."""
    s = t.get_shares_full(start="2008-01-01", end=END)
    if s is None or s.empty:
        return pd.Series(np.nan, index=index)
    s.index = s.index.tz_localize(None).normalize()
    s = s.groupby(level=0).last().sort_index()
    s = s * future_split_factor(s.index, t.splits)
    return s.reindex(index, method="ffill")


_SEC_CIKS = None


def sec_shares(symbol: str, index: pd.DatetimeIndex, ua: str, splits: pd.Series) -> pd.Series:
    """Cover-page share counts on today's split basis, effective from their filing date."""
    global _SEC_CIKS
    headers = {"User-Agent": ua}
    if _SEC_CIKS is None:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30)
        r.raise_for_status()
        _SEC_CIKS = {v["ticker"]: v["cik_str"] for v in r.json().values()}
    cik = _SEC_CIKS.get(yahoo_symbol(symbol))
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
    # Restate each count on today's split basis using its as-of date ("end"), not the filing date.
    df["val"] = df["val"] * future_split_factor(pd.to_datetime(df["end"]), splits)
    s = df.assign(filed=pd.to_datetime(df["filed"])).groupby("filed")["val"].last().sort_index()
    return s.reindex(index, method="ffill")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sec_ua = os.environ.get("SEC_USER_AGENT")

    nyse = mcal.get_calendar("NYSE").schedule(start_date=START, end_date=END).index.normalize()
    nyse = nyse[nyse < pd.Timestamp(END)]

    frames, report = [], {"window": [START, END], "nyse_trading_days": len(nyse),
                          "sec_shares_enabled": bool(sec_ua), "tickers": {}}
    for sym in [INDEX_SYMBOL] + args.tickers:
        h, t = download(sym)
        out = pd.DataFrame({
            "date": h.index, "symbol": sym, "open": h["Open"], "high": h["High"], "low": h["Low"],
            "close": h["Close"], "adj_close": h["Adj Close"], "volume": h["Volume"],
        }).reset_index(drop=True)
        info = {
            "first_date": str(h.index.min().date()), "last_date": str(h.index.max().date()),
            "rows": len(h),
            "missing_nyse_days_after_first_date": int(len(nyse[nyse >= h.index.min()].difference(h.index))),
            "extra_non_nyse_days": int(len(h.index.difference(nyse))),
            "splits_in_or_after_window": {str(k.date()): float(v) for k, v in t.splits.items()
                                          if k.tz_localize(None) >= pd.Timestamp(START)},
        }

        if sym == INDEX_SYMBOL:
            out["market_cap"] = np.nan
        else:
            caps = pd.DataFrame(index=h.index)
            caps["yf_shares"] = h["Close"] * yf_shares(t, h.index)
            if sec_ua:
                caps["sec_shares"] = h["Close"] * sec_shares(sym, h.index, sec_ua, t.splits)
            current_cap = t.fast_info.get("marketCap") or t.fast_info.get("market_cap")
            last_close = t.history(period="5d", auto_adjust=False)["Close"].iloc[-1]
            caps["ratio"] = current_cap * h["Close"] / last_close
            # Plausibility guard: the cover-page tag can hold a single share class
            # (BRK reports only Class A), which yields caps off by orders of magnitude.
            if "sec_shares" in caps:
                implausible = ~(caps["sec_shares"] / caps["ratio"]).between(SEC_RATIO_BOUNDS[0], SEC_RATIO_BOUNDS[1])
                implausible &= caps["sec_shares"].notna()
                info["sec_shares_rejected_implausible"] = int(implausible.sum())
                caps.loc[implausible, "sec_shares"] = np.nan
            for col in caps:
                info[f"{col}_coverage"] = round(float(caps[col].notna().mean()), 3)
                info[f"{col}_first_date"] = str(caps[col].first_valid_index().date()) if caps[col].notna().any() else None
            both = caps[["yf_shares", "ratio"]].dropna()
            if len(both):
                rel = both["ratio"] / both["yf_shares"] - 1
                info["ratio_vs_yf_shares_rel_error"] = {
                    "median": round(float(rel.median()), 3), "min": round(float(rel.min()), 3),
                    "max": round(float(rel.max()), 3)}
            # Preferred method: SEC -> Yahoo shares -> ratio fallback, with the source recorded.
            order = [c for c in ["sec_shares", "yf_shares", "ratio"] if c in caps]
            chosen = caps[order].bfill(axis=1).iloc[:, 0]
            source = caps[order].notna().idxmax(axis=1).where(caps[order].notna().any(axis=1))
            out["market_cap"] = chosen.values
            out["cap_source"] = source.values
            info["cap_source_share"] = {k: round(v, 3) for k, v in
                                        source.value_counts(normalize=True).to_dict().items()}
            caps.assign(symbol=sym).to_csv(OUT_DIR / f"cap_methods_{yahoo_symbol(sym)}.csv", index_label="date")
        report["tickers"][sym] = info
        frames.append(out)
        print(sym, json.dumps(info, default=str), flush=True)

    prices = pd.concat(frames, ignore_index=True)
    prices[["date", "symbol", "cap_source"]].dropna().to_csv(
        OUT_DIR / "prototype_cap_sources.csv", index=False, date_format="%Y-%m-%d")
    prices[PRICE_COLS].to_csv(OUT_DIR / "prototype_historical_prices.csv", index=False, date_format="%Y-%m-%d")

    # Sanity anchor: Apple first closed above a $1 trillion market cap on 2018-08-02.
    if "AAPL" in args.tickers:
        aapl = prices[(prices["symbol"] == "AAPL") & (prices["date"] == "2018-08-02")]
        report["aapl_2018_08_02_market_cap"] = float(aapl["market_cap"].iloc[0])
    (OUT_DIR / "prototype_prices_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "tickers"}, indent=2))


if __name__ == "__main__":
    main()
