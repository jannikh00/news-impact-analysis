"""Record a dated snapshot of the current S&P 500 constituent list.

Writes ``pipeline/datasets/sp500.csv`` with exactly two columns
(``symbol,company_name``) and a provenance sidecar
``pipeline/datasets/sp500_snapshot.json`` (source URL, retrieval time,
count, and the class-share tickers whose spelling differs between sources).

Usage:
    python scripts/build_sp500.py
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline" if (ROOT / "pipeline").exists() else ROOT  # repo layout, or the course's Assignment_3/
OUT_CSV = PIPELINE / "datasets" / "sp500.csv"
OUT_META = PIPELINE / "datasets" / "sp500_snapshot.json"
SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "news-impact-analysis/0.1 (academic research; python-requests)"}


def main():
    resp = requests.get(SOURCE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]

    df = (
        table.rename(columns={"Symbol": "symbol", "Security": "company_name"})[["symbol", "company_name"]]
        .assign(symbol=lambda d: d["symbol"].str.strip().str.upper(),
                company_name=lambda d: d["company_name"].str.strip())
        .drop_duplicates("symbol")
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    if not 495 <= len(df) <= 510:
        raise ValueError(f"unexpected constituent count: {len(df)}")
    df.to_csv(OUT_CSV, index=False)

    meta = {
        "source": SOURCE_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constituents": int(len(df)),
        "symbol_convention": "Exchange spelling with '.' for share classes (e.g. BRK.B); "
                             "Yahoo Finance uses '-' (BRK-B).",
        "class_share_symbols": df.loc[df["symbol"].str.contains(r"\.", regex=True), "symbol"].tolist(),
        "note": "Current constituents applied to 2009-2020 history: survivorship bias.",
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
