"""Milestone A: chunked profile of the two supplied news datasets.

Streams each CSV directly out of ``news_datasets.zip`` (no extracted copy) and
writes a JSON profile plus a few small CSV tables to ``results/feasibility/``.

Usage:
    python scripts/inspect_news.py [--zip PATH] [--sp500 PATH] [--chunksize N]
"""

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline" if (ROOT / "pipeline").exists() else ROOT  # repo layout, or the course's Assignment_3/
DEFAULT_ZIP = PIPELINE / "datasets" / "news_datasets.zip"
DEFAULT_SP500 = PIPELINE / "datasets" / "sp500.csv"
OUT_DIR = ROOT / "results" / "feasibility"
FILES = ["analyst_ratings.csv", "headlines.csv"]

# Timestamp shape classes, checked in order.
TS_PATTERNS = {
    "datetime_with_offset": re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"),
    "naive_midnight": re.compile(r"^\d{4}-\d{2}-\d{2} 00:00:00$"),
    "naive_datetime": re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
    "date_only": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
}


def classify_timestamps(s: pd.Series) -> pd.Series:
    out = pd.Series("other", index=s.index, dtype=object)
    out[s.isna()] = "missing"
    remaining = s.notna()
    for name, pat in TS_PATTERNS.items():
        hit = remaining & s.str.match(pat, na=False)
        out[hit] = name
        remaining &= ~hit
    return out


def session_bucket(et: pd.Series) -> pd.Series:
    """Bucket Eastern-time timestamps relative to the NYSE regular session."""
    minutes = et.dt.hour * 60 + et.dt.minute
    bucket = np.select(
        [et.dt.dayofweek >= 5, minutes < 9 * 60 + 30, minutes < 16 * 60],
        ["weekend", "pre_open", "regular"],
        default="after_close",
    )
    return pd.Series(bucket, index=et.index)


def profile_file(zf: zipfile.ZipFile, name: str, sp500: set, chunksize: int) -> dict:
    stats = {
        "rows": 0,
        "raw_columns": None,
        "missing": Counter(),
        "timestamp_formats": Counter(),
        "utc_offsets": Counter(),
        "session_buckets_et": Counter(),
        "rows_per_year": Counter(),
        "publishers": Counter(),
        "tickers": Counter(),
        "headline_len_chars": [],
        "html_like_headlines": 0,
        "parse_failures": 0,
        "ticker_shapes": Counter(),
    }
    row_hashes, key_hashes = [], []
    min_date, max_date = None, None
    # Company-calendar-day groups -> set of before/after-close flags, for the mixed-day check.
    day_sessions = {}

    with zf.open(name) as fh:
        reader = pd.read_csv(fh, chunksize=chunksize, dtype=str, keep_default_na=False, na_values=[""])
        for chunk in reader:
            if stats["raw_columns"] is None:
                stats["raw_columns"] = list(chunk.columns)
            chunk = chunk.drop(columns=[c for c in chunk.columns if c.startswith("Unnamed")])
            chunk = chunk.rename(columns={"stock": "symbol", "url": "URL"})
            stats["rows"] += len(chunk)

            for col in chunk.columns:
                stats["missing"][col] += int(chunk[col].isna().sum())

            fmt = classify_timestamps(chunk["date"])
            stats["timestamp_formats"].update(fmt.value_counts().to_dict())

            offsets = chunk["date"].str.extract(r"([+-]\d{2}:\d{2})$")[0].dropna()
            stats["utc_offsets"].update(offsets.value_counts().to_dict())

            ts = pd.to_datetime(chunk["date"], utc=True, errors="coerce", format="mixed")
            naive = fmt.isin(["naive_midnight", "naive_datetime", "date_only"])
            # Calendar date as written in the source (no timezone shift for naive stamps).
            cal = pd.to_datetime(chunk["date"].str.slice(0, 10), errors="coerce")
            stats["parse_failures"] += int(cal.isna().sum())

            stats["rows_per_year"].update(cal.dt.year.dropna().astype(int).value_counts().to_dict())
            cmin, cmax = cal.min(), cal.max()
            min_date = cmin if min_date is None or cmin < min_date else min_date
            max_date = cmax if max_date is None or cmax > max_date else max_date

            aware = fmt.eq("datetime_with_offset") & ts.notna()
            if aware.any():
                et = ts[aware].dt.tz_convert("America/New_York")
                buckets = session_bucket(et)
                stats["session_buckets_et"].update(buckets.value_counts().to_dict())
                grp = pd.DataFrame({
                    "symbol": chunk.loc[aware, "symbol"],
                    "cal": et.dt.strftime("%Y-%m-%d"),
                    "late": buckets.isin(["after_close"]).astype(int),
                })
                for (sym, d), late in grp.groupby(["symbol", "cal"])["late"]:
                    day_sessions.setdefault((sym, d), set()).update(late.unique().tolist())
            if naive.any():
                stats["session_buckets_et"]["unknown_time"] += int(naive.sum())

            stats["publishers"].update(chunk["publisher"].value_counts().to_dict())
            stats["tickers"].update(chunk["symbol"].value_counts().to_dict())
            shapes = chunk["symbol"].fillna("").str.replace(r"[A-Z]", "A", regex=True)
            stats["ticker_shapes"].update(shapes.value_counts().head(20).to_dict())

            head = chunk["headline"].fillna("")
            stats["headline_len_chars"].append(head.str.len().to_numpy())
            stats["html_like_headlines"] += int(head.str.contains(r"<[a-zA-Z/][^>]*>|&[a-z]+;|&#\d+;", regex=True).sum())

            row_hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
            key_hashes.append(pd.util.hash_pandas_object(
                pd.DataFrame({"s": chunk["symbol"], "d": chunk["date"].str.slice(0, 10),
                              "h": head.str.strip().str.lower()}), index=False).to_numpy())

    lens = np.concatenate(stats["headline_len_chars"])
    rh, kh = np.concatenate(row_hashes), np.concatenate(key_hashes)
    tickers = stats["tickers"]
    in_universe = {t for t in tickers if t in sp500}
    mixed = sum(1 for v in day_sessions.values() if len(v) > 1)

    return {
        "file": name,
        "rows": stats["rows"],
        "raw_columns": stats["raw_columns"],
        "missing_by_column": dict(stats["missing"]),
        "calendar_date_min": str(min_date.date()),
        "calendar_date_max": str(max_date.date()),
        "date_parse_failures": stats["parse_failures"],
        "rows_per_year": dict(sorted(stats["rows_per_year"].items())),
        "timestamp_formats": dict(stats["timestamp_formats"]),
        "utc_offsets": dict(stats["utc_offsets"]),
        "session_buckets_et": dict(stats["session_buckets_et"]),
        "company_days_with_known_time": len(day_sessions),
        "company_days_mixing_before_and_after_close": mixed,
        "exact_duplicate_rows": int(len(rh) - len(np.unique(rh))),
        "duplicate_symbol_date_headline": int(len(kh) - len(np.unique(kh))),
        "unique_tickers": len(tickers),
        "tickers_in_sp500_snapshot": len(in_universe),
        "rows_in_sp500_snapshot": int(sum(tickers[t] for t in in_universe)),
        "ticker_shapes_top": dict(stats["ticker_shapes"].most_common(10)),
        "unique_publishers": len(stats["publishers"]),
        "top_publishers": dict(stats["publishers"].most_common(10)),
        "top_tickers": dict(tickers.most_common(15)),
        "headline_length_chars": {
            "mean": round(float(lens.mean()), 1),
            "median": float(np.median(lens)),
            "p99": float(np.percentile(lens, 99)),
            "max": int(lens.max()),
        },
        "html_or_entity_headlines": stats["html_like_headlines"],
        "_key_hashes": kh,
        "_tickers": tickers,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--sp500", type=Path, default=DEFAULT_SP500)
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    sp500 = set(pd.read_csv(args.sp500)["symbol"].dropna())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = []
    with zipfile.ZipFile(args.zip) as zf:
        for name in FILES:
            print(f"profiling {name} ...", flush=True)
            profiles.append(profile_file(zf, name, sp500, args.chunksize))

    a, b = profiles
    overlap_keys = np.intersect1d(np.unique(a["_key_hashes"]), np.unique(b["_key_hashes"]))
    all_tickers = a["_tickers"] + b["_tickers"]
    coverage = pd.DataFrame(
        [{"symbol": t,
          "analyst_ratings_rows": a["_tickers"].get(t, 0),
          "headlines_rows": b["_tickers"].get(t, 0)} for t in sorted(sp500)]
    )
    coverage["total_rows"] = coverage["analyst_ratings_rows"] + coverage["headlines_rows"]
    coverage.sort_values("total_rows", ascending=False).to_csv(OUT_DIR / "sp500_news_coverage.csv", index=False)

    non_universe = pd.Series({t: n for t, n in all_tickers.items() if t not in sp500}).sort_values(ascending=False)
    non_universe.head(100).rename_axis("symbol").rename("rows").to_csv(OUT_DIR / "top_non_sp500_tickers.csv")

    for p in profiles:
        p.pop("_key_hashes"), p.pop("_tickers")
    summary = {
        "files": profiles,
        "cross_file_symbol_date_headline_overlap": int(len(overlap_keys)),
        "sp500_snapshot_size": len(sp500),
        "sp500_tickers_with_any_news": int((coverage["total_rows"] > 0).sum()),
        "sp500_tickers_without_news": coverage.loc[coverage["total_rows"] == 0, "symbol"].tolist(),
    }
    (OUT_DIR / "news_profile.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
