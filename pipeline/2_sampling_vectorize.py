"""Company-day events, seven-year chronological sample, and DTM / TF-IDF vectors.

1. Event alignment: every article in news.csv is mapped to its first tradable
   day (utils.map_to_event_date). Headlines of the same company and New York
   calendar date form one bundle; a bundle whose articles span the 4 PM close
   goes to its latest event date and is flagged ``mixed_session``. Bundles of
   several calendar days that land on the same trading day (weekend/holiday
   news) are merged, so each (symbol, event date) appears once.
2. Targets: events are joined one-to-one with historical_impacts_values.csv
   and kept only when both IIS and MIV exist (common sample for both models).
3. Period: np.random.seed(42) picks a start year from 2009-2014; the sample is
   that year's 1 January through 31 December six years later.
4. Split: whole event dates, chronological. Training ends on the first event
   date at which 80% of the eligible events (by count) have occurred; testing
   starts on the next trading day. An event is kept only if all three CAR days
   lie inside its own period, so the last two training trading days are
   dropped. The last 10% of training events form a validation tail for early
   stopping, with the same rule at that boundary.
5. Vectors: CountVectorizer (DTM) and TfidfVectorizer are fit on training text
   only and applied unchanged to the test text.

Writes news_{dtm,tfidf}_{train,test}.csv and sampling_manifest.json to the data
folder, and the fitted vectorizers, sparse .npz copies of the matrices and an
events table to the output folder.

Usage:
    python 2_sampling_vectorize.py [--data-dir DIR] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from utils import (CAR_DAYS, DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, IMPACTS_SCHEMA, NEWS_SCHEMA, SEED, TEXT_SEPARATOR,
                   TOKEN_PATTERN, VECTOR_COLUMNS, VECTORS_SCHEMA, clean_text, format_vector,
                   load_and_validate_csv, map_to_event_date, trading_days, write_json, write_validated_csv)

START_YEARS = [2009, 2010, 2011, 2012, 2013, 2014]
PERIOD_YEARS = 7
TRAIN_SHARE = 0.8
VALIDATION_SHARE = 0.1   # of the training events
VECTORIZER_PARAMS = {
    "token_pattern": TOKEN_PATTERN, "lowercase": False,  # clean_text already lowercases
    "ngram_range": (1, 2), "min_df": 5, "max_df": 0.95,
    "max_features": 1000,  # keeps the four dense-list CSVs at roughly 650 MB for the full sample
}
TFIDF_DECIMALS = 6


def select_period(seed=SEED):
    np.random.seed(seed)
    start_year = int(np.random.choice(START_YEARS))
    return start_year, pd.Timestamp(f"{start_year}-01-01"), pd.Timestamp(f"{start_year + PERIOD_YEARS - 1}-12-31")


def build_events(news, days):
    """One row per (symbol, event_date) with the cleaned, joined headline text."""
    mapped = map_to_event_date(news["date"], days)
    news = news.reset_index(drop=True).assign(
        event_date=mapped["event_date"], local_date=mapped["local_date"], precision=mapped["time_precision"])
    stats = {"articles": len(news), "articles_beyond_calendar": int(news["event_date"].isna().sum())}
    news = news[news["event_date"].notna()].copy()
    news["text"] = [clean_text(h, s) for h, s in zip(news["headline"], news["symbol"])]
    empty = news["text"].eq("")
    stats["articles_empty_after_cleaning"] = int(empty.sum())
    news = news[~empty].sort_values(["symbol", "date"], kind="stable")

    bundles = news.groupby(["symbol", "local_date"], sort=False).agg(
        event_date=("event_date", "max"), n_event_dates=("event_date", "nunique"),
        text=("text", TEXT_SEPARATOR.join), n_headlines=("text", "size"),
        exact_times=("precision", lambda p: int((p == "exact").sum())),
    ).reset_index()
    bundles["mixed_session"] = bundles["n_event_dates"] > 1
    stats["calendar_day_bundles"] = len(bundles)
    stats["mixed_session_bundles"] = int(bundles["mixed_session"].sum())

    events = bundles.groupby(["symbol", "event_date"], sort=True).agg(
        text=("text", TEXT_SEPARATOR.join), n_headlines=("n_headlines", "sum"),
        n_calendar_days=("local_date", "size"), mixed_session=("mixed_session", "any"),
        exact_times=("exact_times", "sum"),
    ).reset_index()
    stats["events"] = len(events)
    stats["events_merging_several_calendar_days"] = int((events["n_calendar_days"] > 1).sum())
    return events, stats


def attach_targets(events, impacts):
    """Join targets one-to-one; keep events with both IIS and MIV and count why the rest fail."""
    targets = impacts[["symbol", "date", "IIS", "MIV"]]
    targets = targets.assign(event_date=pd.to_datetime(targets["date"])).drop(columns="date")
    joined = events.merge(targets, on=["symbol", "event_date"], how="left", validate="one_to_one",
                          indicator=True)
    no_price = joined["_merge"].eq("left_only")
    no_iis = ~no_price & joined["IIS"].isna()
    no_miv = ~no_price & joined["IIS"].notna() & joined["MIV"].isna()
    reasons = {"no_price_row_for_event_date": int(no_price.sum()), "iis_undefined": int(no_iis.sum()),
               "miv_missing_market_cap": int(no_miv.sum())}
    keep = joined["IIS"].notna() & joined["MIV"].notna()
    return joined[keep].drop(columns="_merge").reset_index(drop=True), reasons


def first_date_reaching(dates, share):
    """Earliest date at which `share` of the rows (sorted by date) have occurred."""
    counts = dates.value_counts().sort_index()
    return counts.index[np.searchsorted(counts.cumsum().to_numpy(), share * counts.sum())]


def split_events(events, days, period_start, period_end):
    """Assign each event a partition: inner_train, validation, test, or an excluded_* reason."""
    days = pd.DatetimeIndex(days)
    pos = days.searchsorted(events["event_date"])
    end_pos = pos + CAR_DAYS - 1
    events = events.assign(window_end=pd.NaT)
    in_calendar = end_pos < len(days)
    events.loc[in_calendar, "window_end"] = days[end_pos[in_calendar]]

    period_days = days[(days >= period_start) & (days <= period_end)]
    first_day, last_day = period_days[0], period_days[-1]
    partition = pd.Series("excluded_outside_period", index=events.index)
    in_period = (events["event_date"] >= first_day) & (events["event_date"] <= last_day)
    partition[in_period & ~(events["window_end"] <= last_day)] = "excluded_window_beyond_period"
    eligible = in_period & (events["window_end"] <= last_day)

    train_end = first_date_reaching(events.loc[eligible, "event_date"], TRAIN_SHARE)
    test_start = days[days.searchsorted(train_end, side="right")]
    is_train = eligible & (events["event_date"] <= train_end)
    partition[eligible & (events["event_date"] >= test_start)] = "test"
    partition[is_train & (events["window_end"] > train_end)] = "excluded_window_crosses_test"
    training = is_train & (events["window_end"] <= train_end)

    val_start = first_date_reaching(events.loc[training, "event_date"], 1 - VALIDATION_SHARE)
    inner_end = days[days.searchsorted(val_start) - 1]
    partition[training & (events["event_date"] >= val_start)] = "validation"
    partition[training & (events["event_date"] < val_start) & (events["window_end"] > inner_end)] = \
        "excluded_window_crosses_validation"
    partition[training & (events["window_end"] <= inner_end)] = "inner_train"

    boundaries = {
        "period": {"calendar": [str(period_start.date()), str(period_end.date())],
                   "trading_days": [str(first_day.date()), str(last_day.date())]},
        "train": {"start": str(first_day.date()), "end": str(train_end.date())},
        "validation": {"start": str(val_start.date()), "end": str(train_end.date()),
                       "inner_train_end": str(inner_end.date())},
        "test": {"start": str(test_start.date()), "end": str(last_day.date())},
    }
    return events.assign(partition=partition.values), boundaries


def check_no_leakage(events, boundaries):
    train = events[events["partition"].isin(["inner_train", "validation"])]
    test = events[events["partition"] == "test"]
    inner = events[events["partition"] == "inner_train"]
    assert train["window_end"].max() <= pd.Timestamp(boundaries["train"]["end"])
    assert test["event_date"].min() >= pd.Timestamp(boundaries["test"]["start"])
    assert test["window_end"].max() <= pd.Timestamp(boundaries["test"]["end"])
    assert inner["window_end"].max() <= pd.Timestamp(boundaries["validation"]["inner_train_end"])
    assert not set(train["event_date"]) & set(test["event_date"])


def vectorize(train_text, test_text):
    """Fit each vectorizer on training text only; transform test text with the fitted object."""
    out = {}
    for name, cls in [("dtm", CountVectorizer), ("tfidf", TfidfVectorizer)]:
        vec = cls(**VECTORIZER_PARAMS)
        out[name] = (vec, vec.fit_transform(train_text), vec.transform(test_text))
    assert list(out["dtm"][0].vocabulary_) == list(out["tfidf"][0].vocabulary_)
    return out


def vector_frame(events, matrix, decimals, chunk=5000):
    vectors = []
    for start in range(0, matrix.shape[0], chunk):  # densify in chunks to bound memory
        vectors += [format_vector(row, decimals) for row in matrix[start:start + chunk].toarray().tolist()]
    return pd.DataFrame({
        "date": events["event_date"].dt.strftime("%Y-%m-%d").values,
        "symbol": events["symbol"].values,
        "news_vector": vectors,
        "IIS": events["IIS"].astype(int).values,
        "MIV": events["MIV"].values,
    })[VECTOR_COLUMNS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    news = load_and_validate_csv(args.data_dir / "news.csv", NEWS_SCHEMA)
    impacts = load_and_validate_csv(args.data_dir / "historical_impacts_values.csv", IMPACTS_SCHEMA)
    days = trading_days(impacts["date"].min(), impacts["date"].max())

    events, event_stats = build_events(news, days)
    events, no_target = attach_targets(events, impacts)
    event_stats["events_without_valid_targets"] = no_target
    event_stats["events_with_targets"] = len(events)

    start_year, period_start, period_end = select_period()
    events, boundaries = split_events(events, days, period_start, period_end)
    check_no_leakage(events, boundaries)

    train = events[events["partition"].isin(["inner_train", "validation"])].sort_values(["event_date", "symbol"])
    test = events[events["partition"] == "test"].sort_values(["event_date", "symbol"])
    vectors = vectorize(train["text"], test["text"])

    artifacts = args.out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for name, (vec, X_train, X_test) in vectors.items():
        decimals = None if name == "dtm" else TFIDF_DECIMALS
        write_validated_csv(vector_frame(train, X_train, decimals), args.data_dir / f"news_{name}_train.csv", VECTORS_SCHEMA)
        write_validated_csv(vector_frame(test, X_test, decimals), args.data_dir / f"news_{name}_test.csv", VECTORS_SCHEMA)
        joblib.dump(vec, artifacts / f"{name}_vectorizer.joblib")
        sp.save_npz(artifacts / f"{name}_train.npz", X_train)  # compact copies, same row order as the CSVs
        sp.save_npz(artifacts / f"{name}_test.npz", X_test)
    events.drop(columns=["n_event_dates"], errors="ignore").to_csv(artifacts / "events.csv", index=False)

    params = {k: list(v) if isinstance(v, tuple) else v for k, v in VECTORIZER_PARAMS.items()}
    manifest = {
        "seed": SEED,
        "start_year_candidates": START_YEARS,
        "selected_start_year": start_year,
        **boundaries,
        "split_method": "chronological by whole event dates; 80% boundary by event count (observations), "
                        "validation tail = last 10% of training events by count",
        "event_window_rule": f"all {CAR_DAYS} CAR trading days must lie inside the event's own period",
        "actual_coverage": {"first_event": str(train["event_date"].min().date()),
                            "last_event": str(test["event_date"].max().date())},
        "events": event_stats,
        "partition_counts": events["partition"].value_counts().to_dict(),
        "rows": {"train": len(train), "test": len(test),
                 "validation": int((train["partition"] == "validation").sum())},
        "symbols": {"train": int(train["symbol"].nunique()), "test": int(test["symbol"].nunique())},
        "text_separator": TEXT_SEPARATOR,
        "vectorizer_params": params,
        "tfidf_decimals": TFIDF_DECIMALS,
        "vocabulary_size": len(vectors["dtm"][0].vocabulary_),
        "test_rows_with_empty_vector": int((vectors["dtm"][2].sum(axis=1) == 0).sum()),
    }
    write_json(manifest, args.data_dir / "sampling_manifest.json")
    print({k: manifest[k] for k in ["selected_start_year", "train", "validation", "test", "rows",
                                    "partition_counts", "vocabulary_size"]})


if __name__ == "__main__":
    main()
