import numpy as np
import pandas as pd
import pytest

from conftest import load_script
from utils import trading_days

sampling = load_script("2_sampling_vectorize.py")
DAYS = trading_days("2011-06-01", "2019-03-31")


def test_seed_42_selects_2012_to_2018():
    start_year, start, end = sampling.select_period()
    assert start_year == 2012
    assert (start, end) == (pd.Timestamp("2012-01-01"), pd.Timestamp("2018-12-31"))


@pytest.fixture
def split():
    rng = np.random.default_rng(0)
    dates = DAYS[(DAYS >= "2011-09-01") & (DAYS <= "2019-02-28")]
    events = pd.DataFrame([(s, d) for d in dates for s in ["AAA", "BBB"] if rng.random() < 0.7],
                          columns=["symbol", "event_date"])
    events["IIS"], events["MIV"], events["text"] = 0, 1.0, "headline"
    _, start, end = sampling.select_period()
    return sampling.split_events(events, DAYS, start, end)


def test_partitions_respect_event_windows(split):
    events, b = split
    sampling.check_no_leakage(events, b)
    train_end, test_start = pd.Timestamp(b["train"]["end"]), pd.Timestamp(b["test"]["start"])
    assert test_start == DAYS[DAYS.searchsorted(train_end, side="right")]   # next trading day

    part = events.set_index("event_date")["partition"]
    # The last two training trading days cross into testing: their CAR windows end after train_end.
    last_two = DAYS[DAYS <= train_end][-2:]
    assert set(part[part.index.isin(last_two)]) <= {"excluded_window_crosses_test"}
    # Inside the period but with the window past 2018-12-31: the last two trading days of 2018.
    assert set(part[part.index.isin(pd.to_datetime(["2018-12-28", "2018-12-31"]))]) == {"excluded_window_beyond_period"}
    assert set(part[part.index < "2012-01-01"]) == {"excluded_outside_period"}
    assert set(part[part.index > "2018-12-31"]) == {"excluded_outside_period"}


def test_eighty_percent_by_count_and_whole_dates(split):
    events, b = split
    train = events[events["partition"].isin(["inner_train", "validation"])]
    test = events[events["partition"] == "test"]
    share = len(train) / (len(train) + len(test))
    assert 0.79 < share < 0.81
    assert not set(train["event_date"]) & set(test["event_date"])


def test_validation_tail(split):
    events, b = split
    val = events[events["partition"] == "validation"]
    inner = events[events["partition"] == "inner_train"]
    assert val["event_date"].min() == pd.Timestamp(b["validation"]["start"])
    assert inner["window_end"].max() < val["event_date"].min()
    assert 0.09 < len(val) / (len(val) + len(inner)) < 0.11


def test_vectorizers_fit_on_training_only(monkeypatch):
    monkeypatch.setattr(sampling, "VECTORIZER_PARAMS",
                        {**sampling.VECTORIZER_PARAMS, "min_df": 1, "max_df": 1.0, "max_features": None,
                         "ngram_range": (1, 1)})
    train = pd.Series(["upgrade beats", "downgrade misses", "upgrade raises"])
    test = pd.Series(["upgrade bankruptcy"])
    out = sampling.vectorize(train, test)
    vec, _, X_test = out["tfidf"]
    assert "bankruptcy" not in vec.vocabulary_
    # Smoothed IDF from the 3 training documents only: ln((1+3)/(1+2)) + 1 for "upgrade".
    assert vec.idf_[vec.vocabulary_["upgrade"]] == pytest.approx(np.log(4 / 3) + 1)
    dtm, _, X_test_dtm = out["dtm"]
    assert list(dtm.get_feature_names_out()) == ["beats", "downgrade", "misses", "raises", "upgrade"]
    assert X_test_dtm.toarray().tolist() == [[0, 0, 0, 0, 1]]


def test_build_events_merges_weekend_and_flags_mixed_sessions():
    news = pd.DataFrame({
        "date": ["2019-01-04 15:00:00-04:00",   # Fri 14:00 EST -> Fri
                 "2019-01-04 17:30:00-04:00",   # Fri 16:30 EST -> Mon, same calendar day: mixed
                 "2019-01-05 00:00:00",         # Saturday -> Mon
                 "2019-01-07 00:00:00"],        # Monday
        "symbol": "AAA",
        "headline": ["Upgrade", "Downgrade", "Weekend note", "Monday note"],
    })
    events, stats = sampling.build_events(news, DAYS)
    assert events["event_date"].tolist() == [pd.Timestamp("2019-01-07")]
    row = events.iloc[0]
    assert row["n_headlines"] == 4 and row["n_calendar_days"] == 3 and row["mixed_session"]
    assert row["text"] == "upgrade | downgrade | weekend note | monday note"
    assert stats["mixed_session_bundles"] == 1
