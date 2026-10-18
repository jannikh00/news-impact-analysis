import sys
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from utils import map_to_event_date  # noqa: E402

DAYS = mcal.get_calendar("NYSE").schedule("2019-01-01", "2020-12-31").index.normalize()


@pytest.mark.parametrize("raw, expected, precision", [
    # Summer (EDT = UTC-4): offsets match local time.
    ("2020-06-05 08:00:00-04:00", "2020-06-05", "exact"),   # pre-open Friday
    ("2020-06-05 15:59:59-04:00", "2020-06-05", "exact"),   # last second of session
    ("2020-06-05 16:00:00-04:00", "2020-06-08", "exact"),   # at close -> Monday
    ("2020-06-06 11:00:00-04:00", "2020-06-08", "exact"),   # Saturday
    ("2019-07-03 17:00:00-04:00", "2019-07-05", "exact"),   # after close, next day is July 4th
    # Winter (EST = UTC-5): a fixed -04:00 stamp is one hour earlier in New York.
    ("2020-01-10 16:30:00-04:00", "2020-01-10", "exact"),   # = 15:30 EST, still in session
    ("2020-01-10 17:00:00-04:00", "2020-01-13", "exact"),   # = 16:00 EST -> Monday
    ("2020-01-10 00:30:00-04:00", "2020-01-10", "exact"),   # = 23:30 EST Thursday, after close -> Friday
    # Date-only values.
    ("2020-06-05 00:00:00", "2020-06-05", "date"),
    ("2020-06-07 00:00:00", "2020-06-08", "date"),          # Sunday
    ("2019-12-25 00:00:00", "2019-12-26", "date"),          # Christmas
])
def test_mapping(raw, expected, precision):
    out = map_to_event_date([raw], DAYS)
    assert out.loc[0, "time_precision"] == precision
    assert out.loc[0, "event_date"] == pd.Timestamp(expected)


def test_beyond_calendar_is_nat():
    out = map_to_event_date(["2020-12-31 17:00:00-04:00"], DAYS)
    assert pd.isna(out.loc[0, "event_date"])
