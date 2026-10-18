import numpy as np
import pandas as pd
import pandera as pa

MARKET_TZ = "America/New_York"
MARKET_CLOSE_MINUTES = 16 * 60


def load_and_validate_csv(filepath, schema):
    df = pd.read_csv(filepath)
    schema.validate(df, lazy=True)
    return df


def map_to_event_date(timestamps, trading_days):
    """Map raw news timestamps to the first trading day the market could react.

    timestamps: strings as they appear in the news files. Values with a UTC
        offset (e.g. ``2020-06-05 10:30:54-04:00``) are treated as exact
        instants and converted to US Eastern time. Values at exactly midnight
        without an offset carry no time of day and are treated as date-only.
    trading_days: sorted DatetimeIndex of exchange trading dates (no time).

    Rules: pre-open and regular-session news -> same trading day; news at or
    after 16:00 ET, on weekends, or on holidays -> next trading day. Date-only
    news on a trading day -> that day (the 3-day event window still covers a
    possible after-close reaction on day +1).

    Returns a DataFrame with ``event_date`` and ``time_precision``
    (``"exact"`` or ``"date"``); event_date is NaT beyond the last trading day.
    """
    s = pd.Series(timestamps).astype("string")
    days = pd.DatetimeIndex(trading_days).normalize()

    has_offset = s.str.contains(r"[+-]\d{2}:\d{2}$", regex=True, na=False)
    exact = pd.Series(pd.NaT, index=s.index, dtype=f"datetime64[ns, {MARKET_TZ}]")
    if has_offset.any():
        exact[has_offset] = pd.to_datetime(s[has_offset], utc=True).dt.tz_convert(MARKET_TZ)

    local_date = pd.to_datetime(s.str.slice(0, 10), errors="coerce")
    local_date[has_offset] = exact[has_offset].dt.tz_localize(None).dt.normalize()

    after_close = pd.Series(False, index=s.index)
    minutes = exact.dt.hour * 60 + exact.dt.minute
    after_close[has_offset] = minutes[has_offset] >= MARKET_CLOSE_MINUTES

    # First trading day >= local date; strictly after it when news came after the close.
    pos = days.searchsorted(local_date.to_numpy(), side="left")
    on_trading_day = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == local_date.to_numpy())
    pos = pos + (after_close.to_numpy() & on_trading_day)
    event = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    valid = local_date.notna().to_numpy() & (pos < len(days))
    event[valid] = days[pos[valid]]

    return pd.DataFrame({
        "event_date": event,
        "time_precision": np.where(has_offset, "exact", "date"),
    })
