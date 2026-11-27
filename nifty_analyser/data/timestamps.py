"""IST timestamp helpers implementing the point-in-time "as-of" rule (Blueprint §3.2)."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from nifty_analyser.config import IST, NSE_CLOSE_HOUR_IST, NSE_CLOSE_MINUTE_IST


def to_ist(timestamp: pd.Timestamp | dt.datetime | str) -> pd.Timestamp:
    """Coerce to a tz-aware IST Timestamp (naive input is taken as IST wall-clock)."""
    ts = pd.Timestamp(timestamp)
    return ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)


def drop_unfinished(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove rows not yet knowable, e.g. Yahoo's still-forming bar for a session that has not closed."""
    now = to_ist(now if now is not None else pd.Timestamp.now(tz=IST))
    return frame[frame["available_time"] <= now].reset_index(drop=True)


def eod_bar_available_time(trading_date: pd.Timestamp | dt.date | str) -> pd.Timestamp:
    """A daily bar is only knowable once NSE's cash market has closed (15:30 IST)."""
    day = pd.Timestamp(trading_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    return to_ist(day + pd.Timedelta(hours=NSE_CLOSE_HOUR_IST, minutes=NSE_CLOSE_MINUTE_IST))
