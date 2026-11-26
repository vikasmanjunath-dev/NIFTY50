"""RAG -> ML fusion: late fusion of document-derived event features (Blueprint §4.8).

"For V3, start with late fusion or structured-feature early fusion. Keep a
pure-market baseline. Only promote a RAG pathway after time-sliced ablations
show incremental out-of-sample value." This module implements exactly the
structured-feature route: ``documents/extraction.py`` already turned raw
announcements into daily (instrument, date) event counts and a materiality
signal; this module merges those onto the main daily feature matrix using
the same point-in-time-safe pattern as every other secondary data source
(delivery, macro, fundamentals) in this project.

The one detail specific to event data: a day with *zero* announcements is
real, known information ("nothing happened"), not missing data -- so counts
are filled with 0 after the merge, unlike fundamentals/delivery/macro where
an absent row correctly stays NaN ("not yet known").
"""
from __future__ import annotations

import pandas as pd

EVENT_FEATURE_COLUMNS = [
    "announcement_count",
    "high_materiality_count",
    "medium_materiality_count",
    "days_since_high_materiality_event",
]


def merge_events_into_features(features: pd.DataFrame, daily_events: pd.DataFrame) -> pd.DataFrame:
    """As-of-day broadcast join: each feature row gets that SAME calendar day's
    event counts (if any were published that day) via an exact-date merge,
    then zero-fills the count columns (not the novelty column -- see below).

    Unlike the delivery/macro merges, this is a same-day merge on
    ``event_time``'s calendar date, not a max-of-two-available-times join:
    announcement ``available_time`` and the daily bar's ``available_time`` are
    both within the same trading day for this source (confirmed during
    ingestion: NSE's own dissemination timestamp is typically only ~1 second
    after the announcement, well within the trading day it was made on).
    """
    left = features.copy()
    left["_event_date"] = pd.to_datetime(left["event_time"]).dt.tz_localize(None).dt.normalize()

    right = daily_events[["instrument_id", "event_date", *EVENT_FEATURE_COLUMNS]].rename(
        columns={"event_date": "_event_date"}
    )

    merged = left.merge(right, on=["instrument_id", "_event_date"], how="left")

    count_cols = ["announcement_count", "high_materiality_count", "medium_materiality_count"]
    merged[count_cols] = merged[count_cols].fillna(0)
    # days_since_high_materiality_event is deliberately NOT zero-filled: a day
    # with no announcement at all still has a real, meaningful "days since"
    # value (it just needs to be carried forward from the last known one),
    # handled by forward-filling per instrument in a follow-up pass.
    merged = merged.sort_values(["instrument_id", "event_time"])
    merged["days_since_high_materiality_event"] = merged.groupby("instrument_id")[
        "days_since_high_materiality_event"
    ].ffill()

    return merged.drop(columns=["_event_date"]).reset_index(drop=True)


def add_rolling_event_intensity(features: pd.DataFrame, windows: tuple[int, ...] = (7, 30)) -> pd.DataFrame:
    """Trailing-window event intensity: how many (high-materiality) announcements
    in the last N calendar days -- a smoother signal than a single day's count.
    """
    parts = []
    for _, g in features.groupby("instrument_id", sort=False):
        g = g.sort_values("event_time").reset_index(drop=True).copy()
        for w in windows:
            g[f"announcement_count_{w}d"] = g["announcement_count"].rolling(w, min_periods=1).sum()
            g[f"high_materiality_count_{w}d"] = g["high_materiality_count"].rolling(w, min_periods=1).sum()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
