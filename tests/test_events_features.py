"""Tests for RAG->ML event feature fusion (nifty_analyser/features/events.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_analyser.features.events import add_rolling_event_intensity, merge_events_into_features


def _daily_features(n: int = 30, instrument_id: str = "NSE:TEST") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "event_time": dates,
            "available_time": dates + pd.Timedelta(hours=15, minutes=30),
        }
    )


def _daily_events(instrument_id: str = "NSE:TEST") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": [instrument_id, instrument_id],
            "event_date": pd.to_datetime(["2024-01-03", "2024-01-15"]),
            "announcement_count": [2, 1],
            "high_materiality_count": [1, 0],
            "medium_materiality_count": [1, 1],
            "days_since_high_materiality_event": [pd.NA, 12],
        }
    )


def test_merge_zero_fills_count_columns_on_no_event_days():
    features = _daily_features(30)
    merged = merge_events_into_features(features, _daily_events())
    no_event_day = merged[merged.event_time == pd.Timestamp("2024-01-02", tz="Asia/Kolkata")]
    assert no_event_day["announcement_count"].iloc[0] == 0
    assert no_event_day["high_materiality_count"].iloc[0] == 0


def test_merge_carries_correct_counts_on_event_days():
    features = _daily_features(30)
    merged = merge_events_into_features(features, _daily_events())
    event_day = merged[merged.event_time == pd.Timestamp("2024-01-03", tz="Asia/Kolkata")]
    assert event_day["announcement_count"].iloc[0] == 2
    assert event_day["high_materiality_count"].iloc[0] == 1


def test_days_since_high_materiality_forward_fills_not_zero_fills():
    """The critical distinction from count columns: a no-event day still has a
    meaningful "days since" value, carried forward, not reset to 0.
    """
    features = _daily_features(30)
    merged = merge_events_into_features(features, _daily_events())
    day_after_second_event = merged[merged.event_time == pd.Timestamp("2024-01-16", tz="Asia/Kolkata")]
    # should carry forward the value set on 2024-01-15 (12), not be 0 or NaN
    assert day_after_second_event["days_since_high_materiality_event"].iloc[0] == 12


def test_days_since_high_materiality_is_na_before_first_known_value():
    features = _daily_features(30)
    merged = merge_events_into_features(features, _daily_events())
    day_before_any_event = merged[merged.event_time == pd.Timestamp("2024-01-02", tz="Asia/Kolkata")]
    assert pd.isna(day_before_any_event["days_since_high_materiality_event"].iloc[0])


def test_merge_does_not_cross_contaminate_instruments():
    features = pd.concat(
        [_daily_features(30, "NSE:AAA"), _daily_features(30, "NSE:BBB")], ignore_index=True
    )
    events = pd.concat(
        [_daily_events("NSE:AAA"), _daily_events("NSE:BBB").assign(announcement_count=[99, 99])],
        ignore_index=True,
    )
    merged = merge_events_into_features(features, events)
    aaa_day = merged[(merged.instrument_id == "NSE:AAA") & (merged.event_time == pd.Timestamp("2024-01-03", tz="Asia/Kolkata"))]
    bbb_day = merged[(merged.instrument_id == "NSE:BBB") & (merged.event_time == pd.Timestamp("2024-01-03", tz="Asia/Kolkata"))]
    assert aaa_day["announcement_count"].iloc[0] == 2
    assert bbb_day["announcement_count"].iloc[0] == 99


def test_rolling_intensity_no_lookahead():
    features = _daily_features(60)
    merged = merge_events_into_features(features, _daily_events())
    full = add_rolling_event_intensity(merged, windows=(7, 30))

    truncated_features = _daily_features(40)
    truncated_merged = merge_events_into_features(truncated_features, _daily_events())
    truncated = add_rolling_event_intensity(truncated_merged, windows=(7, 30))

    n = len(truncated)
    pd.testing.assert_series_equal(
        full["announcement_count_7d"].iloc[:n].reset_index(drop=True),
        truncated["announcement_count_7d"].reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
    )


def test_rolling_intensity_sums_correctly():
    features = _daily_features(30)
    merged = merge_events_into_features(features, _daily_events())
    out = add_rolling_event_intensity(merged, windows=(30,))
    last_row = out.iloc[-1]
    assert last_row["announcement_count_30d"] == pytest.approx(merged["announcement_count"].sum())
