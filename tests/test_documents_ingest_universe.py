"""Tests for whole-exchange announcements ingestion (nifty_analyser/documents/ingest.py).

The universe-level fetch itself is mocked (no network); these tests focus on
the filtering-to-our-48-instruments logic, deduplication across windows, and
the checkpoint callback contract -- the parts that are new relative to the
already-tested per-symbol path.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from nifty_analyser.documents.ingest import ingest_announcements_universe


def _record(symbol: str, seq_id: str, an_dt: str = "10-Jan-2024 10:00:00") -> dict:
    return {
        "seq_id": seq_id,
        "symbol": symbol,
        "an_dt": an_dt,
        "exchdisstime": an_dt,
        "desc": "Updates",
        "attchmntText": f"Announcement text for {symbol}.",
        "attchmntFile": None,
        "sm_isin": "INE000A00000",
    }


def test_filters_out_symbols_not_in_our_universe():
    fake_records = [_record("RELIANCE", "1"), _record("SOME_RANDOM_SMALLCAP", "2")]
    with patch(
        "nifty_analyser.documents.ingest.fetch_announcements_window_universe",
        return_value=fake_records,
    ):
        result = ingest_announcements_universe(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))
    assert set(result["instrument_id"]) == {"NSE:RELIANCE"}


def test_keeps_multiple_universe_symbols_from_one_window():
    fake_records = [_record("RELIANCE", "1"), _record("TCS", "2"), _record("INFY", "3")]
    with patch(
        "nifty_analyser.documents.ingest.fetch_announcements_window_universe",
        return_value=fake_records,
    ):
        result = ingest_announcements_universe(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))
    assert set(result["instrument_id"]) == {"NSE:RELIANCE", "NSE:TCS", "NSE:INFY"}


def test_deduplicates_across_windows():
    """A record with the same seq_id appearing in two overlapping-boundary
    window responses must only appear once in the final result.
    """
    fake_records = [_record("RELIANCE", "1")]
    with patch(
        "nifty_analyser.documents.ingest.fetch_announcements_window_universe",
        return_value=fake_records,
    ):
        # A range spanning multiple 150-day windows -> the mock returns the
        # SAME record for every window call, simulating a boundary duplicate.
        result = ingest_announcements_universe(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    assert len(result) == 1


def test_checkpoint_callback_invoked_once_per_window():
    fake_records = [_record("RELIANCE", "1")]
    calls = []

    def _on_window(start, end, df):
        calls.append((start, end, len(df)))

    with patch(
        "nifty_analyser.documents.ingest.fetch_announcements_window_universe",
        return_value=fake_records,
    ):
        ingest_announcements_universe(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), on_window_complete=_on_window
        )
    assert len(calls) >= 2  # a full year spans more than one 150-day window
    # the fake record is deduplicated across windows (same seq_id every call),
    # so only the FIRST window should report it as newly found -- the rest
    # correctly report 0, proving dedup applies at the per-window checkpoint
    # level too, not just in the final concatenated result.
    assert calls[0][2] == 1
    assert all(c[2] == 0 for c in calls[1:])


def test_empty_response_returns_empty_frame_with_correct_columns():
    with patch(
        "nifty_analyser.documents.ingest.fetch_announcements_window_universe",
        return_value=[],
    ):
        result = ingest_announcements_universe(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))
    from nifty_analyser.documents.ingest import DOCUMENT_COLUMNS

    assert result.empty
    assert list(result.columns) == DOCUMENT_COLUMNS


def test_unknown_symbol_field_does_not_crash():
    fake_records = [{"seq_id": "1", "symbol": None, "an_dt": "10-Jan-2024 10:00:00",
                      "exchdisstime": "10-Jan-2024 10:00:00", "desc": "Updates",
                      "attchmntText": "text", "attchmntFile": None, "sm_isin": None}]
    with patch(
        "nifty_analyser.documents.ingest.fetch_announcements_window_universe",
        return_value=fake_records,
    ):
        result = ingest_announcements_universe(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))
    assert result.empty
