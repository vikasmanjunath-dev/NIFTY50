"""Tests for NSE announcements ingestion logic (no network access required)."""
from __future__ import annotations

import pandas as pd
import pytest

from nifty_analyser.data.schemas import QualityFlag
from nifty_analyser.documents.ingest import _date_windows, _to_document_row


def test_date_windows_cover_the_full_range_without_gaps():
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2022-12-31")
    windows = _date_windows(start, end, window_days=45)

    assert windows[0][0] == start
    assert windows[-1][1] == end
    for (s1, e1), (s2, _) in zip(windows, windows[1:]):
        assert s2 == e1 + pd.Timedelta(days=1)  # no gap, no overlap


def test_date_windows_respect_max_width():
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2022-12-31")
    windows = _date_windows(start, end, window_days=45)
    for s, e in windows:
        assert (e - s).days < 45


def test_date_windows_single_short_range():
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2022-01-10")
    windows = _date_windows(start, end, window_days=45)
    assert windows == [(start, end)]


def _raw_record(**overrides) -> dict:
    base = dict(
        seq_id="106783625",
        an_dt="17-Sep-2026 19:50:37",
        exchdisstime="17-Sep-2026 19:50:38",
        desc="Updates",
        attchmntText="Some real announcement text.",
        attchmntFile="https://nsearchives.nseindia.com/corporate/some.pdf",
        sm_isin="INE002A01018",
    )
    base.update(overrides)
    return base


def test_to_document_row_basic_mapping():
    row = _to_document_row(_raw_record(), "NSE:RELIANCE")
    assert row["doc_id"] == "NSE_ANN_106783625"
    assert row["category"] == "Updates"
    assert row["quality_flag"] == QualityFlag.PASS.value


def test_to_document_row_available_time_uses_exchdisstime_not_an_dt():
    row = _to_document_row(_raw_record(an_dt="17-Sep-2026 19:50:30", exchdisstime="17-Sep-2026 19:50:38"), "NSE:RELIANCE")
    assert row["available_time"].second == 38
    assert row["event_time"].second == 30


def test_to_document_row_falls_back_to_an_dt_when_exchdisstime_missing():
    row = _to_document_row(_raw_record(exchdisstime=None), "NSE:RELIANCE")
    assert row["quality_flag"] == QualityFlag.WARN.value
    assert row["available_time"] == row["event_time"]


def test_to_document_row_fails_when_no_text_at_all():
    row = _to_document_row(_raw_record(attchmntText=None, desc=None), "NSE:RELIANCE")
    assert row["quality_flag"] == QualityFlag.FAIL.value


def test_to_document_row_falls_back_to_desc_when_no_attachment_text():
    row = _to_document_row(_raw_record(attchmntText=None), "NSE:RELIANCE")
    assert row["text"] == "Updates"
    assert row["quality_flag"] == QualityFlag.PASS.value
