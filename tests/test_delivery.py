"""Tests for NSE delivery ingestion (bhavcopy) and delivery features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_analyser.data.bhavcopy import parse_bhavcopy
from nifty_analyser.features.delivery import add_delivery_indicators, merge_delivery_into_features

HEADER = "SYMBOL, SERIES, DATE1, PREV_CLOSE, TTL_TRD_QNTY, DELIV_QTY, DELIV_PER"


def _csv(date="17-Nov-2026", rows=None):
    rows = rows or [("RELIANCE", "EQ", 1000, 600, 60.0), ("TCS", "EQ", 500, "-", "-"), ("RELIANCE", "BE", 5, 5, 100.0),
                    ("UNKNOWNCO", "EQ", 10, 5, 50.0), ("INFY", "EQ", 800, 900, 112.5)]
    return HEADER + "\n" + "\n".join(f"{s}, {ser}, {date}, 1, {q}, {dq}, {dp}" for s, ser, q, dq, dp in rows)


def test_parse_keeps_only_pool_eq_rows_and_tags_bad_values_fail():
    out = parse_bhavcopy(_csv(), pd.Timestamp("2026-11-17"))
    assert set(out["instrument_id"]) == {"NSE:RELIANCE", "NSE:TCS", "NSE:INFY"}   # no BE series, no unknown symbol
    flags = out.set_index("instrument_id")["quality_flag"]
    assert flags["NSE:RELIANCE"] == "PASS" and flags["NSE:TCS"] == "FAIL" and flags["NSE:INFY"] == "FAIL"  # '-' and 112.5% rejected


def test_stale_previous_day_file_is_discarded_not_mislabelled():
    """Regression for a real NSE behaviour: HTTP 200 + the PREVIOUS day's file for a non-trading date."""
    assert parse_bhavcopy(_csv(date="14-Nov-2026"), pd.Timestamp("2026-11-17")).empty


def test_available_time_is_after_event_time_at_2100_ist():
    out = parse_bhavcopy(_csv(), pd.Timestamp("2026-11-17"))
    assert (out["available_time"] - out["event_time"] == pd.Timedelta(hours=21)).all()


def _delivery(n=80):
    d = pd.bdate_range("2024-01-01", periods=n, tz="Asia/Kolkata")
    return pd.DataFrame({"instrument_id": "NSE:A", "event_time": d, "available_time": d + pd.Timedelta(hours=21),
                         "delivery_pct": np.linspace(30, 60, n), "delivery_qty": np.linspace(100, 200, n)})


def test_delivery_indicators_have_no_lookahead():
    full = add_delivery_indicators(_delivery(80))
    trunc = add_delivery_indicators(_delivery(80).iloc[:-20])
    cols = ["delivery_pct_5d", "delivery_pct_20d", "delivery_pct_z60", "delivery_qty_ratio_20d"]
    pd.testing.assert_frame_equal(full.iloc[:60][cols].reset_index(drop=True), trunc[cols].reset_index(drop=True))


def test_merge_pushes_available_time_to_the_later_of_the_two_and_keeps_unmatched_rows():
    d = pd.bdate_range("2024-01-01", periods=80, tz="Asia/Kolkata")
    feats = pd.DataFrame({"instrument_id": "NSE:A", "event_time": d, "available_time": d + pd.Timedelta(hours=15, minutes=30)})
    m = merge_delivery_into_features(feats, add_delivery_indicators(_delivery(60)))
    assert len(m) == 80
    assert (m["available_time"].iloc[:60] == d[:60] + pd.Timedelta(hours=21)).all()      # delivery published later -> row waits
    assert (m["available_time"].iloc[60:] == d[60:] + pd.Timedelta(hours=15, minutes=30)).all()  # no delivery -> bar time unchanged
    assert m["delivery_pct"].iloc[60:].isna().all()


def test_xlsx_served_under_csv_url_is_converted_not_dropped():
    """Regression: NSE served an Excel workbook instead of CSV for the real trading day 2022-08-08."""
    import io

    from nifty_analyser.data.bhavcopy import raw_to_csv_text

    df = pd.DataFrame({"SYMBOL": ["RELIANCE"], "SERIES": ["EQ"], "DATE1": [pd.Timestamp("2026-11-17")],
                       "TTL_TRD_QNTY": [1000], "DELIV_QTY": [600], "DELIV_PER": [60.0]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    out = parse_bhavcopy(raw_to_csv_text(buf.getvalue()), pd.Timestamp("2026-11-17"))
    assert out["instrument_id"].tolist() == ["NSE:RELIANCE"] and out["delivery_pct"].iloc[0] == 60.0
    assert raw_to_csv_text(b"SYMBOL,SERIES\nA,EQ\n").startswith("SYMBOL")
