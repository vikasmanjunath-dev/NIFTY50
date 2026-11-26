"""Tests for insurance-sector PDF fundamentals extraction (one regression test per real bug found)."""
from __future__ import annotations

import pandas as pd
import pytest

import nifty_analyser.data.insurance_fundamentals as mod
from nifty_analyser.data.insurance_fundamentals import (
    extract_period_end, find_first_matching_row_value, ingest_insurance_fundamentals_for_instrument, parse_indian_number,
)


@pytest.mark.parametrize("text,expected", [("52942", 52942.0), ("52,942", 52942.0), ("7 .30", 7.30), ("1 ,086", 1086.0),
                                            ("(1,234)", -1234.0), ("₹5.29", 5.29), ("20.9%", 20.9),
                                            ("NA", None), ("NIL", None), ("-", None), ("", None), (None, None)])
def test_parse_indian_number(text, expected):
    assert parse_indian_number(text) == (pytest.approx(expected) if expected is not None else None)


def test_basic_match_priority_and_fallback():
    rows = [["(iv)", "a) Basic EPS before and after extraordinary items (net of tax expense)", "5.29", "5.19"]]
    assert find_first_matching_row_value(rows, ["basic eps before and after extraordinary items"]) == pytest.approx(5.29)
    rows = [["Profit/(loss) after tax and before Extraordinary Items", "52,942"]]
    assert find_first_matching_row_value(rows, ["net profit/ loss after tax", "profit/(loss) after tax and before extraordinary items"]) == pytest.approx(52942.0)
    assert find_first_matching_row_value([["unrelated", "1"]], ["net profit"]) is None


def test_bug1_leading_serial_number_is_not_the_value():
    rows = [["2", "Net premium income1", None, "25,11,647", "22,31,647"]]
    assert find_first_matching_row_value(rows, ["net premium income"]) == pytest.approx(2511647.0)
    rows = [["29", "Profit/ (loss) after tax and before Extraordinary Items", None, "81,080"]]
    assert find_first_matching_row_value(rows, ["profit/ (loss) after tax and before extraordinary items"]) == pytest.approx(81080.0)


def test_bug2_label_split_mid_word_across_cells():
    rows = [["2 Net premium in", "come1", "1,144,396", "753,848"]]
    assert find_first_matching_row_value(rows, ["net premium income"]) == pytest.approx(1144396.0)


def test_bug3_multiline_segment_cells_do_not_create_false_matches_but_multiline_labels_still_match():
    seg = [None, "Segment A :\nNet Premium\nIncome from Investments2\nOther Income", "", "", ""]
    real = ["2 Net premium in", "come1", None, "1,440,222", "1,313,789"]
    assert find_first_matching_row_value([seg, real], ["net premium income"]) == pytest.approx(1440222.0)
    assert find_first_matching_row_value([seg], ["net premium income"]) is None
    eps = [None, "a) Basic EPS before and after extraordinary items (net of tax\nexpense) for the period", "1.48", "1.54"]
    assert find_first_matching_row_value([eps], ["basic eps before and after extraordinary items"]) == pytest.approx(1.48)


def test_bug4_last_occurrence_wins_when_a_table_is_embedded_twice():
    garbled = ["2 Net premi", "um income1", None, "1,437,93", "8 1,311,091"]
    clean = ["2 Net premium in", "come1", None, "1,440,222", "1,313,789"]
    assert find_first_matching_row_value([garbled, clean], ["net premium income"]) == pytest.approx(1440222.0)


def test_extract_period_end():
    assert extract_period_end("results for the period ended September 30, 2024.") == pd.Timestamp("2024-09-30")
    assert extract_period_end("financial results for the period ended March 31 2024.") == pd.Timestamp("2024-03-31")
    assert extract_period_end("nothing here") is None and extract_period_end(None) is None


def _ann(text="results for the period ended September 30, 2024.", iid="NSE:SBILIFE"):
    t = pd.Timestamp("2024-10-23 14:40:42", tz="Asia/Kolkata")
    return pd.DataFrame({"instrument_id": [iid], "event_time": [t], "available_time": [t], "category": ["Financial Result Updates"],
                         "text": [text], "attachment_url": ["https://example.com/x.pdf"]})


def test_ingest_uses_period_end_as_event_time_and_filing_time_as_available(monkeypatch):
    monkeypatch.setattr(mod, "extract_fundamentals_from_filing", lambda u: {"revenue": 4001504.0, "profit": 52942.0, "eps_basic": 5.29})
    out = ingest_insurance_fundamentals_for_instrument("NSE:SBILIFE", _ann(), pause=0)
    assert out.iloc[0]["event_time"] == pd.Timestamp("2024-09-30", tz="Asia/Kolkata")
    assert out.iloc[0]["available_time"] > out.iloc[0]["event_time"] and out.iloc[0]["quality_flag"] == "PASS"


def test_ingest_fails_gracefully(monkeypatch):
    assert ingest_insurance_fundamentals_for_instrument("NSE:SBILIFE", _ann("no period"), pause=0).iloc[0]["quality_flag"] == "FAIL"
    monkeypatch.setattr(mod, "extract_fundamentals_from_filing", lambda u: (_ for _ in ()).throw(RuntimeError("net")))
    r = ingest_insurance_fundamentals_for_instrument("NSE:SBILIFE", _ann(), pause=0).iloc[0]
    assert r["quality_flag"] == "FAIL" and "PDF fetch/parse error" in r["missing_reason"]
    assert ingest_insurance_fundamentals_for_instrument("NSE:SBILIFE", _ann().iloc[0:0], pause=0).empty
    assert ingest_insurance_fundamentals_for_instrument("NSE:HDFCLIFE", _ann(), pause=0).empty
