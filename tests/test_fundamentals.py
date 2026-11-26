"""Tests for fundamentals ingestion (nifty_analyser/data/fundamentals.py).

Uses synthetic XBRL fragments mirroring NSE's real IND-AS taxonomy structure
(including the dual-context "quarterly vs. cumulative" pattern that was
investigated and empirically resolved before writing any extraction code --
see the module docstring in fundamentals.py for the full story).
"""
from __future__ import annotations

import pandas as pd
import pytest

from nifty_analyser.data.fundamentals import (
    _select_first_matching_tag,
    _select_quarterly_value,
)


def _xbrl_fragment(tag: str, quarterly_value: float, cumulative_value: float) -> str:
    return (
        f'<in-bse-fin:{tag} contextRef="OneD" unitRef="INR" decimals="-7">'
        f'{quarterly_value}</in-bse-fin:{tag}>'
        f'<in-bse-fin:{tag} contextRef="FourD" unitRef="INR" decimals="-7">'
        f'{cumulative_value}</in-bse-fin:{tag}>'
    )


def test_selects_smaller_magnitude_as_quarterly_value():
    xml = _xbrl_fragment("RevenueFromOperations", 2438650000000.00, 7155630000000.00)
    value = _select_quarterly_value(xml, "in-bse-fin:RevenueFromOperations")
    assert value == pytest.approx(2438650000000.00)


def test_handles_single_context_tag():
    xml = '<in-bse-fin:EPS contextRef="OneD" unitRef="INR">13.70</in-bse-fin:EPS>'
    value = _select_quarterly_value(xml, "in-bse-fin:EPS")
    assert value == pytest.approx(13.70)


def test_returns_none_when_tag_absent():
    xml = '<in-bse-fin:SomeOtherTag contextRef="OneD">1.0</in-bse-fin:SomeOtherTag>'
    value = _select_quarterly_value(xml, "in-bse-fin:RevenueFromOperations")
    assert value is None


def test_negative_values_compared_by_absolute_magnitude():
    """A loss-making quarter inside a cumulative-positive year: quarterly value
    can be negative but must still be selected over the (larger-magnitude)
    cumulative figure by absolute value, not raw sign.
    """
    xml = _xbrl_fragment("ProfitLossForPeriod", -500.0, 3000.0)
    value = _select_quarterly_value(xml, "in-bse-fin:ProfitLossForPeriod")
    assert value == pytest.approx(-500.0)


def test_malformed_numeric_text_is_skipped_not_crash():
    xml = (
        '<in-bse-fin:EPS contextRef="OneD">NA</in-bse-fin:EPS>'
        '<in-bse-fin:EPS contextRef="FourD">45.0</in-bse-fin:EPS>'
    )
    value = _select_quarterly_value(xml, "in-bse-fin:EPS")
    assert value == pytest.approx(45.0)


def test_falls_back_to_banking_taxonomy_tag_when_general_tag_absent():
    """Regression test for a real, systematic bug found in the full-universe
    backfill: every single banking stock (ICICIBANK, SBIN, KOTAKBANK,
    AXISBANK, INDUSINDBK) returned zero usable data because banks file under
    different XBRL tag names within the same in-bse-fin namespace. Confirmed
    directly against ICICI Bank's real filing: revenue -> InterestEarned,
    profit -> ProfitLossForThePeriod (note "ForThePeriod", not "ForPeriod"),
    eps -> BasicEarningsPerShareBeforeExtraordinaryItems.
    """
    xml = _xbrl_fragment("InterestEarned", 426067200000.00, 1595159200000.00)
    value = _select_first_matching_tag(
        xml, ["in-bse-fin:RevenueFromOperations", "in-bse-fin:InterestEarned"]
    )
    assert value == pytest.approx(426067200000.00)


def test_general_taxonomy_tag_preferred_when_both_present():
    xml = _xbrl_fragment("RevenueFromOperations", 100.0, 400.0)
    value = _select_first_matching_tag(
        xml, ["in-bse-fin:RevenueFromOperations", "in-bse-fin:InterestEarned"]
    )
    assert value == pytest.approx(100.0)


def test_none_when_neither_candidate_tag_present():
    xml = '<in-bse-fin:SomeUnrelatedTag contextRef="OneD">1.0</in-bse-fin:SomeUnrelatedTag>'
    value = _select_first_matching_tag(
        xml, ["in-bse-fin:RevenueFromOperations", "in-bse-fin:InterestEarned"]
    )
    assert value is None


def test_symbol_with_ampersand_is_url_encoded(monkeypatch):
    """Regression test for a real bug found in the full-universe backfill:
    NSE:M&M returned zero filings because the raw "&" in the symbol was
    parsed as a query-string separator, silently corrupting the request URL.
    """
    import nifty_analyser.data.fundamentals as fundamentals_module

    captured_urls = []

    def _fake_http_get(url, max_retries=3):
        captured_urls.append(url)
        return b"[]"

    monkeypatch.setattr(fundamentals_module, "_http_get", _fake_http_get)
    fundamentals_module.fetch_financial_results_list("M&M")

    assert len(captured_urls) == 1
    assert "M&M" not in captured_urls[0]  # must be encoded, not raw
    assert "symbol=M%26M" in captured_urls[0]


def test_cross_validation_flags_mismatch(monkeypatch):
    """End-to-end safety-net test: if our extracted EPS disagrees materially
    with an independently-sourced (mocked) yfinance figure for the same
    quarter, the row must be demoted to WARN, not silently trusted.
    """
    import nifty_analyser.data.fundamentals as fundamentals_module

    class _FakeTicker:
        def __init__(self, *_a, **_k):
            pass

        @property
        def quarterly_income_stmt(self):
            return pd.DataFrame(
                {pd.Timestamp("2024-12-31"): [13.70]}, index=["Basic EPS"]
            )

    monkeypatch.setattr(fundamentals_module.yf if hasattr(fundamentals_module, "yf") else __import__("yfinance"), "Ticker", _FakeTicker, raising=False)

    fundamentals = pd.DataFrame(
        {
            "instrument_id": ["NSE:RELIANCE"],
            "event_time": [pd.Timestamp("2024-12-31", tz="Asia/Kolkata")],
            "eps_basic": [99.0],  # deliberately wrong vs. the mocked 13.70
            "quality_flag": ["PASS"],
            "missing_reason": [None],
        }
    )
    from nifty_analyser.data.instrument_master import get_instrument

    result = fundamentals_module.cross_validate_eps_against_yfinance(
        fundamentals, get_instrument("NSE:RELIANCE")
    )
    assert result.iloc[0]["quality_flag"] == "WARN"
    assert "MISMATCH" in result.iloc[0]["yf_cross_check"]


def test_cross_validation_nan_yfinance_value_is_not_a_mismatch(monkeypatch):
    """Regression test for a real bug found in the full-universe backfill: the
    matched date can exist in yfinance's data while its *value* is itself NaN
    (that line item wasn't populated that quarter) -- this must be treated as
    "nothing to compare", not spuriously reported as a MISMATCH against NaN.
    """
    import nifty_analyser.data.fundamentals as fundamentals_module
    import numpy as np

    class _FakeTicker:
        def __init__(self, *_a, **_k):
            pass

        @property
        def quarterly_income_stmt(self):
            return pd.DataFrame(
                {pd.Timestamp("2024-12-31"): [np.nan]}, index=["Basic EPS"]
            )

    monkeypatch.setattr(__import__("yfinance"), "Ticker", _FakeTicker, raising=False)

    fundamentals = pd.DataFrame(
        {
            "instrument_id": ["NSE:RELIANCE"],
            "event_time": [pd.Timestamp("2024-12-31", tz="Asia/Kolkata")],
            "eps_basic": [13.70],
            "quality_flag": ["PASS"],
            "missing_reason": [None],
        }
    )
    from nifty_analyser.data.instrument_master import get_instrument

    result = fundamentals_module.cross_validate_eps_against_yfinance(
        fundamentals, get_instrument("NSE:RELIANCE")
    )
    assert result.iloc[0]["quality_flag"] == "PASS"  # must NOT be demoted to WARN
    assert result.iloc[0]["yf_cross_check"] == "no_overlap"


def test_cross_validation_passes_matching_value(monkeypatch):
    import nifty_analyser.data.fundamentals as fundamentals_module

    class _FakeTicker:
        def __init__(self, *_a, **_k):
            pass

        @property
        def quarterly_income_stmt(self):
            return pd.DataFrame(
                {pd.Timestamp("2024-12-31"): [13.70]}, index=["Basic EPS"]
            )

    monkeypatch.setattr(__import__("yfinance"), "Ticker", _FakeTicker, raising=False)

    fundamentals = pd.DataFrame(
        {
            "instrument_id": ["NSE:RELIANCE"],
            "event_time": [pd.Timestamp("2024-12-31", tz="Asia/Kolkata")],
            "eps_basic": [13.75],  # within tolerance of 13.70
            "quality_flag": ["PASS"],
            "missing_reason": [None],
        }
    )
    from nifty_analyser.data.instrument_master import get_instrument

    result = fundamentals_module.cross_validate_eps_against_yfinance(
        fundamentals, get_instrument("NSE:RELIANCE")
    )
    assert result.iloc[0]["quality_flag"] == "PASS"
    assert result.iloc[0]["yf_cross_check"] == "MATCH"
