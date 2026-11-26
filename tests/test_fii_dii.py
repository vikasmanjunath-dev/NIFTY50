"""Tests for FII/DII ingestion (mocked NSDL HTML in the real table shape) and features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nifty_analyser.data.fii_dii as mod
from nifty_analyser.features.fii_dii_features import add_all_flow_features, add_flow_indicators, merge_flow_into_features

_HDR = ("<tr><th>Reporting Date</th><th>Debt/Equity</th><th>Investment Route</th><th>Gross Purchases(Rs Crore)</th>"
        "<th>Gross Sales(Rs Crore)</th><th>Net Investment (Rs Crore)</th><th>Net Investment US($) million</th><th>Conversion</th></tr>")


def _fii_html(rows):
    body = "".join(f"<tr><td>{d}</td><td>Equity</td><td>Sub-total</td><td>1</td><td>1</td><td>{n}</td><td>1</td><td>x</td></tr>" for d, n in rows)
    return f"<html><body><table><tr><td>h</td></tr></table><table>{_HDR}{body}</table></body></html>"


def _dii_html():
    h = "<tr><th>Type of DII</th><th>Instrument Type</th><th>Investment Route</th><th>a</th><th>b</th><th>Net Investment (Rs. Crore)</th><th>c</th></tr>"
    rows = "<tr><td>Total</td><td>Equity</td><td>Stock Exchange</td><td>1</td><td>1</td><td>8082.18</td><td>1</td></tr><tr><td>Total</td><td>Equity</td><td>Equity Total</td><td>1</td><td>1</td><td>12356.93</td><td>1</td></tr>"
    return f"<html><body><table><tr><td>h</td></tr></table><table><tr><td>e</td></tr></table><table>{h}{rows}</table></body></html>"


def test_fii_month_parses_negatives_and_ignores_total_rows(monkeypatch):
    html = _fii_html([("01-Jan-2021", "616.58"), ("04-Jan-2021", "(1180.24)"), ("Total for January", "99999"), ("Total for 2021", "99999")])
    monkeypatch.setattr(mod, "_fetch_report_html", lambda *a, **k: html)
    out = mod.fetch_fii_month("29-Jan-2021")
    assert out["net_investment_cr"].tolist() == [pytest.approx(616.58), pytest.approx(-1180.24)]
    assert len(out) == 2


def test_no_data_and_dii_total_row(monkeypatch):
    monkeypatch.setattr(mod, "_fetch_report_html", lambda *a, **k: "No Data Found")
    assert mod.fetch_fii_month("01-Jan-2019").empty and mod.fetch_dii_day("01-May-2026") is None
    monkeypatch.setattr(mod, "_fetch_report_html", lambda *a, **k: _dii_html())
    assert mod.fetch_dii_day("01-Jul-2026") == pytest.approx(12356.93)


def test_ingest_fii_sets_t_plus_1_availability_dedups_and_flags_missing(monkeypatch):
    monkeypatch.setattr(mod, "fetch_fii_month", lambda d: pd.DataFrame({"event_time": [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-04")], "net_investment_cr": [616.58, None]}))
    out = mod.ingest_fii_flow("2021-01-01", "2021-02-28")
    assert len(out) == 2   # both month batches returned the same two days -> deduplicated
    assert out.iloc[0]["available_time"] - out.iloc[0]["event_time"] == pd.Timedelta(days=1)
    # regression: None became NaT inside iterrows; `is not None` wrongly passed it
    assert out["quality_flag"].tolist() == ["PASS", "FAIL"] and "no Equity" in out.iloc[1]["missing_reason"]


def test_ingest_dii_skips_days_without_data(monkeypatch):
    monkeypatch.setattr(mod, "fetch_dii_day", lambda d: 5.0 if d == "01-Jul-2026" else None)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    out = mod.ingest_dii_flow("2026-06-29", "2026-07-01")
    assert len(out) == 1 and out.iloc[0]["event_time"] == pd.Timestamp("2026-07-01", tz="Asia/Kolkata")


def _flow(n=60):
    d = pd.bdate_range("2026-01-01", periods=n, tz="Asia/Kolkata")
    return pd.DataFrame({"event_time": d, "available_time": d + pd.Timedelta(days=1), "net_investment_cr": np.random.default_rng(7).normal(0, 1000, n)})


def _feats(n=60):
    d = pd.bdate_range("2026-01-01", periods=n, tz="Asia/Kolkata")
    return pd.DataFrame({"instrument_id": "NSE:T", "event_time": d, "available_time": d + pd.Timedelta(hours=15, minutes=30)})


def test_rolling_sums_have_no_lookahead():
    a = add_flow_indicators(_flow(60), "fii")["fii_net_rel_20d"]
    b = add_flow_indicators(_flow(60).iloc[:-10], "fii")["fii_net_rel_20d"]
    pd.testing.assert_series_equal(a.iloc[:50].reset_index(drop=True), b.reset_index(drop=True))


def test_merge_is_asof_so_a_row_only_sees_flow_published_before_it():
    f, feats = _flow(30), _feats(30)
    ind = add_flow_indicators(f, "fii")
    m = merge_flow_into_features(feats, ind, "fii")
    assert pd.isna(m["fii_net_rel"].iloc[0])                    # day-0 flow isn't available until day 1 00:00
    assert (m["fii_net_rel"].iloc[1:].to_numpy() == ind["fii_net_rel"].iloc[:-1].to_numpy()).all() or \
        np.allclose(m["fii_net_rel"].iloc[1:].to_numpy(), ind["fii_net_rel"].iloc[:-1].to_numpy(), equal_nan=True)
    assert (m["available_time"] == feats["available_time"]).all() and len(m) == 30
    assert "fii_net_cr" not in m and "net_investment_cr" not in m  # raw Rs-crore flow is never a model input


def test_flow_features_are_scale_free():
    """Rescaling the raw flow (e.g. the market doubling in size) must not change any feature."""
    a = add_flow_indicators(_flow(80), "fii")
    b = add_flow_indicators(_flow(80).assign(net_investment_cr=lambda d: d["net_investment_cr"] * 7.3), "fii")
    cols = ["fii_net_rel", "fii_net_rel_5d", "fii_net_rel_20d"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_all_flow_features_skips_empty_series():
    out = add_all_flow_features(_feats(20), {"fii": _flow(20), "dii": pd.DataFrame()})
    assert "fii_net_rel" in out and "dii_net_rel" not in out
