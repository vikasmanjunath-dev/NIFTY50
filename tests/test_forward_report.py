import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from forward_report import render

from nifty_analyser.ops.audit import AuditLog


def test_report_states_what_has_and_has_not_matured(tmp_path):
    ledger = AuditLog(tmp_path / "v.jsonl")
    ledger.append("vol_forecast", {"date": "2026-01-01", "horizon_days": 5, "forecasts": [{}, {}]})
    empty = render(ledger, {})
    assert "| 2026-01-01 | 5-day | 2 | - |" in empty and "Nothing has matured yet." in empty and "intact: yes" in empty
    graded = render(ledger, {"5": {"n": 2, "accuracy": 0.5, "auc": None, "coverage_90": 1.0}})
    assert "- 5-day: 2 graded, accuracy 50.0%, AUC - (too few dates for an interval), price range held 100%" in graded

    ci = render(ledger, {"5": {"n": 9, "accuracy": 0.5, "auc": 0.7, "auc_ci95": [0.55, 0.81]}})
    assert "AUC 0.700 (95% interval 0.55-0.81)" in ci


def test_interval_and_grading_ignore_price_rescaling(tmp_path):
    import numpy as np
    import pandas as pd

    from nifty_analyser.ops.vol_forecast import auc_interval, graded_rows

    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"date": np.repeat([f"2026-01-{d:02d}" for d in range(1, 29)], 20), "y": rng.integers(0, 2, 560)})
    rows["prob"] = 0.5 + 0.3 * (rows.y - 0.5) + rng.normal(0, 0.2, 560)
    low, high = auc_interval(rows, 5)
    assert 0.6 < low < high < 0.95 and auc_interval(rows[rows.date < "2026-01-12"], 5) is None
    days = pd.date_range("2026-01-01", periods=12, freq="B", tz="Asia/Kolkata")
    bars = pd.DataFrame({"instrument_id": "NSE:INFY", "event_time": days, "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 12)))})
    ledger = AuditLog(tmp_path / "v.jsonl")
    ledger.append("vol_forecast", {"date": "2026-01-02", "horizon_days": 5, "forecasts": [
        {"instrument_id": "NSE:INFY", "prob": 0.7, "med_rv": 0.03, "low_90": -0.02, "high_90": 0.02}]})
    split = bars.assign(close=bars.close / 5)  # a 1:5 split restates the whole history
    pd.testing.assert_frame_equal(graded_rows(ledger, bars), graded_rows(ledger, split))
