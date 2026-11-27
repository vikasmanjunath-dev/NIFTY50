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
    assert "- 5-day: 2 graded, accuracy 50.0%, AUC -, price range held 100%" in graded
