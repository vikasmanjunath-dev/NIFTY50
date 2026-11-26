import numpy as np
import pandas as pd
import pytest
from nifty_analyser.ops import paper
from nifty_analyser.ops.audit import AuditLog

TZ = "Asia/Kolkata"


class Stub:
    def predict_proba_up(self, X):
        return X[:, 0]


def _feats(n=30, days=3):
    d = pd.date_range("2026-01-01", periods=days, freq="B", tz=TZ)
    return pd.DataFrame([(f"I{i}", t, t + pd.Timedelta(hours=16), i / n) for t in d for i in range(n)], columns=["instrument_id", "event_time", "available_time", "f"])


def test_book_uses_only_published_data_and_picks_extremes():
    f = _feats()
    as_of = f["available_time"].iloc[0] + pd.Timedelta(minutes=1)          # only day 1 published
    b = paper.target_book(Stub(), f, ["f"], as_of)
    assert b["decision_date"] == "2026-01-01" and b["k"] == 6
    assert b["long"] == [f"I{i}" for i in range(24, 30)] and b["short"] == [f"I{i}" for i in range(6)]
    with pytest.raises(ValueError):
        paper.target_book(Stub(), f, ["f"], f["available_time"].min() - pd.Timedelta(days=1))
    with pytest.raises(ValueError):
        paper.target_book(Stub(), f[f.instrument_id < "I05"], ["f"], f["available_time"].max())    # too few names


def test_ledger_is_append_only_idempotent_tamper_evident_and_spec_hashed(tmp_path):
    led = AuditLog(tmp_path / "l.jsonl")
    b = {"decision_date": "2026-01-01", "n": 30, "k": 6, "long": ["I1"], "short": ["I2"]}
    e = paper.record_book(led, b, "v1")
    assert e["payload"]["spec_hash"] == paper.SPEC_HASH
    with pytest.raises(ValueError):
        paper.record_book(led, b, "v1")
    assert led.verify()[0]
    (tmp_path / "l.jsonl").write_text((tmp_path / "l.jsonl").read_text().replace("I1", "I9"))
    assert not AuditLog(tmp_path / "l.jsonl").verify()[0]


def test_is_due_respects_hold_days(tmp_path):
    led = AuditLog(tmp_path / "l.jsonl")
    dates = [str(d.date()) for d in pd.bdate_range("2026-01-01", periods=40)]
    assert paper.is_due(led, dates[0], dates)
    paper.record_book(led, {"decision_date": dates[0], "n": 30, "k": 6, "long": [], "short": []}, "v")
    assert not paper.is_due(led, dates[19], dates) and paper.is_due(led, dates[20], dates)


def test_realized_scores_only_prices_after_decision_with_costs(tmp_path):
    led = AuditLog(tmp_path / "l.jsonl")
    d = pd.bdate_range("2026-01-01", periods=5, tz=TZ)
    px = {"A": [100, 100, 110, 121, 121], "B": [100, 100, 90, 81, 81], "C": [100] * 5, "D": [100] * 5}
    bars = pd.DataFrame([(i, d[t], px[i][t]) for i in px for t in range(5)], columns=["instrument_id", "event_time", "close"])
    paper.record_book(led, {"decision_date": "2026-01-02", "n": 4, "k": 1, "long": ["A"], "short": ["B"]}, "v")
    r = paper.realized(led, bars)
    assert len(r) == 3                                              # only days AFTER 2026-01-02
    assert r["gross"].iloc[0] == pytest.approx(0.5 * np.log(1.1) - 0.5 * np.log(0.9))
    assert r["net"].iloc[0] < r["gross"].iloc[0] and r["net"].iloc[1] == pytest.approx(r["gross"].iloc[1])
    assert paper.realized(AuditLog(tmp_path / "empty.jsonl"), bars).empty


def test_blend_arm_has_its_own_spec_hash_and_record_uses_it(tmp_path):
    from nifty_analyser.ops.audit import AuditLog
    assert paper.SPEC_BLEND_HASH != paper.SPEC_HASH and paper.SPEC_BLEND["q"] == paper.SPEC["q"]
    led = AuditLog(tmp_path / "l.jsonl")
    e = paper.record_book(led, {"decision_date": "2026-01-02", "n": 50, "k": 10, "short": [], "long": []}, "v", paper.SPEC_BLEND_HASH)
    assert e["payload"]["spec_hash"] == paper.SPEC_BLEND_HASH
