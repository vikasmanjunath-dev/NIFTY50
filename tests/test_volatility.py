import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nifty_analyser.features.volatility import build_volatility_frame, daily_variance
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.vol_forecast import record, score


def bars(n=420, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for iid in ("NSE:AAA", "NSE:BBB"):
        t = pd.date_range("2022-01-03 15:30", periods=n, freq="B", tz="Asia/Kolkata")
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        open_ = close * np.exp(rng.normal(0, 0.003, n))
        frames.append(pd.DataFrame({"instrument_id": iid, "event_time": t.normalize(), "available_time": t, "open": open_,
                                    "high": np.maximum(open_, close) * 1.005, "low": np.minimum(open_, close) * 0.995,
                                    "close": close, "volume": rng.integers(1000, 5000, n).astype(float)}))
    return pd.concat(frames, ignore_index=True)


def test_daily_variance_is_non_negative():
    assert (daily_variance(bars().query("instrument_id=='NSE:AAA'")).dropna() >= 0).all()


@pytest.mark.parametrize("horizon", [5, 20])
def test_features_do_not_change_when_the_future_is_removed(horizon):
    b = bars()
    cut = np.sort(b.event_time.unique())[-30]
    ann = pd.DataFrame({"instrument_id": "NSE:AAA", "category": "Financial Result Updates",
                        "available_time": b.available_time.iloc[[50, 120, 200, 410]].to_numpy()})
    ann["available_time"] = b.available_time.iloc[[50, 120, 200, 410]].reset_index(drop=True)
    full, cols = build_volatility_frame(b, horizon, announcements=ann)
    part, _ = build_volatility_frame(b[b.event_time <= cut], horizon, announcements=ann[ann.available_time <= cut])
    a = full[full.event_time <= cut].set_index(["instrument_id", "event_time"])[cols].sort_index()
    pd.testing.assert_frame_equal(a, part.set_index(["instrument_id", "event_time"])[cols].sort_index())


def test_target_is_next_horizon_volatility_with_matching_outcome_time():
    b = bars()
    df, cols = build_volatility_frame(b, 5)
    g = df[df.instrument_id == "NSE:AAA"].reset_index(drop=True)
    close = b[b.instrument_id == "NSE:AAA"].close.reset_index(drop=True)
    assert g.fwd_rv[300] == pytest.approx(np.sqrt((np.log(close[300:306]).diff() ** 2).sum()))
    assert g.outcome_time[300] == g.available_time[305]
    assert not {"fwd_rv", "outcome_time", "med_rv", "cur_rv"} & set(cols)


def test_ledger_records_once_and_scores_only_matured(tmp_path):
    b = bars()
    dates = np.sort(b.event_time.unique())
    ledger = AuditLog(tmp_path / "v.jsonl")
    def forecast(day):
        return pd.DataFrame({"instrument_id": ["NSE:AAA", "NSE:BBB"], "event_time": pd.Series([day, day]), "horizon_days": 5,
                             "prob_above_usual": [0.9, 0.1], "med_rv": [0.0, 1.0], "forecast_vol_annualised": [0.2, 0.2]})
    early, late = pd.Timestamp(dates[100]), pd.Timestamp(dates[-2])
    assert record(ledger, forecast(early)) and not record(ledger, forecast(early))
    assert record(ledger, forecast(late))
    assert score(ledger, b) == {"5": {"n": 2, "accuracy": 1.0, "auc": 1.0}}
    assert ledger.verify()[0]


def test_volatility_endpoint_statuses(tmp_path):
    from nifty_analyser.ops.api import create_app
    from nifty_analyser.ops.kill_switch import KillSwitch

    class Ctx:
        killswitch = KillSwitch(tmp_path / "k.json")
        audit = AuditLog(tmp_path / "a.jsonl")
        stale_days = 5
        now = pd.Timestamp("2026-01-10", tz="Asia/Kolkata")
        def now_fn(self):
            return self.now

    ctx = Ctx()
    client = TestClient(create_app(ctx, volatility={"5": {"data_date": "2026-01-09", "forecasts": {"NSE:AAA": {"prob_above_usual": 0.7}}}}))
    get = lambda **p: client.get("/v3/volatility", params={"instrument_id": "NSE:AAA", **p})
    assert get().json()["status"] == "OK"
    assert get(instrument_id="NSE:ZZZ").json()["status"] == "REFUSED"
    assert get(horizon_days=7).status_code == 422
    ctx.now = pd.Timestamp("2026-02-01", tz="Asia/Kolkata")
    assert get().json()["status"] == "ABSTAIN"
    ctx.killswitch.trip("test", "t")
    assert get().json()["forecast"] is None
