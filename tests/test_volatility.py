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


def test_walk_forward_ranges_use_only_realised_outcomes_and_cover():
    from nifty_analyser.models.ranges import walk_forward_ranges

    rng = np.random.default_rng(1)
    dates = np.repeat(np.arange(600), 20)
    scale = rng.uniform(0.01, 0.05, len(dates))
    returns = scale * rng.standard_normal(len(dates))
    q = walk_forward_ranges(dates, returns, scale, horizon=5)
    m = ~np.isnan(q[:, 0])
    assert np.isnan(q[dates < 250]).all()
    assert abs(((returns >= q[:, 0]) & (returns <= q[:, 3]))[m].mean() - 0.9) < 0.01
    changed = returns.copy()
    changed[dates >= 394] = 10.0  # unrealised at date 400 with a 5-day horizon
    np.testing.assert_array_equal(walk_forward_ranges(dates, changed, scale, horizon=5)[dates == 399], q[dates == 399])


def test_ledger_scores_range_coverage(tmp_path):
    b = bars()
    day = pd.Timestamp(np.sort(b.event_time.unique())[100])
    ledger = AuditLog(tmp_path / "v.jsonl")
    record(ledger, pd.DataFrame({"instrument_id": ["NSE:AAA", "NSE:BBB"], "event_time": pd.Series([day, day]), "horizon_days": 5,
                                 "prob_above_usual": [0.9, 0.1], "med_rv": [0.0, 1.0], "forecast_vol_annualised": [0.2, 0.2],
                                 "move_low_90": [-1.0, 0.5], "move_high_90": [1.0, 0.6]}))
    assert score(ledger, b)["5"]["coverage_90"] == 0.5


def test_dashboard_lists_volatility_and_omits_it_when_absent():
    from nifty_analyser.ops.dashboard import render_dashboard

    health = {"state": "GREEN", "reasons": [], "data_age_days": 1.0, "psi_significant_frac": 0.0}
    block = {"5": {"data_date": "2026-01-09", "forecasts": {"NSE:<A>": {"prob_above_usual": 0.7, "forecast_vol_annualised": 0.25,
             "usual_vol_annualised": 0.2, "move_low_90": -0.05, "move_high_90": 0.06}}}}
    html = render_dashboard(health, None, {}, [], True, None, block)
    assert "NSE:&lt;A&gt;" in html and "-5.0% to 6.0%" in html
    assert "Expected volatility" not in render_dashboard(health, None, {}, [], True)


def test_adaptive_ranges_restore_coverage_and_recalibration_ignores_future():
    from nifty_analyser.models.ranges import adaptive_ranges, walk_forward_recalibrate

    rng = np.random.default_rng(2)
    dates = np.repeat(np.arange(900), 20)
    scale = np.full(len(dates), 0.02)
    returns = scale * rng.standard_normal(len(dates)) * (1 + dates / 900)  # tails keep widening
    q, final = adaptive_ranges(dates, returns, scale, horizon=5, gamma=0.02)
    m = ~np.isnan(q[:, 0])
    assert ((returns >= q[:, 0]) & (returns <= q[:, 3]))[m].mean() > 0.86 and final[0.05] < 0 < final[0.95]
    prob, y = rng.uniform(0.1, 0.9, len(dates)), rng.integers(0, 2, len(dates))
    a = walk_forward_recalibrate(dates, prob, y, horizon=5)
    b = walk_forward_recalibrate(dates, prob, np.where(dates >= 600, 1 - y, y), horizon=5)
    np.testing.assert_array_equal(a[dates < 600], b[dates < 600])


def test_extra_features_do_not_change_when_the_future_is_removed():
    from nifty_analyser.features.volatility_extra import extra_features

    b = bars()
    index = b[b.instrument_id == "NSE:AAA"]
    cut = np.sort(b.event_time.unique())[-30]
    full, cols = extra_features(b, index)
    part, _ = extra_features(b[b.event_time <= cut], index[index.event_time <= cut])
    key = ["instrument_id", "event_time"]
    pd.testing.assert_frame_equal(full[full.event_time <= cut].set_index(key)[cols].sort_index(), part.set_index(key)[cols].sort_index())


def test_history_rows_join_training_only_once_realised(tmp_path, monkeypatch):
    from nifty_analyser.ops import vol_forecast as vf

    old = bars(seed=3)
    for c in ("event_time", "available_time"):
        old[c] = old[c] - pd.Timedelta(days=700)
    old["quality_flag"] = "PASS"
    old.to_parquet(tmp_path / "bars.parquet")
    old[old.instrument_id == "NSE:AAA"][["event_time", "available_time", "close", "quality_flag"]].to_parquet(tmp_path / "india_vix.parquet")
    history = vf.load_history_training(5, tmp_path)
    assert len(history) and history.outcome_time.notna().all() and vf.load_history_training(5, tmp_path / "absent") is None
    sizes = []
    monkeypatch.setattr(vf, "instrument_ids_as_of", lambda d: {"NSE:AAA", "NSE:BBB"})
    monkeypatch.setattr(vf, "membership_mask", lambda df: pd.Series(True, index=df.index))
    monkeypatch.setattr(vf, "fit_predict_classifiers", lambda X, y, V: (sizes.append(len(y)), {"blend": np.full(len(V), 0.5)})[1])
    monkeypatch.setattr(vf, "fit_predict_regressors", lambda X, y, V: {"blend": np.zeros(len(V))})
    live = bars()
    vf.latest_forecasts(live, 5)
    vf.latest_forecasts(live, 5, history=history)
    vf.latest_forecasts(live, 5, history=history, as_of=live.available_time.max())
    assert sizes[1] - sizes[0] == len(history) == sizes[2] - sizes[0]


@pytest.mark.parametrize("horizon,uses_extra", [(5, False), (20, True)])
def test_extra_features_reach_only_the_adopted_classifier(monkeypatch, horizon, uses_extra):
    from nifty_analyser.ops import vol_forecast as vf

    width = {}
    monkeypatch.setattr(vf, "instrument_ids_as_of", lambda d: {"NSE:AAA", "NSE:BBB"})
    monkeypatch.setattr(vf, "membership_mask", lambda df: pd.Series(True, index=df.index))
    monkeypatch.setattr(vf, "fit_predict_classifiers", lambda X, y, V: (width.update(cls=X.shape[1]), {"blend": np.full(len(V), 0.5)})[1])
    monkeypatch.setattr(vf, "fit_predict_regressors", lambda X, y, V: (width.update(reg=X.shape[1]), {"blend": np.zeros(len(V))})[1])
    assert len(vf.latest_forecasts(bars(), horizon)) == 2
    assert (width["cls"] > width["reg"]) is uses_extra


def test_stock_features_do_not_change_when_the_future_is_removed(monkeypatch):
    from nifty_analyser.data import instrument_master
    from nifty_analyser.features.volatility_extra import stock_features

    monkeypatch.setattr(instrument_master, "get_instrument", lambda i: type("I", (), {"sector": "X"})())
    b = bars()
    cut = np.sort(b.event_time.unique())[-30]
    full, cols = stock_features(build_volatility_frame(b, 20)[0], 20)
    part, _ = stock_features(build_volatility_frame(b[b.event_time <= cut], 20)[0], 20)
    key = ["instrument_id", "event_time"]
    pd.testing.assert_frame_equal(full[full.event_time <= cut].set_index(key)[cols].sort_index(), part.set_index(key)[cols].sort_index())


def test_global_features_use_only_what_was_already_published():
    from nifty_analyser.features.volatility_global import add_global_features, stamp

    days = pd.date_range("2022-01-03", periods=300, freq="B")
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    raw = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}, index=days)
    us = stamp(raw, "spx")
    rows = pd.DataFrame({"available_time": (days + pd.Timedelta("15:30:00")).tz_localize("Asia/Kolkata")})
    full, groups = add_global_features(rows, {"spx": us})
    assert full.spx_ret_1.iloc[250] == pytest.approx(np.log(close[249] / close[248]))  # previous US day, not the same date
    changed = raw.copy()
    changed.iloc[250:] *= 5
    again, _ = add_global_features(rows, {"spx": stamp(changed, "spx")})
    pd.testing.assert_frame_equal(full.iloc[:251], again.iloc[:251])
    assert groups == {"us": [c for c in full.columns if c.startswith("spx_")], "index": []}
