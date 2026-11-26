"""Scale-free drift monitoring: unit-bearing levels are not monitored, market-wide features are monitored
per distinct day, and genuine drift in scale-free features is still caught."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_analyser.monitoring.drift import is_scale_dependent, market_wide_columns, scale_free_drift_table
from nifty_analyser.ops.kill_switch import GREEN, RED, compute_live_health

IST = "Asia/Kolkata"
REAL_LEVELS = ["sma_5", "sma_200", "ema_9", "ema_200", "atr_14", "obv", "macd", "macd_signal", "macd_hist", "obv_slope_10",
               "ad_line_slope_10", "revenue", "profit", "eps_basic", "usd_inr_level", "brent_crude_level", "india_vix_level",
               "fii_net_cr", "fii_net_cr_5d_sum", "dii_net_cr_20d_sum"]
REAL_SCALE_FREE = ["ret_1d", "ret_60d", "rsi_14", "stoch_k", "bb_pctb", "cci_20", "price_over_sma200", "atr_14_pct", "rvol_20d",
                   "drawdown_from_252d_high", "delivery_pct", "delivery_pct_z60", "delivery_qty_ratio_20d", "profit_margin", "eps_growth_yoy",
                   "revenue_growth_yoy", "beta_idx_250d", "rel_strength_20d", "xs_rank_ret_5d", "sector_rel_ret_5d", "usd_inr_ret_1d",
                   "india_vix_percentile_252d", "brent_crude_percentile_252d", "trend_regime_code", "vol_regime_code", "volume_ratio_20d", "macd_hist_pct"]


def test_classification_of_every_real_level_and_ratio_feature():
    assert all(is_scale_dependent(c) for c in REAL_LEVELS)
    assert not any(is_scale_dependent(c) for c in REAL_SCALE_FREE)


def _panel(seed=0, n_days=400, per_day=30):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days, tz=IST)
    d = days.repeat(per_day)
    n = len(d)
    f = pd.DataFrame({"event_time": d, "available_time": d + pd.Timedelta(hours=15)})
    for i in range(12):
        f[f"x{i}"] = rng.normal(size=n)                                     # per-instrument, stationary
    f["market_state"] = np.repeat(rng.normal(size=n_days), per_day)         # one value per day, stationary
    f["sma_20"] = 100 * np.exp(np.repeat(np.linspace(0, 1.2, n_days), per_day)) * rng.lognormal(0, 0.3, n)   # trending level
    f["usd_inr_level"] = np.repeat(np.linspace(75, 90, n_days), per_day)    # market-wide trending level
    return f, days


COLS = [*(f"x{i}" for i in range(12)), "market_state", "sma_20", "usd_inr_level"]


def _health(f, days, **kw):
    return compute_live_health(f, COLS, days[250], days[-1] + pd.Timedelta(days=1), **kw)


def test_trending_levels_do_not_raise_a_false_alarm_and_are_listed_as_not_monitored():
    f, days = _panel()
    h, t = _health(f, days)
    assert h.state == GREEN
    tiers = t.set_index("feature")["tier"]
    assert tiers["sma_20"] == tiers["usd_inr_level"] == "scale_dependent" and tiers["market_state"] == "market_wide" and tiers["x0"] == "instrument"
    assert t.set_index("feature").loc[["sma_20", "usd_inr_level"], "psi"].isna().all()


def test_market_wide_feature_is_detected_by_one_value_per_day():
    f, _ = _panel()
    assert set(market_wide_columns(f, COLS)) == {"market_state", "usd_inr_level"}


def test_stationary_market_wide_feature_has_a_low_false_alarm_rate():
    """PSI always has some sampling noise; with adaptive bins a stationary market-wide feature averages ~0.05 and
    reaches 'significant' (>= 0.25) in only a few percent of runs -- which the health rule (needs >= 15% of features
    significant) absorbs. Measured over 30 seeds, not asserted per-seed."""
    v = []
    for seed in range(30):
        f, days = _panel(seed)
        t = scale_free_drift_table(f, COLS, days[250], days[-1] + pd.Timedelta(days=1))
        v.append(t.set_index("feature").loc["market_state", "psi"])
    v = np.array(v)
    assert v.mean() < 0.08 and (v >= 0.25).mean() <= 0.10


def test_genuine_drift_in_scale_free_features_is_still_caught():
    f, days = _panel()
    now = days[-1] + pd.Timedelta(days=1)
    cur = f["event_time"] > now - pd.Timedelta(days=60)
    for i in range(12):
        f.loc[cur, f"x{i}"] += 4                                           # per-instrument shift
    h, _ = _health(f, days)
    assert h.state == RED
    f2, days2 = _panel()
    cur2 = f2["event_time"] > days2[-1] + pd.Timedelta(days=1) - pd.Timedelta(days=120)
    f2.loc[cur2, "market_state"] += 4                                      # market-wide shift
    t2 = scale_free_drift_table(f2, COLS, days2[250], days2[-1] + pd.Timedelta(days=1))
    assert t2.set_index("feature").loc["market_state", "status"] == "significant"


def test_market_wide_features_need_enough_distinct_days():
    f, days = _panel()
    short = f[f["event_time"] >= days[-45]]
    short = pd.concat([f[f["event_time"] <= days[250]], short])
    t = scale_free_drift_table(short, COLS, days[250], days[-1] + pd.Timedelta(days=1))
    row = t.set_index("feature").loc["market_state"]
    assert np.isnan(row["psi"]) and "too few distinct days" in row["status"]


def test_nothing_assessable_is_amber_not_healthy():
    f, days = _panel()
    h, _ = compute_live_health(f, ["sma_20", "usd_inr_level"], days[250], days[-1] + pd.Timedelta(days=1))
    assert h.state == "AMBER" and "could not be assessed" in h.reasons[0]


# ---------------- empirical calibration ----------------
from nifty_analyser.monitoring.drift import calibrate_drift_thresholds  # noqa: E402
from nifty_analyser.ops.kill_switch import AMBER, Thresholds  # noqa: E402


def _persistent_panel(seed=0, n_days=1000, per_day=20, k=14):
    """Slow-moving regimes (AR(1), phi=0.995) shared by every stock: features autocorrelated across days and
    correlated across stocks -- the situation where textbook PSI cut-offs (which assume independent samples) over-alarm."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days, tz=IST)
    state = np.zeros((n_days, k))
    for t in range(1, n_days):
        state[t] = 0.995 * state[t - 1] + rng.normal(0, 0.2, k)
    d = days.repeat(per_day)
    f = pd.DataFrame({"event_time": d, "available_time": d + pd.Timedelta(hours=15)})
    for i in range(k):
        f[f"x{i}"] = np.repeat(state[:, i], per_day) + rng.normal(0, 1, len(d))
    return f, days, [f"x{i}" for i in range(k)]


def test_textbook_thresholds_overalarm_on_persistent_features_but_calibrated_ones_do_not():
    f, days, cols = _persistent_panel()
    cal = calibrate_drift_thresholds(f, cols)
    s = np.array(cal["series"])
    assert (s >= 0.15).mean() >= 0.7                      # textbook AMBER level would fire most of the time
    th = Thresholds.calibrated(cal)
    assert th.psi_frac_amber == pytest.approx(cal["psi_frac_amber"]) and th.psi_frac_red >= th.psi_frac_amber + 0.10 - 1e-9 or th.psi_frac_red == 1.0
    assert (s >= th.psi_frac_amber).mean() <= 0.35        # ~10% by construction (ties on a discrete share add some); textbook: >= 70%
    assert cal["n_points"] >= 8 and cal["share_max"] <= th.psi_frac_red


def test_calibrated_thresholds_still_flag_a_genuine_structural_break():
    f, days, cols = _persistent_panel(seed=1)
    th = Thresholds.calibrated(calibrate_drift_thresholds(f, cols))
    now = days[-1] + pd.Timedelta(days=1)
    broken = f.copy()
    broken.loc[broken["event_time"] > now - pd.Timedelta(days=60), cols] += 8     # far beyond any historical variation
    assert compute_live_health(broken, cols, days[-8], now, th)[0].state == RED
    assert compute_live_health(f, cols, days[-8], now, th)[0].state in (GREEN, AMBER)


def test_too_little_history_refuses_to_calibrate():
    f, days = _panel()
    with pytest.raises(ValueError, match="need >= 8"):
        calibrate_drift_thresholds(f, COLS)
