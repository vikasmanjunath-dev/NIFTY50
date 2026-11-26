"""Tests for the rebuilt Phase 2 layers: macro, regime, cross-sectional, calibration,
selective prediction, ensemble, costs, drift."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_analyser.data.macro import available_time_for, validate_and_tag_macro
from nifty_analyser.eval.costs import CostAssumption, cost_stress_table, portfolio_returns, signal_from_proba
from nifty_analyser.features.cross_sectional import add_index_relative_features, add_same_day_cross_sectional
from nifty_analyser.features.macro_features import add_macro_series_indicators, merge_macro_into_features
from nifty_analyser.features.regime import add_regime_features, trend_regime, vol_regime
from nifty_analyser.models.calibration import expected_calibration_error, fit_calibrator
from nifty_analyser.models.ensemble import fit_ensemble
from nifty_analyser.models.selective import abstain_mask, confidence, coverage_precision_curve, threshold_for_coverage
from nifty_analyser.monitoring.drift import classify_psi, feature_psi_table, psi
from nifty_analyser.validation.walk_forward import carve_calibration_slice

IST = "Asia/Kolkata"


def _equity(n=300, iid="NSE:A", seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2023-01-02", periods=n, tz=IST)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"instrument_id": iid, "event_time": d, "available_time": d + pd.Timedelta(hours=15, minutes=30), "close": close})


# ---------------- macro ----------------
def test_macro_availability_lags_per_series():
    d = pd.Timestamp("2026-01-05")
    assert available_time_for("india_vix", d) == pd.Timestamp("2026-01-05 15:30", tz=IST)
    assert available_time_for("brent_crude", d) == pd.Timestamp("2026-01-06 09:00", tz=IST)


def test_macro_tagging_flags_nonpositive_and_keeps_zero_volume_irrelevant():
    raw = pd.DataFrame({"close": [15.0, -1.0, 16.0]}, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]))
    out = validate_and_tag_macro(raw, "india_vix")
    assert out["quality_flag"].tolist() == ["PASS", "FAIL", "PASS"]


def test_macro_merge_never_uses_unpublished_value_and_carries_forward():
    d = pd.bdate_range("2023-01-02", periods=60, tz=IST)
    macro = pd.DataFrame({"event_time": d, "available_time": d + pd.Timedelta(days=1, hours=9), "close": np.linspace(75.0, 90.0, 60)})
    feats = _equity(60)
    ind = add_macro_series_indicators(macro, "usd_inr")
    m = merge_macro_into_features(feats, ind, "usd_inr")
    # value for day i becomes available next morning, so equity row i (15:30 same day) must see day i-1's value
    assert pd.isna(m["usd_inr_ret_1d"].iloc[0]) and pd.isna(m["usd_inr_ret_1d"].iloc[1])      # ret_1d of day 0 is itself NaN
    assert np.allclose(m["usd_inr_ret_20d"].iloc[21:].to_numpy(), ind["usd_inr_ret_20d"].iloc[20:-1].to_numpy())
    assert "usd_inr_level" not in m and len(m) == len(feats)          # raw level is never a model input


def test_macro_features_are_scale_free():
    """Rescaling the series (e.g. USD-INR quoted in paise) must not change any merged feature."""
    d = pd.bdate_range("2022-01-03", periods=300, tz=IST)
    base = pd.DataFrame({"event_time": d, "available_time": d, "close": 80 + np.random.default_rng(3).normal(0, 1, 300).cumsum() * 0.2})
    a, b = add_macro_series_indicators(base, "x"), add_macro_series_indicators(base.assign(close=base["close"] * 100), "x")
    cols = [c for c in a.columns if c.startswith("x_") and not c.endswith("_level")]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_macro_percentile_has_no_lookahead():
    d = pd.bdate_range("2022-01-03", periods=400, tz=IST)
    full = pd.DataFrame({"event_time": d, "available_time": d, "close": 15 + np.random.default_rng(1).normal(0, 1, 400).cumsum() * 0.1 + 30})
    a = add_macro_series_indicators(full, "v")["v_percentile_252d"]
    b = add_macro_series_indicators(full.iloc[:-40], "v")["v_percentile_252d"]
    pd.testing.assert_series_equal(a.iloc[:len(b)].reset_index(drop=True), b.reset_index(drop=True))


# ---------------- regime ----------------
def test_trend_regime_uptrend_downtrend():
    d = pd.bdate_range("2022-01-03", periods=320, tz=IST)
    up = pd.DataFrame({"event_time": d, "available_time": d, "close": np.linspace(100, 300, 320)})
    dn = up.assign(close=np.linspace(300, 100, 320))
    assert trend_regime(up)["trend_regime"].iloc[-1] == "Uptrend"
    assert trend_regime(dn)["trend_regime"].iloc[-1] == "Downtrend"
    assert trend_regime(up)["trend_regime"].iloc[:100].isna().all()  # needs 200 bars of history


def test_vol_regime_terciles_and_merge_is_asof():
    d = pd.bdate_range("2022-01-03", periods=320, tz=IST)
    vix = pd.DataFrame({"event_time": d, "available_time": d + pd.Timedelta(hours=15, minutes=30), "india_vix_percentile_252d": np.linspace(0, 1, 320)})
    assert vol_regime(vix)["vol_regime"].iloc[0] == "LowVol" and vol_regime(vix)["vol_regime"].iloc[-1] == "HighVol"
    idx = pd.DataFrame({"event_time": d, "available_time": d + pd.Timedelta(hours=15, minutes=30), "close": np.linspace(100, 300, 320)})
    out = add_regime_features(_equity(320).assign(event_time=d, available_time=d + pd.Timedelta(hours=15, minutes=30)), idx, vix)
    assert {"trend_regime", "vol_regime", "regime", "trend_regime_code"} <= set(out.columns) and len(out) == 320


# ---------------- cross-sectional ----------------
def test_beta_of_index_clone_is_one_and_relative_strength_zero():
    idx = _equity(400, "IDX")
    stock = idx.assign(instrument_id="NSE:CLONE")
    for h in (5, 20, 60):
        stock[f"ret_{h}d"] = np.log(stock["close"]).diff(h)
    out = add_index_relative_features(stock, idx)
    assert out["beta_idx_120d"].dropna().iloc[-1] == pytest.approx(1.0)
    assert out["rel_strength_20d"].dropna().abs().max() < 1e-12


def test_same_day_ranks_and_sector_relative_return():
    rows = [{"event_time": pd.Timestamp("2024-01-01", tz=IST), "instrument_id": f"NSE:{s}", "ret_5d": r, "ret_20d": r, "rsi_14": r, "volume_ratio_20d": r}
            for s, r in [("TCS", 0.1), ("INFY", 0.3), ("HDFCBANK", 0.2), ("ICICIBANK", 0.4)]]
    out = add_same_day_cross_sectional(pd.DataFrame(rows)).set_index("instrument_id")
    assert out.loc["NSE:ICICIBANK", "xs_rank_ret_5d"] == 1.0 and out.loc["NSE:TCS", "xs_rank_ret_5d"] == 0.25
    assert out.loc["NSE:INFY", "sector_rel_ret_5d"] == pytest.approx(0.1)   # IT peers: TCS .1, INFY .3 -> mean .2
    assert out.loc["NSE:TCS", "sector_rel_ret_5d"] == pytest.approx(-0.1)


# ---------------- calibration ----------------
def test_calibrated_probabilities_are_clipped_never_certain():
    rng = np.random.default_rng(0)
    s = rng.random(500)
    y = (s > 0.5).astype(int)  # perfectly separable -> raw isotonic hits exactly 0/1
    cal = fit_calibrator(s, y, "isotonic")
    p = cal.apply(np.array([0.0, 0.5, 1.0]))
    assert p.min() >= 0.01 and p.max() <= 0.99


def test_calibration_reduces_ece_on_overconfident_scores():
    rng = np.random.default_rng(1)
    true_p = rng.random(4000)
    y = (rng.random(4000) < true_p).astype(int)
    overconfident = np.clip(0.5 + (true_p - 0.5) * 3, 0.001, 0.999)
    cal = fit_calibrator(overconfident[:2000], y[:2000], "sigmoid")
    assert expected_calibration_error(y[2000:], cal.apply(overconfident[2000:])) < expected_calibration_error(y[2000:], overconfident[2000:])


# ---------------- selective ----------------
def test_confidence_and_abstention():
    assert confidence([0.5, 1.0, 0.0, 0.75]).tolist() == [0.0, 1.0, 1.0, 0.5]
    assert abstain_mask([0.5, 0.9], 0.5).tolist() == [True, False]


def test_coverage_precision_improves_with_threshold_on_informative_scores():
    rng = np.random.default_rng(2)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    rows = coverage_precision_curve(y, p, (0.0, 0.4, 0.8))
    assert rows[0].coverage == 1.0 and rows[0].coverage > rows[1].coverage > rows[2].coverage
    assert rows[2].accuracy_on_acted > rows[0].accuracy_on_acted
    t = threshold_for_coverage(p, 0.3)
    assert (~abstain_mask(p, t)).mean() == pytest.approx(0.3, abs=0.02)


# ---------------- ensemble ----------------
def _signal_frame(n_days=500, n_names=8, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2021-01-01", periods=n_days, tz=IST)
    rows = []
    for i in range(n_names):
        x = rng.normal(size=(n_days, 3))
        y = (x[:, 0] + 0.5 * rng.normal(size=n_days) > 0).astype(int)   # learnable from f0
        rows.append(pd.DataFrame({"instrument_id": f"I{i}", "event_time": d, "out": d + pd.Timedelta(days=5), "f0": x[:, 0], "f1": x[:, 1], "f2": x[:, 2], "y": y}))
    return pd.concat(rows, ignore_index=True).sort_values("event_time").reset_index(drop=True)


def test_carve_calibration_slice_is_purged_and_disjoint():
    df = _signal_frame()
    fit, cal = carve_calibration_slice(df, np.arange(len(df)), 0.3, "out", pd.Timedelta(days=5))
    start = df["event_time"].iloc[cal].min()
    assert not set(fit) & set(cal)
    assert df["out"].iloc[fit].max() < start and df["event_time"].iloc[fit].max() < start - pd.Timedelta(days=5)


def test_ensemble_slices_are_mutually_leakage_free_and_model_beats_random():
    df = _signal_frame()
    train, val = np.flatnonzero(df["event_time"] < df["event_time"].quantile(0.7)), np.flatnonzero(df["event_time"] >= df["event_time"].quantile(0.75))
    ens = fit_ensemble(df, ["f0", "f1", "f2"], "y", train, "out", pd.Timedelta(days=5), base_models=("logistic_elasticnet", "lightgbm"))
    s = ens.slices
    for early, late in (("fit", "meta"), ("meta", "calibration")):
        assert df["out"].iloc[s[early]].max() < df["event_time"].iloc[s[late]].min()
    p = ens.predict_proba_up(df[["f0", "f1", "f2"]].to_numpy()[val])
    assert ((p > 0.5) == df["y"].to_numpy()[val]).mean() > 0.65
    assert p.min() >= 0.01 and p.max() <= 0.99


# ---------------- costs ----------------
def _preds(positions, rets, names=("A", "B")):
    d = pd.bdate_range("2024-01-01", periods=len(positions), tz=IST)
    return pd.concat([pd.DataFrame({"instrument_id": n, "event_time": d, "pos": positions, "ret": rets}) for n in names], ignore_index=True)


def test_constant_position_pays_only_entry_cost():
    r = portfolio_returns(_preds([1.0] * 10, [0.0] * 10), "pos", "ret", CostAssumption(10, 10))
    assert r["turnover"].iloc[0] == 1.0 and (r["turnover"].iloc[1:] == 0).all()
    assert r["net"].iloc[0] == pytest.approx(-20 / 1e4) and (r["net"].iloc[1:] == 0).all()


def test_flip_flopping_costs_more_and_stress_degrades_sharpe_monotonically():
    rng = np.random.default_rng(3)
    ret = rng.normal(0.0005, 0.01, 300)
    flip = _preds(np.tile([1.0, -1.0], 150), ret)
    tbl = cost_stress_table(flip, "pos", "ret", CostAssumption(), (1.0, 2.0, 5.0))
    assert tbl["annualised_sharpe"].is_monotonic_decreasing and tbl["mean_daily_turnover"].nunique() == 1
    assert tbl["mean_daily_turnover"].iloc[0] > 1.5


def test_signal_from_proba():
    assert signal_from_proba([0.0, 0.5, 1.0]).tolist() == [-1.0, 0.0, 1.0]
    assert signal_from_proba([0.2, 0.9], "sign").tolist() == [-1.0, 1.0]


# ---------------- drift ----------------
def test_psi_identical_near_zero_and_shifted_large():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(2, 1, 5000)) > 0.25
    assert classify_psi(0.05) == "stable" and classify_psi(0.15) == "watch" and classify_psi(0.4) == "significant"


def test_feature_psi_table_ranks_the_drifted_feature_first():
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(0, 1, 3000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(3, 1, 3000)})
    t = feature_psi_table(ref, cur, ["a", "b"])
    assert t["feature"].iloc[0] == "b" and t["status"].iloc[0] == "significant" and t["status"].iloc[1] == "stable"
