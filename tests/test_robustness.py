"""Tests for CPCV, impact-cost model, fast PIT dataset, learned regime."""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from nifty_analyser.eval.costs import ImpactCostModel, portfolio_returns_impact
from nifty_analyser.eval.dataset import attach_labels, liquidity_panel
from nifty_analyser.features.regime_learned import LearnedRegime, market_state
from nifty_analyser.models.baseline import fit_logistic
from nifty_analyser.validation.cpcv import cpcv_evaluate, cpcv_splits

TZ = "Asia/Kolkata"
H = pd.Timedelta(days=7)


def _panel(n_days=240, n_inst=5, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B", tz=TZ)
    rows = []
    for i in range(n_inst):
        x = rng.normal(size=n_days)
        y = (signal * x + rng.normal(size=n_days) > 0).astype(int)
        rows.append(pd.DataFrame({"instrument_id": f"I{i}", "event_time": dates, "x": x, "y": y,
                                  "outcome": dates + H, "ret": rng.normal(0, .01, n_days)}))
    return pd.concat(rows, ignore_index=True).sort_values(["event_time", "instrument_id"]).reset_index(drop=True)


def test_cpcv_counts_and_path_coverage():
    df = _panel()
    splits, paths, gid = cpcv_splits(df, 6, 2, "outcome", pd.Timedelta(days=3))
    assert len(splits) == comb(6, 2) and len(paths) == comb(5, 1)
    for p in paths:                               # every path uses each group exactly once, from a split that tests it
        assert sorted(p) == list(range(6))
        assert all(g in splits[sid].test_groups for g, sid in p.items())
    assert (gid >= 0).all()


def test_cpcv_purge_and_embargo_no_leakage():
    df = _panel()
    emb = pd.Timedelta(days=5)
    splits, _, _ = cpcv_splits(df, 6, 2, "outcome", emb)
    for s in splits:
        tr, te = df.iloc[s.train_idx], df.iloc[s.test_idx]
        assert not set(s.train_idx) & set(s.test_idx)
        te_dates = pd.DatetimeIndex(np.sort(te.event_time.unique()))
        for d in te_dates:
            assert not ((tr.event_time <= d) & (tr.outcome >= d)).any(), "train label overlaps a test date"
            assert not ((tr.event_time > d) & (tr.event_time <= d + emb)).any(), "train row inside embargo"


def test_cpcv_detects_signal_and_not_noise():
    sig = cpcv_evaluate(_panel(signal=2.0), ["x"], fit_logistic, 6, 2, "outcome", H, "ret")
    noise = cpcv_evaluate(_panel(signal=0.0, seed=1), ["x"], fit_logistic, 6, 2, "outcome", H, "ret")
    assert len(sig) == 5
    assert sig["roc_auc"].min() > 0.75
    assert abs(noise["roc_auc"].mean() - 0.5) < 0.05


def _liq_panel(n_days=30, n_inst=3, adv=1e9, sigma=0.02):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B", tz=TZ)
    d = pd.DataFrame([(f"I{i}", t) for i in range(n_inst) for t in dates], columns=["instrument_id", "event_time"])
    d["adv_value"], d["sigma"], d["ret"] = adv, sigma, 0.0
    return d


def _alternating(d):
    d = d.copy()
    d["pos"] = np.where(d.groupby("instrument_id").cumcount() % 2 == 0, 1.0, -1.0)
    return d


def test_impact_zero_trades_zero_cost_and_cost_monotone():
    d = _liq_panel()
    d["pos"] = 1.0
    m = ImpactCostModel(capital=1e8)
    r = portfolio_returns_impact(d, "pos", "ret", d, m)
    assert r["cost"].iloc[1:].abs().max() == 0.0 and r["cost"].iloc[0] > 0     # only the initial build costs
    a = _alternating(d)
    lo = portfolio_returns_impact(a, "pos", "ret", a, ImpactCostModel(capital=1e7))["cost"].sum()
    hi = portfolio_returns_impact(a, "pos", "ret", a, ImpactCostModel(capital=1e10))["cost"].sum()
    assert hi > lo > 0


def test_impact_sqrt_scaling_in_capital():
    a = _alternating(_liq_panel(adv=1e12))                                        # participation tiny -> never capped
    mk = lambda cap: ImpactCostModel(fixed_bps=0, half_spread_bps=0, impact_coef=0.5, capital=cap)
    c1 = portfolio_returns_impact(a, "pos", "ret", a, mk(1e8))["cost"].sum()
    c4 = portfolio_returns_impact(a, "pos", "ret", a, mk(4e8))["cost"].sum()
    # cost = dw * Y*sigma*sqrt(cap*dw/ADV): quadrupling capital doubles per-unit impact (dw unchanged)
    assert c4 / c1 == pytest.approx(2.0, rel=1e-6)


def test_impact_unknown_liquidity_is_charged_conservatively():
    a = _alternating(_liq_panel())
    known = portfolio_returns_impact(a, "pos", "ret", a, ImpactCostModel())["cost"].sum()
    unk = a.assign(adv_value=np.nan)
    unknown = portfolio_returns_impact(unk, "pos", "ret", unk, ImpactCostModel())["cost"].sum()
    assert unknown > known


def test_attach_labels_matches_manual_forward_return():
    dates = pd.date_range("2024-01-01", periods=30, freq="B", tz=TZ)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, .01, 30)))
    bars = pd.DataFrame({"instrument_id": "I0", "event_time": dates, "available_time": dates + pd.Timedelta(hours=16),
                         "open": close, "high": close, "low": close, "close": close, "volume": 1e6})
    serving = pd.DataFrame({"instrument_id": "I0", "event_time": dates, "available_time": dates + pd.Timedelta(hours=16), "f": 1.0})
    out = attach_labels(serving, bars, 5)
    assert len(out) == 25
    first = out.iloc[0]
    assert first["fwd_ret_5d"] == pytest.approx(np.log(close[5] / close[0]))
    assert first["y"] == int(close[5] > close[0])
    assert out["event_time"].is_monotonic_increasing
    assert (out["fwd_ret_5d_outcome_available_time"] > out["event_time"]).all()


def test_liquidity_panel_is_lagged():
    dates = pd.date_range("2024-01-01", periods=40, freq="B", tz=TZ)
    vol = np.full(40, 1e6)
    vol[30] = 1e12                                   # spike on day 30 must not affect day 30's own ADV
    bars = pd.DataFrame({"instrument_id": "I0", "event_time": dates, "close": 100.0, "volume": vol})
    lp = liquidity_panel(bars).set_index("event_time")
    assert lp["adv_value"].iloc[30] == pytest.approx(1e8)
    assert lp["adv_value"].iloc[31] > 1e9


def _regime_state(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cols = ["ret_5d", "ret_20d", "rvol_20d", "drawdown_from_252d_high", "india_vix_percentile_252d", "india_vix_dev_252d"]
    z = np.repeat([0, 1, 2], n // 3 + 1)[:n]
    mu = np.array([[.01, .02, .1, -.02, .2, -.1], [0, 0, .25, -.1, .5, 0], [-.03, -.06, .6, -.3, .9, .5]])
    X = mu[z] + rng.normal(0, .03, (n, 6))
    return pd.DataFrame(X, columns=cols, index=pd.date_range("2020-01-01", periods=n, tz=TZ)), z


def test_learned_regime_recovers_ordered_states_and_probs_sum_to_one():
    s, z = _regime_state()
    p = LearnedRegime(3).fit(s).transform(s)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p.to_numpy().argmax(axis=1) == z).mean() > 0.95        # component 0 = calm ... 2 = most volatile


def test_learned_regime_fit_ignores_data_after_training_window():
    s, _ = _regime_state()
    a = LearnedRegime(3).fit(s.iloc[:250]).transform(s.iloc[250:])
    s2 = s.copy(); s2.iloc[300:] *= 50                           # corrupt the "future"
    b = LearnedRegime(3).fit(s2.iloc[:250]).transform(s.iloc[250:])
    pd.testing.assert_frame_equal(a, b)


def test_market_state_is_one_row_per_date():
    df = pd.DataFrame({"event_time": list(pd.date_range("2024-01-01", periods=3)) * 2, "rvol_20d": [1, 2, 3, 3, 4, 5.0], "ret_5d": 0.0})
    ms = market_state(df)
    assert len(ms) == 3 and ms["rvol_20d"].tolist() == [2.0, 3.0, 4.0]


def test_nested_search_is_leakage_free_deterministic_and_reports_optimism():
    from nifty_analyser.validation.nested import nested_random_search, sample_params
    sp = {"n_estimators": ("int", 20, 40), "learning_rate": ("log", 0.05, 0.2), "num_leaves": ("int", 3, 8),
          "min_child_samples": ("log_int", 10, 100), "subsample": ("float", 0.6, 1.0), "colsample_bytree": ("float", 0.5, 1.0),
          "reg_lambda": ("log", 0.1, 10.0)}
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = sample_params(sp, rng)
        assert all(sp[k][1] <= v <= sp[k][2] for k, v in p.items())
    df = _panel(n_days=300, signal=2.0)
    kw = dict(space=sp, outer_splits=3, inner_splits=2, n_trials=3, seed=1)
    a = nested_random_search(df, ["x"], "outcome", H, **kw)
    b = nested_random_search(df, ["x"], "outcome", H, **kw)
    pd.testing.assert_frame_equal(a.rows.drop(columns="best_params"), b.rows.drop(columns="best_params"))
    s = a.summary()
    assert s["n_outer"] == 3 and s["tuned_auc"] > 0.7 and s["default_auc"] > 0.7      # learnable signal found either way
    assert {"selection_optimism", "tuned_minus_default", "folds_tuned_better"} <= set(s)


def test_path_metrics_date_neutral_measures_detect_cross_sectional_signal_not_base_rate_shift():
    from nifty_analyser.validation.cpcv import path_metrics
    df = _panel(n_days=120, n_inst=20, signal=2.0)
    good = path_metrics(df, 1 / (1 + np.exp(-2 * df["x"].to_numpy())), "ret")
    assert good["within_date_auc"] > 0.7
    # a per-date constant score ranks nothing within a date, however well it tracks the date's base rate
    base = df.groupby("event_time")["y"].transform("mean").to_numpy().clip(0.01, 0.99)
    shift = path_metrics(df, base, "ret")
    assert abs(shift["within_date_auc"] - 0.5) < 1e-9


def test_cpcv_paths_accepts_arbitrary_predictor():
    from nifty_analyser.validation.cpcv import cpcv_paths
    df = _panel()
    paths = cpcv_paths(df, lambda tr, te: np.full(len(te), 0.5), 6, 2, "outcome", H)
    assert len(paths) == 5 and all((p == 0.5).all() for p in paths)
