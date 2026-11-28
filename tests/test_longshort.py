import numpy as np
import pandas as pd

from nifty_analyser.eval.costs import ImpactCostModel
from nifty_analyser.eval.longshort import longshort_returns, permutation_null, pivots, sharpe, weights


def _frames(T=120, N=20, skill=1.0, seed=0, adv=1e11):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(T, N))
    ret = 0.002 * skill * z + rng.normal(0, 0.01, (T, N))
    idx = pd.date_range("2024-01-01", periods=T, freq="B")
    mk = lambda a: pd.DataFrame(a, index=idx, columns=[f"I{i}" for i in range(N)])
    return {"score": mk(z), "ret": mk(ret), "adv": mk(np.full((T, N), adv)), "sigma": mk(np.full((T, N), 0.01))}


def test_weights_are_dollar_neutral_and_held_between_rebalances():
    s = np.random.default_rng(1).normal(size=(12, 10))
    w = weights(s, 0.2, 5)
    assert np.allclose(w.sum(axis=1), 0) and np.allclose(np.abs(w).sum(axis=1), 1.0)
    assert (w[0] == w[4]).all() and not (w[4] == w[5]).all()
    assert (w[0] > 0).sum() == 2 and (w[0] < 0).sum() == 2


def test_nan_scores_are_excluded_and_tiny_universe_stays_flat():
    s = np.random.default_rng(2).normal(size=(5, 10)); s[0, :4] = np.nan
    assert (weights(s, 0.2, 5)[0][:4] == 0).all()
    assert (weights(np.random.default_rng(3).normal(size=(3, 3)), 0.2, 5) == 0).all()


def test_market_neutral_ignores_common_move_and_skill_pays_but_noise_does_not():
    pv = _frames(skill=0.0)
    pv["ret"] = pv["ret"] + 0.05                               # huge common market move must cancel
    assert abs(longshort_returns(pv, model=ImpactCostModel(0, 0, 0))["gross"].mean()) < 0.002
    skilled = longshort_returns(_frames(skill=3.0), model=ImpactCostModel(0, 0, 0))
    assert sharpe(skilled["gross"]) > 2.0


def test_costs_reduce_returns_and_scale_with_turnover():
    r = longshort_returns(_frames(skill=3.0))
    assert r["net"].sum() < r["gross"].sum() and r["turnover"].sum() > 0
    hi = longshort_returns(_frames(skill=3.0), model=ImpactCostModel(fixed_bps=50))
    assert hi["net"].sum() < r["net"].sum()


def test_permutation_null_is_centered_and_real_skill_sits_in_its_tail():
    pv = _frames(T=300, skill=3.0)
    null = permutation_null(pv, n=40, model=ImpactCostModel(0, 0, 0))
    real = sharpe(longshort_returns(pv, model=ImpactCostModel(0, 0, 0))["gross"])
    assert abs(null["gross"].mean()) < 0.7 and real > null["gross"].max()


def test_pivots_roundtrip():
    pv = _frames(T=10, N=4)
    long = pv["score"].stack().rename("s").reset_index()
    long.columns = ["event_time", "instrument_id", "s"]
    long["r"] = pv["ret"].stack().to_numpy()
    liq = long[["instrument_id", "event_time"]].assign(adv_value=1e9, sigma=0.01)
    out = pivots(long, "s", "r", liq)
    pd.testing.assert_frame_equal(out["score"], pv["score"], check_names=False, check_freq=False)
