import numpy as np, pandas as pd
from nifty_analyser.eval.spread_estimators import abdi_ranaldo, corwin_schultz, half_spread_bps_by_instrument


def _bars(n=400, spread=0.004, vol=0.01, seed=0):
    rng = np.random.default_rng(seed); mid = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    hi = mid * (1 + np.abs(rng.normal(0, vol / 2, n))) * (1 + spread / 2); lo = mid * (1 - np.abs(rng.normal(0, vol / 2, n))) * (1 - spread / 2)
    return pd.DataFrame({"high": hi, "low": lo, "close": mid, "event_time": pd.date_range("2024-01-01", periods=n, freq="B", tz="Asia/Kolkata")})


def test_cs_nonnegative_and_wider_spread_gives_wider_estimate():
    a, b = _bars(spread=0.0005), _bars(spread=0.02)
    sa, sb = corwin_schultz(a.high, a.low), corwin_schultz(b.high, b.low)
    assert (sa.dropna() >= 0).all() and sb.mean() > sa.mean()


def test_per_instrument_table_and_short_history_skipped():
    d = pd.concat([_bars().assign(instrument_id="A"), _bars(n=30).assign(instrument_id="B")])
    out = half_spread_bps_by_instrument(d)
    assert list(out.instrument_id) == ["A"] and out.cs_half_spread_bps.iloc[0] > 0 and np.isfinite(abdi_ranaldo(d.high, d.low, d.close)).any()
