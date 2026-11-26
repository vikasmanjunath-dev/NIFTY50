"""Bid-ask spread estimated from public daily OHLC (Corwin & Schultz 2012; Abdi & Ranaldo 2017) - the only spread evidence available
without tick data. Both are noisy and biased (high-low overstates on volatile days; Abdi-Ranaldo needs the close inside the range), so
treat the cross-sectional median over many names/months as a rough UPPER-ish bound for half-spread, never as a fill-level calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily proportional spread S_t from two-day high/low (negative estimates floored at 0, per the paper)."""
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    gamma = np.log(pd.concat([high, high.shift(1)], axis=1).max(axis=1) / pd.concat([low, low.shift(1)], axis=1).min(axis=1)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0.0)


def abdi_ranaldo(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Daily squared-spread estimator from close and mid-range: S^2 = 4 (c_t - mean(eta_t, eta_t+1)) (c_t - eta_t+1) (monthly mean floored at 0 by caller)."""
    eta = (np.log(high) + np.log(low)) / 2
    c = np.log(close)
    return 4 * (c - (eta + eta.shift(-1)) / 2) * (c - eta.shift(-1))


def half_spread_bps_by_instrument(bars: pd.DataFrame, month_freq: str = "M") -> pd.DataFrame:
    """Per instrument: median over months of the monthly-mean Corwin-Schultz half-spread, and Abdi-Ranaldo (sqrt of floored monthly mean S^2), in bps."""
    rows = []
    for inst, g in bars.sort_values("event_time").groupby("instrument_id"):
        g = g.dropna(subset=["high", "low", "close"]); g = g[(g.high >= g.low) & (g.low > 0)]
        if len(g) < 60:
            continue
        idx = pd.DatetimeIndex(g["event_time"]).tz_localize(None).to_period(month_freq)
        cs = corwin_schultz(g.high.reset_index(drop=True), g.low.reset_index(drop=True)).set_axis(idx).groupby(level=0).mean() / 2 * 1e4
        ar = abdi_ranaldo(g.high.reset_index(drop=True), g.low.reset_index(drop=True), g.close.reset_index(drop=True)).set_axis(idx).groupby(level=0).mean().clip(lower=0) ** 0.5 / 2 * 1e4
        rows.append({"instrument_id": inst, "cs_half_spread_bps": float(cs.median()), "ar_half_spread_bps": float(ar.median()), "n_months": int(len(cs))})
    return pd.DataFrame(rows)
