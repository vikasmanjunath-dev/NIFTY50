"""Hand-implemented technical indicators (Blueprint §2.3) -- auditable formulas, backward-looking only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _aroon(high: pd.Series, low: pd.Series, n: int = 25):
    up = high.rolling(n + 1).apply(lambda x: 100.0 * np.argmax(x) / n, raw=True)
    dn = low.rolling(n + 1).apply(lambda x: 100.0 * np.argmin(x) / n, raw=True)
    return up, dn


def add_all_technical_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    out: dict[str, pd.Series] = {}
    # Raw price-unit levels (sma_N, ema_N, macd, obv, ...) are never emitted: they trend with the price level,
    # are not comparable across stocks, and make drift monitoring meaningless. Dimensionless forms only.
    sma = {w: c.rolling(w).mean() for w in (5, 10, 20, 50, 100, 200)}
    ema = {w: c.ewm(span=w, adjust=False, min_periods=w).mean() for w in (5, 9, 12, 21, 26, 50, 100, 200)}
    for w, m in sma.items():
        out[f"price_over_sma{w}"] = c / m - 1.0
    for w, m in ema.items():
        out[f"price_over_ema{w}"] = c / m - 1.0
    out["sma50_over_sma200"] = sma[50] / sma[200] - 1.0
    out["golden_cross"] = (sma[50] > sma[200]).astype(float).where(sma[200].notna())
    macd = ema[12] - ema[26]
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_pct"], out["macd_signal_pct"], out["macd_hist_pct"] = macd / c, sig / c, (macd - sig) / c
    rsi = _rsi(c)
    out["rsi_14"], out["rsi_14_slope"] = rsi, rsi.diff(5)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    out["stoch_k"], out["stoch_d"] = k, k.rolling(3).mean()
    out["williams_r"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    for w in (10, 12, 20):
        out[f"roc_{w}"] = c.pct_change(w)
    vol10 = v.rolling(10).sum().replace(0, np.nan)
    out["obv_flow_10"] = (np.sign(c.diff()).fillna(0) * v).rolling(10).sum() / vol10   # signed volume share in [-1, 1]
    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_pctb"] = (c - (m - 2 * sd)) / (4 * sd).replace(0, np.nan)
    out["bb_width"] = 4 * sd / m
    tp = (h + l + c) / 3
    out["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True))
    au, ad = _aroon(h, l)
    out["aroon_up"], out["aroon_down"] = au, ad
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    out["cmf_20"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    out["ad_flow_10"] = (mfm.fillna(0) * v).rolling(10).sum() / vol10
    tr_mid = (h.rolling(20).max() + l.rolling(20).min()) / 2
    out["donchian_pos"] = (c - tr_mid) / (h.rolling(20).max() - l.rolling(20).min()).replace(0, np.nan)
    return pd.concat([g, pd.DataFrame(out, index=g.index)], axis=1)
