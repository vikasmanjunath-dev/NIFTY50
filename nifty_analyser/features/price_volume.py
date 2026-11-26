"""Price/volume features (Blueprint §2.2): every value at row t uses data up to and including t only."""
from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_HORIZONS = (1, 2, 3, 5, 10, 20, 60)
VOL_WINDOWS = (5, 10, 20, 60)


def add_return_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    logc = np.log(g["close"])
    for h in RETURN_HORIZONS:
        g[f"ret_{h}d"] = logc.diff(h)
    g["ret_overnight"] = np.log(g["open"] / g["close"].shift(1))
    g["ret_intraday"] = np.log(g["close"] / g["open"])
    g["gap_pct"] = g["open"] / g["close"].shift(1) - 1.0
    g["range_pct"] = (g["high"] - g["low"]) / g["close"]
    return g


def add_volatility_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    r1 = np.log(g["close"]).diff()
    for w in VOL_WINDOWS:
        g[f"rvol_{w}d"] = r1.rolling(w).std()
    prev_close = g["close"].shift(1)
    tr = pd.concat([g["high"] - g["low"], (g["high"] - prev_close).abs(), (g["low"] - prev_close).abs()], axis=1).max(axis=1)
    g["atr_14"] = tr.rolling(14).mean()
    g["atr_14_pct"] = g["atr_14"] / g["close"]
    return g.drop(columns="atr_14")   # the price-unit ATR is not emitted; only the dimensionless atr_14_pct


def add_distance_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for w in (120, 252):
        g[f"drawdown_from_{w}d_high"] = g["close"] / g["close"].rolling(w).max() - 1.0
    g["volume_ratio_20d"] = g["volume"] / g["volume"].rolling(20).mean().replace(0, np.nan)
    return g
