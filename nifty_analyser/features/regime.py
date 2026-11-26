"""Rule-based market regime (Blueprint §3.8, the blueprint's own "acceptable design (a)").

Trend: Nifty 50 index close vs its 50/200-day SMAs. Volatility: India VIX's 252-day rolling
percentile, split into terciles. Market-wide (not per-instrument), merged as-of on availability.
Numeric codes are provided so models can use them; the string labels are for segmentation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_regime(index_bars: pd.DataFrame) -> pd.DataFrame:
    g = index_bars.sort_values("event_time").reset_index(drop=True)
    c = g["close"]
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    up = (c > s50) & (s50 > s200)
    down = (c < s50) & (s50 < s200)
    label = np.where(s200.isna(), None, np.where(up, "Uptrend", np.where(down, "Downtrend", "Range")))
    code = np.where(s200.isna(), np.nan, np.where(up, 1.0, np.where(down, -1.0, 0.0)))
    return pd.DataFrame({"available_time": g["available_time"], "trend_regime": label, "trend_regime_code": code})


def vol_regime(vix_indicators: pd.DataFrame, col: str = "india_vix_percentile_252d") -> pd.DataFrame:
    g = vix_indicators.sort_values("event_time").reset_index(drop=True)
    p = g[col]
    label = np.where(p.isna(), None, np.where(p < 1 / 3, "LowVol", np.where(p < 2 / 3, "MidVol", "HighVol")))
    code = np.where(p.isna(), np.nan, np.where(p < 1 / 3, 0.0, np.where(p < 2 / 3, 1.0, 2.0)))
    return pd.DataFrame({"available_time": g["available_time"], "vol_regime": label, "vol_regime_code": code})


def add_regime_features(features: pd.DataFrame, index_bars: pd.DataFrame, vix_indicators: pd.DataFrame) -> pd.DataFrame:
    out = features.reset_index(drop=True).assign(_ord=lambda d: np.arange(len(d))).sort_values("available_time")
    for table in (trend_regime(index_bars), vol_regime(vix_indicators)):
        out = pd.merge_asof(out, table.sort_values("available_time"), on="available_time", direction="backward")
    out["regime"] = out["trend_regime"].astype(str) + "/" + out["vol_regime"].astype(str)
    return out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
