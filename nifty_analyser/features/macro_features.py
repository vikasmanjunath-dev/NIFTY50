"""Macro features (Blueprint §2.5), scale-free only: 1/5/20-day log returns, log deviation from the 252-day mean
(the level, detrended), and the 252-day rolling percentile. The raw level (USD-INR, Brent, VIX) trends by nature and is
not a model input.

Merge rule: an AS-OF (backward) join on ``available_time`` -- each equity row sees the latest
macro value already knowable at that row's own availability time. Handles mismatched trading
calendars (US/FX holidays) by carrying the last known value forward, and cannot look ahead
by construction. The equity row's ``available_time`` is unchanged (anything merged is by
definition already available then)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_percentile(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    return values.rolling(window, min_periods=min_periods).apply(lambda w: float((w <= w[-1]).mean()), raw=True)


MACRO_FEATURE_SUFFIXES = ("ret_1d", "ret_5d", "ret_20d", "dev_252d", "percentile_252d")


def add_macro_series_indicators(macro_df: pd.DataFrame, series_name: str) -> pd.DataFrame:
    out = macro_df.sort_values("event_time").reset_index(drop=True).copy()
    logc = np.log(out["close"])
    out[f"{series_name}_level"] = out["close"]          # kept for inspection; not merged as a feature
    for k in (1, 5, 20):
        out[f"{series_name}_ret_{k}d"] = logc.diff(k)
    out[f"{series_name}_dev_252d"] = logc - np.log(out["close"].rolling(252, min_periods=60).mean())
    out[f"{series_name}_percentile_252d"] = _rolling_percentile(out["close"], 252, 60)
    return out


def merge_macro_into_features(features: pd.DataFrame, indicators: pd.DataFrame, series_name: str) -> pd.DataFrame:
    cols = [f"{series_name}_{k}" for k in MACRO_FEATURE_SUFFIXES]
    left = features.reset_index(drop=True).assign(_ord=lambda d: np.arange(len(d))).sort_values("available_time")
    right = indicators[["available_time", *cols]].sort_values("available_time")
    merged = pd.merge_asof(left, right, on="available_time", direction="backward")
    return merged.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def add_all_macro_features(features: pd.DataFrame, macro_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = features
    for name, df in macro_frames.items():
        out = merge_macro_into_features(out, add_macro_series_indicators(df, name), name)
    return out
