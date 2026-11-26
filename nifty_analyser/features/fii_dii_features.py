"""FII/DII flow features, scale-free: net flow (and its 5/20-day sums) in units of the series' own typical daily size
(trailing 60-day mean absolute flow). Raw Rs-crore flows grow with market size and are not model inputs.
Merged AS-OF on availability."""
from __future__ import annotations

import numpy as np
import pandas as pd


def flow_feature_columns(series: str) -> list[str]:
    return [f"{series}_net_rel", f"{series}_net_rel_5d", f"{series}_net_rel_20d"]


def add_flow_indicators(flow_df: pd.DataFrame, series_name: str) -> pd.DataFrame:
    out = flow_df.sort_values("event_time").reset_index(drop=True).copy()
    net = out["net_investment_cr"]
    scale = net.abs().rolling(60, min_periods=20).mean().replace(0, np.nan)
    out[f"{series_name}_net_rel"] = net / scale
    out[f"{series_name}_net_rel_5d"] = net.rolling(5, min_periods=3).sum() / scale
    out[f"{series_name}_net_rel_20d"] = net.rolling(20, min_periods=10).sum() / scale
    return out


def merge_flow_into_features(features: pd.DataFrame, indicators: pd.DataFrame, series_name: str) -> pd.DataFrame:
    left = features.reset_index(drop=True).assign(_ord=lambda d: np.arange(len(d))).sort_values("available_time")
    right = indicators[["available_time", *flow_feature_columns(series_name)]].sort_values("available_time")
    merged = pd.merge_asof(left, right, on="available_time", direction="backward")
    return merged.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def add_all_flow_features(features: pd.DataFrame, flow_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = features
    for name, df in flow_frames.items():
        if not df.empty:
            out = merge_flow_into_features(out, add_flow_indicators(df, name), name)
    return out
