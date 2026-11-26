"""Delivery-% features (India-specific: fraction of traded volume actually delivered).

The report is published at 21:00 IST, after the 15:30 bar. Merged on (instrument, date); the merged
row's ``available_time`` becomes the later of the two, so a model is never handed the delivery
figure before it was really published."""
from __future__ import annotations

import numpy as np
import pandas as pd

DELIVERY_FEATURE_COLUMNS = ["delivery_pct", "delivery_pct_5d", "delivery_pct_20d", "delivery_pct_z60", "delivery_qty_ratio_20d"]


def add_delivery_indicators(delivery: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in delivery.groupby("instrument_id", sort=False):
        g = g.sort_values("event_time").copy()
        p = g["delivery_pct"]
        g["delivery_pct_5d"], g["delivery_pct_20d"] = p.rolling(5, min_periods=3).mean(), p.rolling(20, min_periods=10).mean()
        g["delivery_pct_z60"] = (p - p.rolling(60, min_periods=30).mean()) / p.rolling(60, min_periods=30).std().replace(0, np.nan)
        g["delivery_qty_ratio_20d"] = g["delivery_qty"] / g["delivery_qty"].rolling(20, min_periods=10).mean().replace(0, np.nan)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def merge_delivery_into_features(features: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    right = indicators[["instrument_id", "event_time", "available_time", *DELIVERY_FEATURE_COLUMNS]].rename(
        columns={"available_time": "_delivery_available"})
    merged = features.merge(right, on=["instrument_id", "event_time"], how="left")
    has = merged["_delivery_available"].notna()
    merged.loc[has, "available_time"] = merged.loc[has, ["available_time", "_delivery_available"]].max(axis=1)
    return merged.drop(columns="_delivery_available")
