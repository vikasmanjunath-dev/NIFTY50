"""Assemble the feature matrix across a universe, strictly per-instrument (Blueprint §2.2-2.3)."""
from __future__ import annotations

import pandas as pd

from nifty_analyser.features.price_volume import add_distance_features, add_return_features, add_volatility_features
from nifty_analyser.features.technical import add_all_technical_features

META_COLS = ["instrument_id", "event_time", "available_time", "source_id", "quality_flag"]
RAW_PRICE_COLS = ["open", "high", "low", "close", "volume"]


def _build_one(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("event_time").reset_index(drop=True)
    return add_all_technical_features(add_distance_features(add_volatility_features(add_return_features(g))))


def build_feature_matrix(bars: pd.DataFrame) -> pd.DataFrame:
    parts = [_build_one(g) for _, g in bars.groupby("instrument_id", sort=False)]
    return pd.concat(parts, ignore_index=True).sort_values(["instrument_id", "event_time"]).reset_index(drop=True)


def feature_columns(features: pd.DataFrame) -> list[str]:
    exclude = set(META_COLS) | set(RAW_PRICE_COLS) | {"revision_id", "missing_reason"}
    return [c for c in features.columns if c not in exclude and pd.api.types.is_numeric_dtype(features[c])]
