"""Fundamental feature engineering (Blueprint §2.4).

Fundamentals update quarterly, not daily, so merging them onto a daily
feature matrix is a *carry-forward as-of* join, not the same-day broadcast
join used for delivery/macro/regime: each daily row should see whatever
fundamentals were most recently *known* (by ``available_time``, the filing's
own broadcast timestamp) at that day's decision time, and nothing published
later. ``pandas.merge_asof(..., direction="backward")`` implements exactly
this and is incapable of looking forward by construction, which is why it is
used here directly rather than a hand-rolled loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FUNDAMENTAL_FEATURE_COLUMNS = [
    "revenue",
    "profit",
    "eps_basic",
    "profit_margin",
    "eps_growth_yoy",
    "revenue_growth_yoy",
    "profit_growth_yoy",
    "eps_ttm",
]

# Rs-valued levels: merged only to derive scale-free features, then dropped (never model inputs).
LEVEL_FUNDAMENTAL_COLUMNS = ["revenue", "profit", "eps_basic", "eps_ttm"]


def add_fundamental_ratios(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Derived ratios computed per-instrument from the raw quarterly figures:
    profit margin (this quarter) and YoY growth (vs. the same quarter one year,
    i.e. 4 quarterly filings, earlier).

    A zero prior-year EPS/revenue (real: some quarters are reported as exactly 0.0) must give NaN,
    not +/-inf: infinities crash the linear model's imputer ("Input X contains infinity") the first
    time fundamentals reach a real fit. ``profit_margin`` already guarded its denominator the same way.
    """
    if fundamentals.empty:
        return fundamentals.assign(
            profit_margin=pd.Series(dtype=float),
            eps_growth_yoy=pd.Series(dtype=float),
            revenue_growth_yoy=pd.Series(dtype=float),
            profit_growth_yoy=pd.Series(dtype=float),
            eps_ttm=pd.Series(dtype=float),
        )

    parts = []
    for _, g in fundamentals.groupby("instrument_id", sort=False):
        g = g.sort_values("event_time").reset_index(drop=True).copy()
        g["profit_margin"] = g["profit"] / g["revenue"].replace(0.0, np.nan)
        g["eps_growth_yoy"] = g["eps_basic"] / g["eps_basic"].shift(4).replace(0.0, np.nan) - 1.0
        g["revenue_growth_yoy"] = g["revenue"] / g["revenue"].shift(4).replace(0.0, np.nan) - 1.0
        # growth vs a loss-making or zero base is not a meaningful ratio: NaN, never +/-inf or a sign flip
        base = g["profit"].shift(4)
        g["profit_growth_yoy"] = (g["profit"] / base.where(base > 0)) - 1.0
        g["eps_ttm"] = g["eps_basic"].rolling(4, min_periods=4).sum()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def finalize_fundamental_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Derive the scale-free valuation feature (trailing-4-quarter EPS / price) and drop every Rs-valued level."""
    out = feats.copy()
    out["earnings_yield"] = out["eps_ttm"] / out["close"]
    return out.drop(columns=[c for c in LEVEL_FUNDAMENTAL_COLUMNS if c in out.columns])


def merge_fundamentals_into_features(
    features: pd.DataFrame, fundamentals_with_ratios: pd.DataFrame
) -> pd.DataFrame:
    """As-of (carry-forward) join: each daily feature row gets the most recent
    fundamentals available at that row's ``available_time``, per instrument.

    The joined row's ``available_time`` is left as the feature row's own time
    (unlike the delivery/macro merges): a fundamentals value that was already
    known before today does not push today's availability later. If no
    fundamentals have ever been published yet for an instrument as of a given
    row, the fundamental columns are NaN -- correctly representing "not yet
    known", not silently backfilled with a future value.
    """
    left = features.sort_values("available_time").reset_index(drop=True)
    right = (
        fundamentals_with_ratios[["instrument_id", "available_time", *FUNDAMENTAL_FEATURE_COLUMNS]]
        .sort_values("available_time")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        left,
        right,
        on="available_time",
        by="instrument_id",
        direction="backward",
    )
    return merged.sort_values(["instrument_id", "event_time"]).reset_index(drop=True)
