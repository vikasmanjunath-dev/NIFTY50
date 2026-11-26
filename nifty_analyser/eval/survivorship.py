"""Survivorship-bias measurement (Blueprint §3.4).

Two universes over the same dates:
  * BIASED -- today's 50 constituents projected backward over the whole history
              (what a naive backtest does);
  * PIT    -- the real point-in-time membership on each date (`index_membership`).
Rows in BIASED-but-not-PIT are *phantom* (a stock before it joined the index, kept
only because it later got promoted); rows in PIT-but-not-BIASED are *omitted*
(stocks that were real members but later dropped out and so vanish from a naive
backtest). The bias is measured directly as the realised forward return of each group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_analyser.data.index_membership import ANCHOR_MEMBERS, membership_mask
from nifty_analyser.labels.targets import add_direction_label, add_forward_return

TRADING_DAYS = 252


def label_dataset(features: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Per-instrument labels: h-day direction (+1/-1), 1-day forward return; flat rows dropped."""
    parts = []
    for _, g in features.groupby("instrument_id", sort=False):
        g = add_forward_return(add_direction_label(g.sort_values("event_time"), horizon), 1)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    out_col = f"fwd_ret_{horizon}d_outcome_available_time"
    df = df.dropna(subset=[f"dir_label_{horizon}d", out_col, "fwd_ret_1d"])
    df = df[df[f"dir_label_{horizon}d"] != 0].copy()
    df["y"] = (df[f"dir_label_{horizon}d"] == 1).astype(int)
    return df.sort_values(["event_time", "instrument_id"]).reset_index(drop=True)


def universe_frames(labelled: pd.DataFrame, start: str = "2021-09-20") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(biased, pit) frames over the same date range, each with a fresh RangeIndex."""
    df = labelled[labelled["event_time"] >= pd.Timestamp(start, tz="Asia/Kolkata")]
    current_ids = {f"NSE:{s}" for s in ANCHOR_MEMBERS}
    biased = df[df["instrument_id"].isin(current_ids)].reset_index(drop=True)
    pit = df[membership_mask(df)].reset_index(drop=True)
    return biased, pit


def _ann(mean_daily: float) -> float:
    return float(mean_daily * TRADING_DAYS)


def bias_diagnostics(biased: pd.DataFrame, pit: pd.DataFrame, ret_col: str = "fwd_ret_1d") -> dict:
    """Realised return of phantom / omitted rows and of the equal-weight universes."""
    key = ["instrument_id", "event_time"]
    b_idx = pd.MultiIndex.from_frame(biased[key])
    p_idx = pd.MultiIndex.from_frame(pit[key])
    phantom = biased[~b_idx.isin(p_idx)]
    omitted = pit[~p_idx.isin(b_idx)]

    def daily_ew(df):
        return df.groupby("event_time")[ret_col].mean()

    ew_b, ew_p = daily_ew(biased), daily_ew(pit)
    both = pd.concat([ew_b, ew_p], axis=1, keys=["b", "p"]).dropna()
    diff = both["b"] - both["p"]

    def group(df):
        return {"n_rows": int(len(df)), "n_instruments": int(df["instrument_id"].nunique()),
                "mean_daily_return": float(df[ret_col].mean()) if len(df) else float("nan"),
                "annualised_return": _ann(df[ret_col].mean()) if len(df) else float("nan")}

    def sharpe(s):
        return float(s.mean() / s.std() * np.sqrt(TRADING_DAYS)) if s.std() > 0 else float("nan")

    return {
        "phantom_rows (in biased, not real members)": group(phantom),
        "omitted_rows (real members missing from biased)": group(omitted),
        "equal_weight_biased": {"annualised_return": _ann(ew_b.mean()), "sharpe": sharpe(ew_b)},
        "equal_weight_pit": {"annualised_return": _ann(ew_p.mean()), "sharpe": sharpe(ew_p)},
        "survivorship_premium_annualised": _ann(diff.mean()),
        "survivorship_premium_tstat": float(diff.mean() / (diff.std() / np.sqrt(len(diff)))) if diff.std() > 0 else float("nan"),
        "row_counts": {"biased": int(len(biased)), "pit": int(len(pit))},
    }


def daily_cross_sectional_ic(df: pd.DataFrame, score_col: str, ret_col: str, min_names: int = 10) -> float:
    """Mean per-day Spearman rank correlation of a score vs. realised return (cross-sectional IC)."""
    ics = []
    for _, g in df.groupby("event_time"):
        if len(g) >= min_names and g[score_col].nunique() > 1 and g[ret_col].nunique() > 1:
            ics.append(g[score_col].rank().corr(g[ret_col].rank()))
    return float(np.nanmean(ics)) if ics else float("nan")
