"""Cross-sectional features (Blueprint §2.7).

Two stages, deliberately separated so universe choice (biased vs point-in-time) is explicit:
  1. ``add_index_relative_features`` -- per instrument, vs the real Nifty 50 index series
     (relative strength, rolling beta/correlation). Independent of who else is in the universe.
  2. ``add_same_day_cross_sectional`` -- same-day percentile ranks and sector-relative
     returns. MUST be applied AFTER the universe filter: ranking against stocks that were not
     index members on that date would leak the very survivorship bias being removed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_analyser.data.instrument_master import NIFTY50

RANK_COLS = ("ret_5d", "ret_20d", "rsi_14", "volume_ratio_20d")


def add_index_relative_features(features: pd.DataFrame, index_bars: pd.DataFrame) -> pd.DataFrame:
    idx = index_bars.sort_values("event_time").set_index("event_time")["close"]
    idx_logret = np.log(idx).diff()
    parts = []
    for _, g in features.groupby("instrument_id", sort=False):
        g = g.sort_values("event_time").copy()
        ri = idx_logret.reindex(g["event_time"]).to_numpy()
        r = np.log(g["close"]).diff().to_numpy()
        for h in (5, 20, 60):
            g[f"rel_strength_{h}d"] = g[f"ret_{h}d"] - np.log(idx).diff(h).reindex(g["event_time"]).to_numpy()
        rs, ris = pd.Series(r, index=g.index), pd.Series(ri, index=g.index)
        for w in (120, 250):
            cov = rs.rolling(w, min_periods=w).cov(ris)
            g[f"beta_idx_{w}d"] = cov / ris.rolling(w, min_periods=w).var()
            g[f"corr_idx_{w}d"] = rs.rolling(w, min_periods=w).corr(ris)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["instrument_id", "event_time"]).reset_index(drop=True)


def add_same_day_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in RANK_COLS:
        if c in out:
            out[f"xs_rank_{c}"] = out.groupby("event_time")[c].rank(pct=True)
    sector = out["instrument_id"].map(lambda i: NIFTY50[i].sector if i in NIFTY50 else "Unknown")
    for h in (5, 20):
        peers = out.groupby([out["event_time"], sector])[f"ret_{h}d"].transform("mean")
        out[f"sector_rel_ret_{h}d"] = out[f"ret_{h}d"] - peers
    return out
