"""Labels (Blueprint §3.3). Each carries ``outcome_available_time`` -- when the label's
outcome is knowable -- which is what makes purging overlapping labels exact, not guesswork.
Operate on ONE instrument's frame sorted by event_time."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_return(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = g.copy()
    g[f"fwd_ret_{horizon}d"] = np.log(g["close"].shift(-horizon) / g["close"])
    g[f"fwd_ret_{horizon}d_outcome_available_time"] = g["available_time"].shift(-horizon)
    return g


def add_direction_label(g: pd.DataFrame, horizon: int, flat_threshold: float = 0.0) -> pd.DataFrame:
    """+1 / -1 / 0 (flat band) on the forward ``horizon``-day log return."""
    g = add_forward_return(g, horizon)
    r = g[f"fwd_ret_{horizon}d"]
    label = np.where(r > flat_threshold, 1.0, np.where(r < -flat_threshold, -1.0, 0.0))
    g[f"dir_label_{horizon}d"] = np.where(r.isna(), np.nan, label)
    return g
