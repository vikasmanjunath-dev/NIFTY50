"""Triple-barrier labels, sample-uniqueness weights and meta-labels (Blueprint §3.3).

Barrier half-width is volatility-scaled: ``k * sigma_daily * sqrt(horizon)`` with sigma the trailing log-return std
LAGGED one day (nothing from the entry bar or later sets its own barrier). Touches are detected on daily closes. Each
label records when its outcome became knowable (``outcome_available_time``) so purging stays exact. Operates on ONE
instrument's frame sorted by event_time (like labels/targets.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_triple_barrier_label(g: pd.DataFrame, horizon: int = 5, k: float = 1.0, vol_window: int = 20) -> pd.DataFrame:
    g = g.copy()
    c = g["close"].to_numpy(float)
    n = len(c)
    sigma = np.log(g["close"]).diff().rolling(vol_window, min_periods=vol_window // 2).std().shift(1).to_numpy()
    width = k * sigma * np.sqrt(horizon)
    lab = np.full(n, np.nan); held = np.full(n, np.nan); ret = np.full(n, np.nan); y = np.full(n, np.nan)
    end_pos = np.full(n, -1)
    for i in range(n - horizon):
        if not np.isfinite(width[i]) or width[i] <= 0:
            continue
        path = np.log(c[i + 1:i + horizon + 1] / c[i])
        up, dn = np.flatnonzero(path >= width[i]), np.flatnonzero(path <= -width[i])
        iu, idn = (up[0] if len(up) else horizon), (dn[0] if len(dn) else horizon)
        if iu == horizon and idn == horizon:
            j, lab[i] = horizon - 1, 0.0                                  # vertical barrier
            y[i] = 1.0 if path[j] > 0 else 0.0
        elif iu <= idn:                                                    # a same-bar double touch cannot occur on closes
            j, lab[i], y[i] = iu, 1.0, 1.0
        else:
            j, lab[i], y[i] = idn, -1.0, 0.0
        held[i], ret[i], end_pos[i] = j + 1, path[j], i + 1 + j
    avail = g["available_time"].to_numpy()
    g["tb_label"], g["tb_y"], g["tb_return"], g["tb_holding_days"], g["tb_width"] = lab, y, ret, held, width
    g["tb_outcome_available_time"] = pd.Series([avail[p] if p >= 0 else pd.NaT for p in end_pos], index=g.index)
    g["_tb_start"], g["_tb_end"] = np.arange(n), end_pos
    return g


def uniqueness_weights(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Average uniqueness (Lopez de Prado, AFML ch.4): label i lives on bars [start_i, end_i]; its weight is the mean over
    its life of 1/(number of labels alive on that bar). Overlapping labels share information, so they count less.
    ``end < 0`` (unresolved) gets NaN. O(n) via a difference array and prefix sums."""
    ok = end >= start
    n = int(max(end.max(initial=0), start.max(initial=0))) + 2
    conc = np.zeros(n)
    np.add.at(conc, start[ok], 1)
    np.add.at(conc, end[ok] + 1, -1)
    conc = np.cumsum(conc)
    inv = np.divide(1.0, conc, out=np.zeros(n), where=conc > 0)
    pref = np.concatenate([[0.0], np.cumsum(inv)])
    w = np.full(len(start), np.nan)
    w[ok] = (pref[end[ok] + 1] - pref[start[ok]]) / (end[ok] - start[ok] + 1)
    return w


def meta_labels(primary_side: np.ndarray, tb_y: np.ndarray) -> np.ndarray:
    """1 where the primary model's side (+1 long / -1 short) agreed with the realised barrier outcome, else 0. A secondary
    model trained on this decides whether the primary trade is worth taking (it never chooses direction)."""
    side_up = np.asarray(primary_side) > 0
    return (side_up == (np.asarray(tb_y) > 0.5)).astype(int)
