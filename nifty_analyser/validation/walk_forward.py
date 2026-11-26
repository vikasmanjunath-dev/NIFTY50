"""Purged & embargoed expanding-window walk-forward CV (Blueprint §3.4).

Validation blocks are consecutive chunks of unique dates; training for a fold is
everything strictly before the block, minus (purge) rows whose label outcome
resolves on/after the block start and (embargo) rows within ``embargo`` of it.
Indices are positional labels of a RangeIndex frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def purged_embargoed_walk_forward(
    df: pd.DataFrame, n_splits: int, outcome_time_col: str, embargo: pd.Timedelta,
    event_time_col: str = "event_time",
) -> list[Fold]:
    ev = df[event_time_col]
    out_t = df[outcome_time_col]
    dates = pd.Index(sorted(ev.unique()))
    chunks = np.array_split(np.arange(len(dates)), n_splits + 1)
    folds = []
    for k in range(1, n_splits + 1):
        block = dates[chunks[k]]
        val_start, val_end = block[0], block[-1]
        val_mask = (ev >= val_start) & (ev <= val_end)
        train_mask = (ev < val_start - embargo) & (out_t < val_start)
        folds.append(Fold(k - 1, np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(val_mask.to_numpy()), val_start, val_end))
    return folds


def assert_folds_are_leakage_free(df: pd.DataFrame, folds: list[Fold], outcome_time_col: str) -> None:
    for f in folds:
        if np.intersect1d(f.train_idx, f.val_idx).size:
            raise AssertionError(f"fold {f.fold_id}: train/val overlap")
        latest_outcome = df[outcome_time_col].iloc[f.train_idx].max()
        if latest_outcome >= f.val_start:
            raise AssertionError(f"fold {f.fold_id}: train label resolves at {latest_outcome} >= val_start {f.val_start}")


def carve_calibration_slice(
    df: pd.DataFrame, idx: np.ndarray, calib_frac: float, outcome_time_col: str,
    embargo: pd.Timedelta, event_time_col: str = "event_time",
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``idx`` chronologically into (fit, calibration). The calibration slice is the
    latest ``calib_frac`` of dates; fit rows are purged (label resolves before the slice starts)
    and embargoed, so a calibrator never sees data the model it calibrates was trained on."""
    ev, out_t = df[event_time_col].iloc[idx], df[outcome_time_col].iloc[idx]
    dates = pd.Index(sorted(ev.unique()))
    start = dates[int(len(dates) * (1 - calib_frac))]
    fit = idx[((ev < start - embargo) & (out_t < start)).to_numpy()]
    calib = idx[(ev >= start).to_numpy()]
    return fit, calib
