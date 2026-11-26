"""Combinatorial purged cross-validation (Blueprint §3.4; Lopez de Prado, AFML ch.12).

Dates are cut into N groups; every combination of k groups is a test set (C(N,k) splits). Each split trains on the
rest after PURGING training rows whose label interval [event, outcome] overlaps a test group and EMBARGOING rows right
after it. Each group appears in C(N-1,k-1) splits, so predictions assemble into phi = C(N-1,k-1) complete
out-of-sample *paths* -- a distribution of performance, not 4 correlated numbers. (Training on data later than the test
group is intended here: this measures model quality under many regimes, not a deployable chronology.)"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from nifty_analyser.eval.metrics import evaluate_binary_predictions


@dataclass(frozen=True)
class CPCVSplit:
    split_id: int
    test_groups: tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray


def cpcv_splits(df: pd.DataFrame, n_groups: int, k: int, outcome_time_col: str, embargo: pd.Timedelta,
                event_time_col: str = "event_time") -> tuple[list[CPCVSplit], list[dict[int, int]], np.ndarray]:
    """(splits, paths, group_id_per_row); ``paths[p][g]`` = split whose test set supplies group g in path p."""
    dates = pd.Index(sorted(df[event_time_col].unique()))
    chunks = np.array_split(np.arange(len(dates)), n_groups)
    bounds = [(dates[c[0]], dates[c[-1]]) for c in chunks]
    ev, out = df[event_time_col], df[outcome_time_col]
    gid = np.full(len(df), -1)
    for g, (s, e) in enumerate(bounds):
        gid[((ev >= s) & (ev <= e)).to_numpy()] = g
    splits = []
    for sid, groups in enumerate(combinations(range(n_groups), k)):
        test_mask = np.isin(gid, groups)
        banned = np.zeros(len(df), dtype=bool)
        for g in groups:
            s, e = bounds[g]
            banned |= ((ev <= e) & (out >= s)).to_numpy()                       # label interval overlaps the test group (purge)
            banned |= ((ev > e) & (ev <= e + embargo)).to_numpy()               # embargo after the test group
        splits.append(CPCVSplit(sid, groups, np.flatnonzero(~test_mask & ~banned), np.flatnonzero(test_mask)))
    by_group = {g: [s.split_id for s in splits if g in s.test_groups] for g in range(n_groups)}
    paths = [{g: by_group[g][p] for g in range(n_groups)} for p in range(comb(n_groups - 1, k - 1))]
    return splits, paths, gid


def cpcv_paths(df: pd.DataFrame, predict, n_groups: int, k: int, outcome_time_col: str, embargo: pd.Timedelta) -> list[np.ndarray]:
    """``predict(train_idx, test_idx) -> proba`` is called once per split; returns one full out-of-sample probability
    vector per assembled path (works for any model, including ones that need the whole frame, e.g. the ensemble)."""
    splits, paths, gid = cpcv_splits(df, n_groups, k, outcome_time_col, embargo)
    preds = {}
    for s in splits:
        p = np.full(len(df), np.nan)
        p[s.test_idx] = predict(s.train_idx, s.test_idx)
        preds[s.split_id] = p
    out = []
    for path in paths:
        proba = np.full(len(df), np.nan)
        for g, sid in path.items():
            m = gid == g
            proba[m] = preds[sid][m]
        assert not np.isnan(proba).any(), "a path must cover every row exactly once"
        out.append(proba)
    return out


def path_metrics(df: pd.DataFrame, proba: np.ndarray, return_col: str, path_id: int = 0) -> dict:
    """Pooled metrics plus DATE-NEUTRAL ones: pooled AUC is confounded by shifts in the up-day base rate across dates, so
    also report the mean daily cross-sectional Spearman IC and the mean within-date AUC (the quantities a stock-ranking
    signal actually needs)."""
    from sklearn.metrics import roc_auc_score
    base = evaluate_binary_predictions(df["y"].to_numpy(), proba, df[return_col].to_numpy(), path_id, "cpcv").to_dict()
    d = pd.DataFrame({"t": df["event_time"].to_numpy(), "p": proba, "r": df[return_col].to_numpy(), "y": df["y"].to_numpy()})
    ics, aucs = [], []
    for _, g in d.groupby("t"):
        if len(g) >= 10:
            ics.append(g["p"].rank().corr(g["r"].rank()))
            if g["y"].nunique() > 1:
                aucs.append(roc_auc_score(g["y"], g["p"]))
    return {**base, "daily_ic": float(np.nanmean(ics)), "daily_ic_t": float(np.nanmean(ics) / (np.nanstd(ics, ddof=1) / np.sqrt(len(ics)))),
            "within_date_auc": float(np.mean(aucs))}


def cpcv_evaluate(df: pd.DataFrame, cols: list[str], fit, n_groups: int, k: int, outcome_time_col: str, embargo: pd.Timedelta,
                  return_col: str, seed: int = 42) -> pd.DataFrame:
    """One metrics row per assembled path for a (X, y, random_state) -> model fitter."""
    X, y = df[cols].to_numpy(dtype=float), df["y"].to_numpy()
    predict = lambda tr, te: fit(X[tr], y[tr], random_state=seed).predict_proba_up(X[te])
    return pd.DataFrame([{"path": i, **path_metrics(df, p, return_col, i)} for i, p in enumerate(cpcv_paths(df, predict, n_groups, k, outcome_time_col, embargo))])
