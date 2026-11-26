"""Market-neutral cross-sectional long/short test (the quantity a daily cross-sectional IC suggests): on each rebalance
date go long the top-q and short the bottom-q names by score (each leg 50% of capital), hold ``hold`` days, charge the
impact-cost model on every weight change. Plus a within-date permutation null: shuffling scores across names each day
keeps everything but the ranking information, giving the Sharpe distribution of 'no skill' under identical costs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_analyser.eval.costs import ImpactCostModel

TRADING_DAYS = 252


def pivots(df: pd.DataFrame, score_col: str, ret_col: str, liq: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = df[["instrument_id", "event_time", score_col, ret_col]].merge(liq[["instrument_id", "event_time", "adv_value", "sigma"]],
                                                                    on=["instrument_id", "event_time"], how="left")
    pv = {k: d.pivot_table(index="event_time", columns="instrument_id", values=v, dropna=False) for k, v in
          (("score", score_col), ("ret", ret_col), ("adv", "adv_value"), ("sigma", "sigma"))}
    idx, cols = pv["score"].index.sort_values(), pv["score"].columns
    return {k: v.reindex(index=idx, columns=cols) for k, v in pv.items()}


def weights(score: np.ndarray, q: float, hold: int) -> np.ndarray:
    """Target weights (dates x names): rebalance every ``hold`` dates, +-0.5/n_leg on top/bottom-q, NaN scores excluded."""
    T, N = score.shape
    w = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(T):
        if t % hold == 0:
            s = score[t]
            ok = ~np.isnan(s)
            n = int(ok.sum())
            cur = np.zeros(N)
            k = max(int(round(q * n)), 1)
            if n >= 2 * k and n >= 4:
                order = np.argsort(np.where(ok, s, np.nan))                     # NaN sorted last
                cur[order[:k]] = -0.5 / k
                cur[order[n - k:n]] = 0.5 / k
        w[t] = cur
    return w


def longshort_returns(pv: dict[str, pd.DataFrame], q: float = 0.2, hold: int = 5, model: ImpactCostModel = ImpactCostModel(),
                      score: np.ndarray | None = None) -> pd.DataFrame:
    s = pv["score"].to_numpy() if score is None else score
    ret = np.nan_to_num(pv["ret"].to_numpy())
    w = weights(s, q, hold)
    dw = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, w.shape[1]))))
    adv, sig = pv["adv"].to_numpy(), pv["sigma"].to_numpy()
    sig = np.where(np.isnan(sig), np.nanmedian(sig) if np.isfinite(sig).any() else 0.02, sig)
    part = np.where(adv > 0, dw * model.capital / np.where(adv > 0, adv, 1.0), 1.0).clip(max=1.0)       # unknown ADV -> 100%
    cost_bps = model.fixed_bps + model.half_spread_bps + model.impact_coef * sig * np.sqrt(part) * 1e4
    gross = (w * ret).sum(axis=1)
    cost = (dw * cost_bps / 1e4).sum(axis=1)
    return pd.DataFrame({"gross": gross, "net": gross - cost, "turnover": dw.sum(axis=1)}, index=pv["score"].index)


def sharpe(r: pd.Series | np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else float("nan")


def permutation_null(pv: dict[str, pd.DataFrame], n: int = 200, seed: int = 0, **kw) -> pd.DataFrame:
    """Sharpe of long/short portfolios whose within-date score ranking is randomly permuted."""
    rng = np.random.default_rng(seed)
    base = pv["score"].to_numpy()
    rows = []
    for _ in range(n):
        s = base.copy()
        for t in range(s.shape[0]):
            ok = np.flatnonzero(~np.isnan(s[t]))
            s[t, ok] = s[t, rng.permutation(ok)]
        r = longshort_returns(pv, score=s, **kw)
        rows.append({"gross": sharpe(r["gross"]), "net": sharpe(r["net"])})
    return pd.DataFrame(rows)
