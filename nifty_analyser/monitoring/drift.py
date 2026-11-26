"""Feature drift & data freshness monitoring (Blueprint §5.5): Population Stability Index
with the industry thresholds (<0.10 stable, 0.10-0.25 watch, >=0.25 significant)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def psi(reference, current, bins: int = 10) -> float:
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def classify_psi(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    return "stable" if value < 0.10 else "watch" if value < 0.25 else "significant"


def feature_psi_table(reference: pd.DataFrame, current: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = [{"feature": c, "psi": (v := psi(reference[c], current[c])), "status": classify_psi(v)} for c in cols]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def staleness_check(latest_event_time: pd.Timestamp, now: pd.Timestamp, max_age: pd.Timedelta) -> dict:
    age = now - latest_event_time
    return {"age": age, "stale": bool(age > max_age)}


# ---- scale-free drift monitoring -------------------------------------------------------------
# PSI only means something on a stationary, dimensionless quantity. Two real failure modes were found
# on the live service: (1) unit-bearing LEVELS (price/volume-unit indicators, Rs-crore flows, USD-INR,
# Brent, VIX level, revenue/profit) trend by nature, so they "drift" forever; (2) MARKET-WIDE features
# repeat one value per day across ~59 stocks, so ~43 current days look like thousands of rows and PSI's
# small-sample bias (~bins/n) is badly understated. Hence three tiers.
SCALE_DEPENDENT = re.compile(
    r"^(?:(?:sma|ema)_\d+|atr_14|obv|macd|macd_signal|macd_hist|obv_slope_\d+|ad_line_slope_\d+|revenue|profit|eps_basic|eps_ttm"
    r"|(?:usd_inr|brent_crude|india_vix)_level|(?:fii|dii)_net_cr(?:_\d+d_sum)?)$")


def is_scale_dependent(name: str) -> bool:
    return bool(SCALE_DEPENDENT.match(name))


def market_wide_columns(df: pd.DataFrame, cols: list[str], time_col: str = "event_time", frac: float = 0.95) -> list[str]:
    """Columns holding a single value per day across all instruments (>= ``frac`` of days)."""
    if df.empty or not cols:
        return []
    per_day = df.groupby(time_col)[cols].nunique(dropna=True)
    return [c for c in cols if (per_day[c] <= 1).mean() >= frac]


def scale_free_drift_table(features: pd.DataFrame, cols: list[str], trained_through: pd.Timestamp, now: pd.Timestamp,
                           current_days: int = 60, market_days: int = 120, min_current_rows: int = 500,
                           min_market_days: int = 60, sample: int = 20000) -> pd.DataFrame:
    """One row per feature: psi (NaN when not assessable), status, tier, n_current. Tiers:
    ``instrument`` (monitored on rows), ``market_wide`` (monitored on one value per DAY over a longer
    window, needs >= ``min_market_days`` distinct days), ``scale_dependent`` (never monitored)."""
    cols = [c for c in cols if c in features.columns]
    scale_dep = [c for c in cols if is_scale_dependent(c)]
    monitored = [c for c in cols if c not in scale_dep]
    ref = features[features["event_time"] <= trained_through]
    recent = features[features["event_time"] > now - pd.Timedelta(days=market_days)]
    mw = market_wide_columns(recent, monitored)
    inst = [c for c in monitored if c not in mw]
    rows = [{"feature": c, "psi": np.nan, "status": "not monitored (scale-dependent level)", "tier": "scale_dependent", "n_current": 0} for c in scale_dep]

    cur = features[features["event_time"] > now - pd.Timedelta(days=current_days)]
    if len(cur) >= min_current_rows and not ref.empty:
        r = ref.sample(min(sample, len(ref)), random_state=0)
        for c in inst:
            v = psi(r[c], cur[c])
            rows.append({"feature": c, "psi": v, "status": classify_psi(v), "tier": "instrument", "n_current": int(cur[c].notna().sum())})
    else:
        rows += [{"feature": c, "psi": np.nan, "status": "too few current rows", "tier": "instrument", "n_current": len(cur)} for c in inst]

    if mw:
        daily_ref = ref.groupby("event_time")[mw].first()
        daily_cur = recent.groupby("event_time")[mw].first()
        for c in mw:
            n = int(daily_cur[c].notna().sum())
            ok = n >= min_market_days and daily_ref[c].notna().sum() >= min_market_days
            # PSI's small-sample bias is ~(bins-1)/n: with ~85 distinct days, 10 bins reads ~0.14 on a
            # perfectly stationary series, so bins scale with the number of observations (~1 per 20).
            bins = int(max(3, min(10, n // 20)))
            v = psi(daily_ref[c], daily_cur[c], bins=bins) if ok else np.nan
            rows.append({"feature": c, "psi": v, "status": classify_psi(v) if ok else f"too few distinct days ({n} < {min_market_days})",
                         "tier": "market_wide", "n_current": n})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def calibrate_drift_thresholds(features: pd.DataFrame, cols: list[str], step_days: int = 30, min_ref_days: int = 540,
                               ref_lag_days: int = 7, **drift_kw) -> dict:
    """Empirical alert levels from the model's OWN history. Textbook PSI cut-offs assume independent samples; these
    features are autocorrelated and move together across stocks, so with the standard cut-offs the live monitor
    would have been AMBER in 100% of historical months (measured). Here, at each monthly date the current window is
    compared with an expanding reference (everything up to ``ref_lag_days`` earlier -- the same relationship as live),
    giving the historical distribution of "share of scale-free features with PSI >= 0.25".
    AMBER = 90th percentile of that share; RED = above anything ever observed (never below AMBER + 0.10)."""
    t0 = features["event_time"].min() + pd.Timedelta(days=min_ref_days)
    dates = pd.date_range(t0.tz_localize(None), features["event_time"].max().tz_localize(None), freq=f"{step_days}D", tz=t0.tz)
    series = []
    for d in dates:
        sub = features[features["event_time"] <= d]
        t = scale_free_drift_table(sub, cols, d - pd.Timedelta(days=ref_lag_days), d, **drift_kw)
        a = t[t["psi"].notna()]
        if len(a):
            series.append(float((a["psi"] >= 0.25).mean()))
    if len(series) < 8:
        raise ValueError(f"only {len(series)} historical points; need >= 8 to calibrate")
    arr = np.array(series)
    amber = float(np.quantile(arr, 0.90))
    red = float(min(1.0, max(arr.max(), amber + 0.10)))
    return {"psi_frac_amber": amber, "psi_frac_red": red, "n_points": len(arr), "share_median": float(np.median(arr)),
            "share_min": float(arr.min()), "share_max": float(arr.max()), "series": [round(x, 4) for x in arr]}
