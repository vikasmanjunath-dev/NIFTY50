"""Second-tier volatility features, all point-in-time and scale-free."""
import numpy as np
import pandas as pd

from nifty_analyser.features.volatility import EPS, daily_variance


def extra_features(bars: pd.DataFrame, index_bars: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    for iid, g in bars.sort_values(["instrument_id", "event_time"]).groupby("instrument_id", sort=False):
        r = np.log(g.close).diff()
        var = daily_variance(g)
        lv = np.log(var + EPS)
        mean = lambda s, w: s.rolling(w, min_periods=w // 2).mean()
        out = g[["instrument_id", "event_time"]].copy()
        out["har_132"] = np.log(mean(var, 132) + EPS) - np.log(mean(var, 22) + EPS)
        out["vol_of_vol"] = lv.rolling(22).std()
        out["max_move"] = np.log(r.abs().rolling(22).max() + EPS) - 0.5 * np.log(mean(var, 22) + EPS)
        out["range_vs_close"] = np.log(mean(var, 22) + EPS) - np.log(mean(r ** 2, 22) + EPS)
        out["overnight_share"] = mean(np.log(g.open / g.close.shift(1)) ** 2, 22) / (mean(var, 22) + EPS)
        out["pct_rank_252"] = lv.rolling(5).mean().rolling(252, min_periods=120).rank(pct=True)
        out["ewma_rel"] = np.log(var.ewm(alpha=0.06, min_periods=60).mean() + EPS) - np.log(var.rolling(252, min_periods=120).median() + EPS)
        out["ret_1"], out["down_22"] = r, np.log(mean(r.clip(upper=0) ** 2, 22) + EPS) - np.log(mean(r ** 2, 22) + EPS)
        out["_ret"] = r
        parts.append(out)
    f = pd.concat(parts, ignore_index=True)
    by_day = f.groupby("event_time")
    f["dispersion"] = np.log(by_day["_ret"].transform("std") + EPS)
    f["breadth_down"] = by_day["_ret"].transform(lambda s: (s < 0).mean())
    f["rank_today"] = by_day["pct_rank_252"].rank(pct=True)
    f["month"] = f.event_time.dt.month.astype(float)
    f["month_end_in"] = (f.event_time.dt.days_in_month - f.event_time.dt.day).astype(float)
    if index_bars is not None:
        ix = index_bars.sort_values("event_time")
        var = daily_variance(ix)
        m = pd.DataFrame({"event_time": ix.event_time})
        for w in (1, 5, 22):
            m[f"index_har_{w}"] = np.log(var.rolling(w).mean() + EPS)
        m["index_ret_5"] = np.log(ix.close).diff(5)
        f = f.merge(m, on="event_time", how="left")
    f = f.drop(columns="_ret")
    return f, [c for c in f.columns if c not in ("instrument_id", "event_time")]
