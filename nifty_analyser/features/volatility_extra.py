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


def stock_features(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    """What is persistent about a stock: how often its volatility has exceeded usual (realised outcomes only) and where it sits
    versus its sector and the market today. `df` is a volatility frame (features/volatility.py) for the whole pool."""
    from nifty_analyser.data.instrument_master import get_instrument

    df = df.sort_values(["instrument_id", "event_time"]).copy()
    above = (df.fwd_rv > df.med_rv).astype(float).where(df.fwd_rv.notna() & df.med_rv.notna())
    past = above.groupby(df.instrument_id).shift(horizon)  # the newest outcome used is fully realised today
    df["own_exceed_rate"] = past.groupby(df.instrument_id).transform(lambda s: s.rolling(504, min_periods=120).mean())
    sector = df.instrument_id.map({i: get_instrument(i).sector for i in df.instrument_id.unique()})
    for w in (5, 66):
        col = df[f"har_{w}"]
        df[f"vs_sector_{w}"] = col - col.groupby([df.event_time, sector]).transform("mean")
        df[f"vs_market_{w}"] = col - col.groupby(df.event_time).transform("mean")
    return df.sort_values(["event_time", "instrument_id"]).reset_index(drop=True), [
        "own_exceed_rate", "vs_sector_5", "vs_market_5", "vs_sector_66", "vs_market_66"]
