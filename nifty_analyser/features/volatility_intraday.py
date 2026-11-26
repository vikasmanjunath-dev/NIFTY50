"""Realised-variance summaries from 1-minute bars (5-minute returns, regular session 09:15-15:30 IST)."""
import numpy as np
import pandas as pd


def daily_summary(minutes: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol and day. `timestamp` is UTC (naive or aware)."""
    t = pd.DatetimeIndex(minutes["timestamp"])
    t = (t.tz_localize("UTC") if t.tz is None else t).tz_convert("Asia/Kolkata")
    m = minutes.assign(t=t, date=t.strftime("%Y-%m-%d"), minute=t.hour * 60 + t.minute)
    m = m[(m.minute >= 555) & (m.minute < 930) & (m.close > 0)].sort_values(["symbol", "t"])
    m["bucket"] = (m.minute - 555) // 5
    keys = ["symbol", "date"]
    bars = m.groupby(keys + ["bucket"], as_index=False).agg(open=("open", "first"), close=("close", "last"), volume=("volume", "sum"))
    prev = bars.groupby(keys)["close"].shift(1).fillna(bars["open"])
    bars["r"] = np.log(bars["close"].astype(float) / prev.astype(float))
    bars["r2"] = bars.r ** 2
    bars["down2"] = np.minimum(bars.r, 0) ** 2
    bars["bp"] = bars.r.abs() * bars.groupby(keys)["r"].shift(1).abs() * (np.pi / 2)
    bars["late2"] = bars.r2.where(bars.bucket >= 63, 0.0)  # from 14:30
    bars["late_vol"] = bars.volume.where(bars.bucket >= 63, 0)
    out = bars.groupby(keys, as_index=False).agg(rv=("r2", "sum"), down=("down2", "sum"), bipower=("bp", "sum"), late=("late2", "sum"),
                                                  max_r2=("r2", "max"), volume=("volume", "sum"), late_volume=("late_vol", "sum"),
                                                  n_bars=("r", "size"), open=("open", "first"), close=("close", "last"))
    return out


def intraday_features(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Features known at each day's close, keyed by (instrument_id, date). Scale-free; trailing windows only."""
    eps = 1e-8
    s = summary.sort_values(["symbol", "date"]).reset_index(drop=True)
    parts = []
    for symbol, g in s.groupby("symbol", sort=False):
        rv = g.rv.astype(float)
        base = rv.rolling(252, min_periods=60).median()
        out = pd.DataFrame({"symbol": symbol, "date": g.date})
        for w in (1, 5, 22):
            mean = rv.rolling(w, min_periods=max(1, w // 2)).mean()
            out[f"rv_{w}"] = np.log(mean + eps) - np.log(base + eps)
            out[f"rv_down_{w}"] = g.down.rolling(w, min_periods=max(1, w // 2)).mean() / (mean + eps)
        out["rv_jump_5"] = (1 - g.bipower / (rv + eps)).clip(lower=0).rolling(5, min_periods=3).mean()
        out["rv_late_5"] = (g.late / (rv + eps)).rolling(5, min_periods=3).mean()
        out["rv_spike_5"] = (g.max_r2 / (rv + eps)).rolling(5, min_periods=3).mean()
        share = g.late_volume / g.volume.replace(0, np.nan)
        out["late_volume_rel"] = share.rolling(5, min_periods=3).mean() - share.rolling(66, min_periods=30).mean()
        out["rv_of_rv_22"] = np.log(rv + eps).rolling(22, min_periods=11).std()
        parts.append(out)
    f = pd.concat(parts, ignore_index=True)
    index = f[f.symbol == "NIFTY"][["date", "rv_1", "rv_5", "rv_22"]].rename(columns=lambda c: c if c == "date" else f"index_{c}")
    f = f[f.symbol != "NIFTY"].merge(index, on="date", how="left")
    f["instrument_id"] = "NSE:" + f.pop("symbol")
    return f, [c for c in f.columns if c not in ("instrument_id", "date")]
