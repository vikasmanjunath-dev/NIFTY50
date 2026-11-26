"""Overnight US market state and Nifty index volatility. Each series is joined on the time it became known, so an Indian
close sees the previous US session, never the same calendar day's."""
import numpy as np
import pandas as pd

from nifty_analyser.features.volatility import EPS, daily_variance

IST = "Asia/Kolkata"
# name -> (Yahoo ticker, exchange time zone, local close time)
GLOBAL_SERIES = {"spx": ("^GSPC", "America/New_York", "16:00"), "us_vix": ("^VIX", "America/New_York", "16:15"),
                 "nifty": ("^NSEI", IST, "15:30")}


def stamp(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    _, zone, close = GLOBAL_SERIES[name]
    days = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    known = (days + pd.Timedelta(f"{close}:00")).tz_localize(zone).tz_convert(IST)
    out = raw[["open", "high", "low", "close"]].reset_index(drop=True)
    out.insert(0, "available_time", known.astype("datetime64[ns, Asia/Kolkata]"))
    return out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)].sort_values("available_time").reset_index(drop=True)


def _series_features(s: pd.DataFrame, name: str) -> pd.DataFrame:
    log_close = np.log(s.close)
    out = pd.DataFrame({"available_time": s.available_time})
    if name == "us_vix":
        out["us_vix_rel"] = log_close - log_close.rolling(252, min_periods=120).median()
        out["us_vix_chg_1"], out["us_vix_chg_5"] = log_close.diff(), log_close.diff(5)
        return out
    var = daily_variance(s)
    base = var.rolling(252, min_periods=120).median()
    for w in (1, 5, 22):
        out[f"{name}_har_{w}"] = np.log(var.rolling(w).mean() + EPS) - np.log(base + EPS)
    out[f"{name}_ret_1"], out[f"{name}_ret_5"] = log_close.diff(), log_close.diff(5)
    return out


def add_global_features(df: pd.DataFrame, series: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Returns the frame (order preserved) and the new columns grouped as `us` and `index`."""
    out = df.reset_index(drop=True).copy()
    left = out[["available_time"]].astype({"available_time": "datetime64[ns, Asia/Kolkata]"}).assign(_row=np.arange(len(out)))
    left = left.sort_values("available_time")
    groups = {"us": [], "index": []}
    for name, s in series.items():
        right = _series_features(s.sort_values("available_time").reset_index(drop=True), name)
        merged = pd.merge_asof(left, right, on="available_time", direction="backward", tolerance=pd.Timedelta(days=7)).sort_values("_row")
        cols = [c for c in right.columns if c != "available_time"]
        for c in cols:
            out[c] = merged[c].to_numpy()
        groups["index" if name == "nifty" else "us"] += cols
    return out, groups
