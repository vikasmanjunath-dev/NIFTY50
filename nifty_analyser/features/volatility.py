"""Point-in-time features and targets for realised-volatility forecasting (HAR-RV family).

Every feature at row t uses only information available by t's close. Daily variance is range-based
(Garman-Klass plus the overnight gap), which is far less noisy than a squared close-to-close return.
All features are scale-free."""
import numpy as np
import pandas as pd

EPS = 1e-8
HAR_WINDOWS = (1, 5, 22, 66)
RESULT_CATEGORIES = r"financial result|outcome of board meeting"


def daily_variance(g: pd.DataFrame) -> pd.Series:
    hl = np.log(g["high"] / g["low"]) ** 2
    co = np.log(g["close"] / g["open"]) ** 2
    overnight = np.log(g["open"] / g["close"].shift(1)) ** 2
    return (0.5 * hl - (2 * np.log(2) - 1) * co).clip(lower=0) + overnight


def _instrument(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = g.sort_values("event_time").reset_index(drop=True)
    ret = np.log(g["close"] / g["close"].shift(1))
    var = daily_variance(g)
    out = g[["instrument_id", "event_time", "available_time"]].copy()
    for w in HAR_WINDOWS:
        out[f"har_{w}"] = np.log(var.rolling(w, min_periods=w).mean() + EPS)
    out["har_down_5"] = np.log((ret.clip(upper=0) ** 2).rolling(5).mean() + EPS) - np.log((ret ** 2).rolling(5).mean() + EPS)
    out["ret_5"] = ret.rolling(5).sum()
    out["ret_22"] = ret.rolling(22).sum()
    out["volume_surprise"] = np.log((g["volume"].rolling(5).mean() + 1) / (g["volume"].rolling(66).mean() + 1))
    # target pieces: same definition as the original study (close-to-close, versus own trailing median)
    cur = np.sqrt((ret ** 2).rolling(horizon).sum())
    fwd = cur.shift(-horizon)
    med = cur.rolling(252, min_periods=120).median()  # every value in it is fully realised by t
    out["cur_rv"], out["fwd_rv"], out["med_rv"] = cur, fwd, med
    out["rel_cur"] = np.log((cur + EPS) / (med + EPS))
    for w in (5, 22, 66):
        out[f"rel_har_{w}"] = out[f"har_{w}"] - out[f"har_{w}"].rolling(252, min_periods=120).median()
    out["outcome_time"] = out["available_time"].shift(-horizon)
    return out


def _days_since_results(frame: pd.DataFrame, announcements: pd.DataFrame) -> pd.Series:
    hits = announcements[announcements["category"].str.contains(RESULT_CATEGORIES, case=False, na=False)]
    out = pd.Series(np.nan, index=frame.index)
    for iid, rows in frame.groupby("instrument_id"):
        times = np.sort(pd.DatetimeIndex(hits.loc[hits["instrument_id"] == iid, "available_time"]).asi8)
        if not len(times):
            continue
        now = pd.DatetimeIndex(rows["available_time"]).asi8
        pos = np.searchsorted(times, now, side="right") - 1
        out.loc[rows.index] = np.where(pos >= 0, (now - times[np.maximum(pos, 0)]) / 86_400e9, np.nan)
    return out


def build_volatility_frame(bars: pd.DataFrame, horizon: int, vix: pd.DataFrame | None = None,
                           announcements: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = pd.concat([_instrument(g, horizon) for _, g in bars.groupby("instrument_id", sort=False)], ignore_index=True)
    for w in (1, 5, 22):
        df[f"mkt_har_{w}"] = df.groupby("event_time")[f"har_{w}"].transform("mean")
    df["mkt_rel_22"] = df.groupby("event_time")["rel_har_22"].transform("mean")
    if vix is not None:
        v = vix.sort_values("available_time")[["available_time", "close"]].copy()
        lv = np.log(v["close"])
        v["vix_log"] = lv
        v["vix_rel"] = lv - lv.rolling(252, min_periods=120).median()
        v["vix_chg_5"] = lv.diff(5)
        v = v.drop(columns="close").astype({"available_time": df["available_time"].dtype})
        df = pd.merge_asof(df.sort_values("available_time"), v, on="available_time", direction="backward")
    if announcements is not None:
        df = df.reset_index(drop=True)
        since = _days_since_results(df, announcements)
        df["days_since_results"] = since.clip(upper=200)
        # results arrive roughly every 91 days, so the next one is "expected" late in the cycle
        df["results_due"] = ((since + horizon * 1.45 >= 85) & (since <= 110)).astype(float).where(since.notna())
    meta = {"instrument_id", "event_time", "available_time", "outcome_time", "cur_rv", "fwd_rv", "med_rv"}
    cols = [c for c in df.columns if c not in meta]
    return df.sort_values(["event_time", "instrument_id"]).reset_index(drop=True), cols
