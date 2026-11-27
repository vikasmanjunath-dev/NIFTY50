"""Scheduled board meetings. A meeting counts only from the moment NSE published the intimation, so at any date we see
exactly the upcoming dates that had been announced."""
import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
CAP = 100.0
COLUMNS = ["ev_days_to_meeting", "ev_days_to_results", "ev_results_in_horizon", "ev_days_since_results", "ev_market_results_share"]


def normalise(records: list[dict], instrument_id: str) -> pd.DataFrame:
    rows = []
    for r in records:
        day = pd.to_datetime(r.get("bm_date"), format="%d-%b-%Y", errors="coerce")
        known = pd.to_datetime(r.get("bm_timestamp"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
        if pd.isna(day) or pd.isna(known):
            continue
        text = f"{r.get('bm_purpose')} {r.get('bm_desc')}".lower()
        rows.append((instrument_id, day.tz_localize(IST), known.tz_localize(IST), "result" in text))
    return pd.DataFrame(rows, columns=["instrument_id", "meeting_date", "available_time", "is_results"])


def add_event_features(df: pd.DataFrame, meetings: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    out = df.reset_index(drop=True).copy()
    values = np.full((len(out), 4), np.nan)
    day_ns = 86_400 * 10**9
    ns = lambda s: pd.DatetimeIndex(s).as_unit("ns").asi8
    now, today = ns(out.available_time), ns(pd.DatetimeIndex(out.available_time).normalize())
    by_inst = {i: g for i, g in meetings.groupby("instrument_id")}
    for iid, idx in out.groupby("instrument_id").indices.items():
        g = by_inst.get(iid)
        if g is None:
            continue
        known, meet, res = ns(g.available_time), ns(g.meeting_date), g.is_results.to_numpy(bool)
        seen = known[None, :] <= now[idx, None]
        gap = (meet[None, :] - today[idx, None]) / day_ns
        upcoming = seen & (gap >= 0) & (gap <= CAP)
        next_any = np.where(upcoming, gap, CAP).min(1)
        next_res = np.where(upcoming & res, gap, CAP).min(1)
        since = np.where(seen & res & (gap < 0), -gap, CAP * 2).min(1)
        values[idx] = np.c_[next_any, next_res, (next_res <= horizon * 1.45).astype(float), since]
        before_first = ~seen.any(1)
        values[idx[before_first]] = np.nan
    for j, c in enumerate(COLUMNS[:4]):
        out[c] = values[:, j]
    out["ev_market_results_share"] = out.groupby("available_time")["ev_results_in_horizon"].transform("mean")
    return out, list(COLUMNS)
