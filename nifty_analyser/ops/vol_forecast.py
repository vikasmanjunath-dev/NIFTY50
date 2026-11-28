"""Live volatility forecasts and their forward (untainted) scorecard.

`latest_forecasts` trains only on outcomes already realised at `as_of` and forecasts the newest row of each current
index member. `record` appends them to a hash-chained ledger once per data date; `score` grades matured entries."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.index_membership import instrument_ids_as_of, membership_mask
from nifty_analyser.data.store import load_all_macro_series, load_universe_bars
from nifty_analyser.features.volatility import build_volatility_frame
from nifty_analyser.features.volatility_events import add_event_features
from nifty_analyser.features.volatility_extra import extra_features
from nifty_analyser.models.volatility import fit_predict_classifiers, fit_predict_regressors

LEDGER_KIND = "vol_forecast"


def load_inputs():
    return (load_universe_bars(), load_all_macro_series()["india_vix"],
            pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"))


HISTORY_DIR = DATA_RAW_DIR / "history"


# Horizons whose classifier also uses the second-tier features: only those where the held-out gain was clear
# (reports/vol_auc_search2.json). The volatility-level regression, and so the price ranges, do not use them.
EXTRA_FEATURE_HORIZONS = frozenset({20})


# Horizons whose classifier uses announced board-meeting dates (reports/vol_events.json).
EVENT_FEATURE_HORIZONS = frozenset({5})
MEETINGS_PATH = DATA_RAW_DIR / "board_meetings.parquet"


def load_meetings(path=None) -> pd.DataFrame | None:
    path = path or MEETINGS_PATH
    return pd.read_parquet(path) if path.exists() else None


def _with_extra(df, bars):
    extra, cols = extra_features(bars)
    return df.merge(extra, on=["instrument_id", "event_time"], how="left"), cols


def load_history_training(horizon: int, directory=None, extra: bool = True,
                          meetings: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Pre-2021 rows (scripts/backfill_history.py) used only for training; None if not downloaded."""
    directory = directory or HISTORY_DIR
    if not (directory / "bars.parquet").exists():
        return None
    bars, vix = pd.read_parquet(directory / "bars.parquet"), pd.read_parquet(directory / "india_vix.parquet")
    for frame in (bars, vix):
        for c in ("event_time", "available_time"):
            frame[c] = pd.to_datetime(frame[c], utc=True).dt.tz_convert("Asia/Kolkata").astype("datetime64[ns, Asia/Kolkata]")
    bars = bars[bars.quality_flag != "FAIL"]
    df, _ = build_volatility_frame(bars, horizon, vix[vix.quality_flag != "FAIL"])
    if extra and horizon in EXTRA_FEATURE_HORIZONS:
        df, _ = _with_extra(df, bars)
    if meetings is not None and horizon in EVENT_FEATURE_HORIZONS:
        df, _ = add_event_features(df, meetings, horizon)
    df = df.dropna(subset=["fwd_rv", "med_rv", "cur_rv", "outcome_time", "har_66"])
    return df[(df.fwd_rv > 0) & (df.cur_rv > 0)].reset_index(drop=True)


def latest_complete_date(df: pd.DataFrame, coverage: float = 0.95) -> pd.Timestamp:
    """Newest date on which nearly all of that day's index members have a bar. A data vendor that is late for some
    stocks must not shrink the forecast (and the permanent forward record) to whoever arrived first."""
    for day in sorted(df.event_time.unique(), reverse=True)[:20]:
        day = pd.Timestamp(day)
        members = instrument_ids_as_of(day)
        have = df.loc[df.event_time == day, "instrument_id"]
        if not members or have.isin(members).sum() >= coverage * len(members):
            return day
    raise RuntimeError("no recent date has bars for enough index members; refresh the data")


def latest_forecasts(bars, horizon: int, vix=None, announcements=None, as_of=None,
                     quantiles: dict | None = None, history: pd.DataFrame | None = None,
                     meetings: pd.DataFrame | None = None) -> pd.DataFrame:
    if as_of is not None:
        bars = bars[bars.available_time <= as_of]
        vix = None if vix is None else vix[vix.available_time <= as_of]
        announcements = None if announcements is None else announcements[announcements.available_time <= as_of]
    df, cols = build_volatility_frame(bars, horizon, vix, announcements)
    class_cols = cols
    if horizon in EXTRA_FEATURE_HORIZONS:
        df, extra_cols = _with_extra(df, bars)
        class_cols = cols + extra_cols
    if meetings is not None and horizon in EVENT_FEATURE_HORIZONS:
        known = meetings if as_of is None else meetings[meetings.available_time <= as_of]
        df, event_cols = add_event_features(df, known, horizon)
        class_cols = class_cols + event_cols
    df = df[membership_mask(df).to_numpy() & df.har_66.notna().to_numpy()]
    train = df.dropna(subset=["fwd_rv", "med_rv", "outcome_time"])
    train = train[train.fwd_rv > 0]
    if history is not None:
        old = history if as_of is None else history[history.outcome_time <= as_of]
        train = pd.concat([old.reindex(columns=train.columns), train], ignore_index=True)  # missing features stay NaN
    last = latest_complete_date(df)
    live = df[(df.event_time == last) & df.instrument_id.isin(instrument_ids_as_of(last)) & df.med_rv.notna()]
    X, X_live = train[cols].to_numpy(float), live[cols].to_numpy(float)
    prob = fit_predict_classifiers(train[class_cols].to_numpy(float), (train.fwd_rv > train.med_rv).astype(int).to_numpy(),
                                   live[class_cols].to_numpy(float))["blend"]
    logrv = fit_predict_regressors(X, np.log(train.fwd_rv.to_numpy()), X_live)["blend"]
    out = live[["instrument_id", "event_time", "med_rv"]].copy()
    out["horizon_days"] = horizon
    out["prob_above_usual"] = prob
    out["forecast_vol_annualised"] = np.exp(logrv) * np.sqrt(252 / horizon)  # median forecast
    out["usual_vol_annualised"] = out.med_rv * np.sqrt(252 / horizon)
    if quantiles:  # expected h-day price move: 90% range, from scripts/run_vol_ranges.py
        scale = np.exp(logrv)
        out["move_low_90"] = np.expm1(scale * float(quantiles["0.05"]))
        out["move_high_90"] = np.expm1(scale * float(quantiles["0.95"]))
    return out.sort_values("prob_above_usual", ascending=False).reset_index(drop=True)


def record(ledger, forecasts: pd.DataFrame, model: str = "v2") -> bool:
    """Append once per (data date, horizon); returns False if already recorded."""
    date, horizon = str(forecasts.event_time.iloc[0].date()), int(forecasts.horizon_days.iloc[0])
    for e in ledger.read():
        if e["kind"] == LEDGER_KIND and (e["payload"]["date"], e["payload"]["horizon_days"]) == (date, horizon):
            return False
    ranged = "move_low_90" in forecasts
    rows = [{**({"low_90": round(float(r.move_low_90), 5), "high_90": round(float(r.move_high_90), 5)} if ranged else {}),
             "instrument_id": r.instrument_id, "prob": round(float(r.prob_above_usual), 4), "med_rv": float(r.med_rv),
             "forecast_vol": round(float(r.forecast_vol_annualised), 4)} for r in forecasts.itertuples()]
    ledger.append(LEDGER_KIND, {"date": date, "horizon_days": horizon, "model": model, "forecasts": rows})
    return True


def graded_rows(ledger, bars: pd.DataFrame) -> pd.DataFrame:
    """One row per recorded forecast whose horizon has fully passed: date, horizon, y, prob, inside (range hit or NaN)."""
    closes = {i: g.sort_values("event_time").set_index("event_time").close for i, g in bars.groupby("instrument_id")}
    days = {i: c.index.strftime("%Y-%m-%d") for i, c in closes.items()}
    rows = []
    for e in ledger.read():
        if e["kind"] != LEDGER_KIND:
            continue
        p = e["payload"]
        h = p["horizon_days"]
        for f in p["forecasts"]:
            c = closes.get(f["instrument_id"])
            if c is None:
                continue
            i = int(np.searchsorted(days[f["instrument_id"]], p["date"]))
            if i + h < len(c):
                rv = np.sqrt((np.log(c.iloc[i: i + h + 1]).diff() ** 2).sum())
                move = c.iloc[i + h] / c.iloc[i] - 1
                inside = float(f["low_90"] <= move <= f["high_90"]) if "low_90" in f else np.nan
                rows.append((p["date"], h, int(rv > f["med_rv"]), f["prob"], inside))
    return pd.DataFrame(rows, columns=["date", "horizon", "y", "prob", "inside"])


def auc_interval(rows: pd.DataFrame, horizon: int, samples: int = 400, seed: int = 0):
    """95% interval for AUC, resampling blocks of `horizon` consecutive dates because nearby outcomes overlap.
    None with fewer than four blocks: too little to say anything."""
    dates = sorted(rows.date.unique())
    blocks = [rows[rows.date.isin(dates[i: i + horizon])] for i in range(0, len(dates), horizon)]
    if len(blocks) < 4:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        draw = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if draw.y.nunique() == 2:
            values.append(roc_auc_score(draw.y, draw.prob))
    return [round(float(v), 3) for v in np.percentile(values, [2.5, 97.5])] if values else None


def score(ledger, bars: pd.DataFrame) -> dict:
    """AUC, accuracy and range coverage of recorded forecasts whose horizon has fully elapsed."""
    graded = graded_rows(ledger, bars)
    out = {}
    for h, rows in graded.groupby("horizon") if len(graded) else []:
        y, p = rows.y.to_numpy(), rows.prob.to_numpy()
        out[str(h)] = {"n": len(y), "accuracy": round(float(((p > 0.5) == y).mean()), 4),
                       "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None}
        if rows.inside.notna().any():
            out[str(h)]["coverage_90"] = round(float(rows.inside.mean()), 4)
        dates, interval = rows.date.nunique(), auc_interval(rows, int(h))
        if interval:
            out[str(h)]["auc_ci95"] = interval
        if dates > 1:
            out[str(h)]["dates"] = int(dates)
    return out
