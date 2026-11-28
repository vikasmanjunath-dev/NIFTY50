"""Macro series ingestion (Blueprint §2.5): India VIX, USD-INR, Brent crude.

A separate pipeline from the equity adapter on purpose: these series routinely report
zero volume as their NORMAL state, so the equity "flat OHLC + zero volume = FAIL" rule
would misfire. ``available_time`` models when each series' daily value is really
knowable relative to NSE's 15:30 IST close: India VIX closes with NSE (same day);
USD-INR and Brent settle after NSE has closed, so their value for date D is treated as
available at 09:00 IST on D+1 -- a conservative, documented assumption (not a verified SLA).
"""
from __future__ import annotations

import time

import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.schemas import QualityFlag
from nifty_analyser.data.timestamps import eod_bar_available_time, to_ist, drop_unfinished, write_parquet_atomic

SOURCE_ID = "yfinance_macro"
MACRO_SERIES: dict[str, str] = {"india_vix": "^INDIAVIX", "usd_inr": "INR=X", "brent_crude": "BZ=F"}
_NEXT_MORNING = {"usd_inr", "brent_crude"}
MACRO_COLUMNS = ["series_name", "event_time", "available_time", "source_id", "revision_id",
                 "quality_flag", "missing_reason", "close"]


def available_time_for(series_name: str, event_date: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(event_date).normalize().tz_localize(None) if pd.Timestamp(event_date).tzinfo else pd.Timestamp(event_date).normalize()
    if series_name in _NEXT_MORNING:
        return to_ist(day + pd.Timedelta(days=1, hours=9))
    return eod_bar_available_time(day)


def validate_and_tag_macro(raw: pd.DataFrame, series_name: str) -> pd.DataFrame:
    df = raw[["close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    bad = df["close"].isna() | (df["close"] <= 0)
    return pd.DataFrame({
        "series_name": series_name,
        "event_time": [to_ist(d) for d in df.index],
        "available_time": [available_time_for(series_name, d) for d in df.index],
        "source_id": SOURCE_ID, "revision_id": "v1",
        "quality_flag": [QualityFlag.FAIL.value if b else QualityFlag.PASS.value for b in bad],
        "missing_reason": ["missing or non-positive close" if b else None for b in bad],
        "close": df["close"].values,
    })[MACRO_COLUMNS].reset_index(drop=True)


def _path(name: str):
    return DATA_RAW_DIR / f"macro_{name}.parquet"


def ingest_macro(name: str, start: str = "2021-01-01", max_retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(MACRO_SERIES[name], start=start, progress=False, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError(f"empty response for {name}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            out = drop_unfinished(validate_and_tag_macro(df.rename(columns=str.lower), name))
            write_parquet_atomic(out, _path(name))
            return out
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed to ingest macro series {name}") from last


def load_cached_macro(name: str) -> pd.DataFrame:
    df = pd.read_parquet(_path(name))
    for c in ("event_time", "available_time"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert("Asia/Kolkata")
    return df
