"""yfinance OHLCV ingestion with row-level quality tagging and Parquet caching (Blueprint §5.2)."""
from __future__ import annotations

import logging
import re
import time

import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.instrument_master import Instrument, get_instrument
from nifty_analyser.data.schemas import QualityFlag
from nifty_analyser.data.timestamps import eod_bar_available_time, to_ist, drop_unfinished

logger = logging.getLogger(__name__)
SOURCE_ID = "yfinance"
BAR_COLUMNS = [
    "instrument_id", "event_time", "available_time", "source_id", "revision_id",
    "quality_flag", "missing_reason", "open", "high", "low", "close", "volume",
]
_MAX_GAP_DAYS = 10


def cache_path(instrument_id: str):
    return DATA_RAW_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", instrument_id) + ".parquet")


def fetch_ohlcv(ticker: str, start: str, max_retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError(f"empty response for {ticker}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed to fetch {ticker}") from last_exc


def validate_and_tag(raw: pd.DataFrame, instrument_id: str) -> pd.DataFrame:
    """Tag each bar PASS/WARN/FAIL. FAIL rows (non-positive/inconsistent prices, or the
    degenerate flat-OHLC zero-volume placeholders Yahoo emits on non-trading days)
    are kept but flagged so downstream code can exclude them explicitly."""
    df = raw.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    flag = pd.Series(QualityFlag.PASS.value, index=df.index)
    reason = pd.Series(None, index=df.index, dtype=object)

    def mark(mask, level, why):
        nonlocal flag, reason
        mask = mask & (flag != QualityFlag.FAIL.value)
        flag = flag.where(~mask, level)
        reason = reason.where(~mask, why)

    px = df[["open", "high", "low", "close"]]
    mark(df[["open", "high", "low", "close"]].isna().any(axis=1), "FAIL", "missing price")
    mark((px <= 0).any(axis=1), "FAIL", "non-positive price")
    mark((df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1)), "FAIL", "inconsistent OHLC")
    mark((df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"]) & (df["volume"] == 0), "FAIL", "flat OHLC with zero volume (placeholder bar)")
    gap = df.index.to_series().diff().dt.days
    mark(gap > _MAX_GAP_DAYS, "WARN", f"gap > {_MAX_GAP_DAYS} days before this bar")

    out = pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "event_time": [to_ist(d) for d in df.index],
            "available_time": [eod_bar_available_time(d) for d in df.index],
            "source_id": SOURCE_ID,
            "revision_id": "v1",
            "quality_flag": flag.values,
            "missing_reason": reason.values,
            **{c: df[c].values for c in ("open", "high", "low", "close", "volume")},
        }
    )
    return out[BAR_COLUMNS].reset_index(drop=True)


def ingest_instrument(instrument: Instrument | str, start: str = "2021-01-01") -> pd.DataFrame:
    inst = get_instrument(instrument) if isinstance(instrument, str) else instrument
    tagged = drop_unfinished(validate_and_tag(fetch_ohlcv(inst.yahoo_ticker, start), inst.instrument_id))
    tagged.to_parquet(cache_path(inst.instrument_id), index=False)
    return tagged


def load_cached(instrument_id: str) -> pd.DataFrame:
    df = pd.read_parquet(cache_path(instrument_id))
    for c in ("event_time", "available_time"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert("Asia/Kolkata")
    return df
