"""Point-in-time feature store (Blueprint §3.2, §5.1): as_of / latest_as_of / assert_no_lookahead."""
from __future__ import annotations

import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR

from nifty_analyser.data.ingest import cache_path, load_cached
from nifty_analyser.data.instrument_master import ingestible_instrument_ids


class LookaheadError(AssertionError):
    """Raised when a frame contains a row not yet available at the query time."""


def load_universe_bars(instrument_ids: list[str] | None = None, drop_fail: bool = True) -> pd.DataFrame:
    """Concatenate cached bars for the instrument pool (instruments with no cache are skipped
    rather than silently fabricated)."""
    ids = instrument_ids or ingestible_instrument_ids()
    frames = [load_cached(i) for i in ids if cache_path(i).exists()]
    df = pd.concat(frames, ignore_index=True)
    if drop_fail:
        df = df[df["quality_flag"] != "FAIL"]
    return df.sort_values(["instrument_id", "event_time"]).reset_index(drop=True)


def as_of(df: pd.DataFrame, decision_time: pd.Timestamp, time_col: str = "available_time") -> pd.DataFrame:
    return df[df[time_col] <= decision_time].copy()


def latest_as_of(df, decision_time, time_col="available_time", group_col="instrument_id") -> pd.DataFrame:
    eligible = as_of(df, decision_time, time_col)
    if eligible.empty:
        return eligible
    return eligible.loc[eligible.groupby(group_col)[time_col].idxmax()].reset_index(drop=True)


def assert_no_lookahead(df: pd.DataFrame, decision_time: pd.Timestamp, time_col: str = "available_time") -> None:
    bad = df[df[time_col] > decision_time]
    if not bad.empty:
        ids = bad["instrument_id"].unique().tolist() if "instrument_id" in bad else []
        raise LookaheadError(f"{len(bad)} row(s) with {time_col} after {decision_time}; instruments: {ids}")


def _to_ist(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("event_time", "available_time"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert("Asia/Kolkata")
    return df


def load_delivery_universe(drop_fail: bool = True) -> pd.DataFrame:
    """Cached NSE delivery backfill (scripts/backfill_delivery.py)."""
    df = _to_ist(pd.read_parquet(DATA_RAW_DIR / "delivery_nifty50.parquet"))
    if drop_fail:
        df = df[df["quality_flag"] != "FAIL"]
    return df.sort_values(["instrument_id", "event_time"]).reset_index(drop=True)


def load_all_macro_series(drop_fail: bool = True) -> dict[str, pd.DataFrame]:
    from nifty_analyser.data.macro import MACRO_SERIES, load_cached_macro

    out = {}
    for name in MACRO_SERIES:
        df = load_cached_macro(name)
        out[name] = df[df["quality_flag"] != "FAIL"].reset_index(drop=True) if drop_fail else df
    return out


_FII_DII_FILES = {"fii": "fii_flow_nifty50.parquet", "dii": "dii_flow_nifty50.parquet"}


def load_all_fii_dii_flows(drop_fail: bool = True) -> dict[str, pd.DataFrame]:
    """FII and DII are separate files: their real coverage windows differ sharply (DII from 2026-06)."""
    out = {}
    for name, fn in _FII_DII_FILES.items():
        path = DATA_RAW_DIR / fn
        df = _to_ist(pd.read_parquet(path)) if path.exists() else pd.DataFrame()
        if drop_fail and not df.empty:
            df = df[df["quality_flag"] != "FAIL"]
        out[name] = df.sort_values("event_time").reset_index(drop=True) if not df.empty else df
    return out


_FUND_COLS = ["instrument_id", "event_time", "available_time", "source_id", "revision_id", "quality_flag",
              "missing_reason", "revenue", "profit", "eps_basic"]


def load_combined_fundamentals(drop_fail: bool = True) -> pd.DataFrame:
    """General XBRL fundamentals + insurance PDF fundamentals (disjoint instruments by construction)."""
    frames = [pd.read_parquet(DATA_RAW_DIR / fn)[_FUND_COLS]
              for fn in ("fundamentals_nifty50.parquet", "insurance_fundamentals_nifty50.parquet") if (DATA_RAW_DIR / fn).exists()]
    df = _to_ist(pd.concat(frames, ignore_index=True))
    if drop_fail:
        df = df[df["quality_flag"] != "FAIL"]
    return df.sort_values(["instrument_id", "event_time"]).reset_index(drop=True)
