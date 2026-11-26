"""Pre-2021 daily bars and India VIX (Yahoo), kept in data/raw/history/ so the live caches and ledgers are untouched.
Used only as extra training data: index membership is not reconstructed before 2021-09, so these rows carry survivorship bias."""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.ingest import validate_and_tag
from nifty_analyser.data.instrument_master import get_instrument, ingestible_instrument_ids
from nifty_analyser.data.macro import MACRO_SERIES, validate_and_tag_macro

OUT = DATA_RAW_DIR / "history"
START, END = "2008-01-01", "2021-01-01"  # end is exclusive


def fetch(ticker):
    for attempt in (1, 2, 3):
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.rename(columns=str.lower)
        time.sleep(1.5 * attempt)
    return None


OUT.mkdir(parents=True, exist_ok=True)
bars, missing = [], []
for iid in ingestible_instrument_ids():
    raw = fetch(get_instrument(iid).yahoo_ticker)
    if raw is None:
        missing.append(iid)
        continue
    bars.append(validate_and_tag(raw[["open", "high", "low", "close", "volume"]], iid))
all_bars = pd.concat(bars, ignore_index=True)
all_bars.to_parquet(OUT / "bars.parquet", index=False)
vix = validate_and_tag_macro(fetch(MACRO_SERIES["india_vix"]), "india_vix")
vix.to_parquet(OUT / "india_vix.parquet", index=False)
print(f"instruments {len(bars)} rows {len(all_bars)} fail {int((all_bars.quality_flag == 'FAIL').sum())} {all_bars.event_time.min().date()}..{all_bars.event_time.max().date()}")
print(f"vix rows {len(vix)} from {vix.event_time.min().date()}; no pre-2021 data: {missing}")
