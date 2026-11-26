"""Daily S&P 500, US VIX and Nifty 50 index bars from Yahoo into data/raw/global/, each stamped with the moment it became known."""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.features.volatility_global import GLOBAL_SERIES, stamp

OUT = DATA_RAW_DIR / "global"


def fetch(ticker, start="2008-01-01"):
    for attempt in (1, 2, 3):
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.rename(columns=str.lower)[["open", "high", "low", "close"]]
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"no data for {ticker}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (ticker, _, _) in GLOBAL_SERIES.items():
        frame = stamp(fetch(ticker), name)
        frame.to_parquet(OUT / f"{name}.parquet", index=False)
        print(name, len(frame), frame.available_time.min().date(), frame.available_time.max().date())
