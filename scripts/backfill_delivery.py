"""One-time backfill of NSE delivery data: one request per weekday covers the whole exchange.
Run: python3 scripts/backfill_delivery.py  (resumable: skips dates already in the cache)"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.bhavcopy import DELIVERY_COLUMNS, fetch_bhavcopy_csv, parse_bhavcopy

PATH = DATA_RAW_DIR / "delivery_nifty50.parquet"
START = "2021-01-01"


def run() -> None:
    parts = [pd.read_parquet(PATH)] if PATH.exists() else []
    done = set(pd.to_datetime(parts[0]["event_time"], utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d")) if parts else set()
    days = [d for d in pd.bdate_range(START, pd.Timestamp.today().normalize()) if d.strftime("%Y-%m-%d") not in done]
    print(f"{len(days)} weekdays to fetch ({len(done)} dates already cached)")
    got = skipped = 0
    bad: list[str] = []
    for i, day in enumerate(days, 1):
        try:
            text = fetch_bhavcopy_csv(day)
        except RuntimeError as exc:
            print("FAILED", day.date(), exc)
            continue
        try:
            df = parse_bhavcopy(text, day) if text else pd.DataFrame(columns=DELIVERY_COLUMNS)
        except Exception as exc:  # noqa: BLE001  (a malformed archive file must not abort or hide a date)
            bad.append(day.strftime("%Y-%m-%d"))
            print("UNPARSEABLE", day.date(), type(exc).__name__, str(exc)[:80], flush=True)
            continue
        if df.empty:
            skipped += 1
        else:
            parts.append(df)
            got += 1
        if i % 100 == 0:
            pd.concat(parts, ignore_index=True).to_parquet(PATH, index=False)
            print(f"{i}/{len(days)} fetched={got} non-trading={skipped}", flush=True)
        time.sleep(0.25)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(PATH, index=False)
    print("Unparseable dates:", bad)
    print(f"Wrote {len(out)} rows, {out['event_time'].nunique()} dates, quality: {out['quality_flag'].value_counts().to_dict()}")


if __name__ == "__main__":
    run()
