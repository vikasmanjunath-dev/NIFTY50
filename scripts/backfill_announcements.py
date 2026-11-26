"""One-time backfill of NSE corporate announcements across the full Nifty 50 universe.

Uses the whole-exchange endpoint (~12-13 requests for a 5-year backfill),
not one request per (symbol, date-window) -- see ingest.py's module
docstring for the investigation that found this and the accuracy check that
confirmed it changes nothing about the data returned, only how many requests
it takes to get it (a first attempt at the naive per-symbol approach was
killed after 13 minutes once this was discovered).

Checkpoints incrementally to a partial file after every window, so a failure
partway through a multi-year backfill does not discard everything fetched so
far -- unlike the delivery/fundamentals backfills, where each run is short
enough that this wasn't necessary.

Run with: python3 scripts/backfill_announcements.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.documents.ingest import ingest_announcements_universe

_CHECKPOINT_PATH = DATA_RAW_DIR / "announcements_nifty50_partial.parquet"
_FINAL_PATH = DATA_RAW_DIR / "announcements_nifty50.parquet"


def run() -> None:
    bars = load_universe_bars()
    start = bars["event_time"].min().normalize().tz_localize(None)
    end = bars["event_time"].max().normalize().tz_localize(None)
    print(f"Backfilling whole-exchange announcements, {start.date()} .. {end.date()} "
          f"(filtered to our 48-instrument universe)...")

    accumulated: list[pd.DataFrame] = []

    def _on_window_complete(window_start, window_end, window_df):
        accumulated.append(window_df)
        running_total = sum(len(d) for d in accumulated)
        print(f"  window {window_start.date()} .. {window_end.date()}: "
              f"+{len(window_df)} rows (running total {running_total})")
        pd.concat(accumulated, ignore_index=True).to_parquet(_CHECKPOINT_PATH, index=False)

    all_docs = ingest_announcements_universe(start, end, on_window_complete=_on_window_complete)

    if all_docs.empty:
        print("No announcements ingested at all.")
        return

    all_docs.to_parquet(_FINAL_PATH, index=False)
    _CHECKPOINT_PATH.unlink(missing_ok=True)
    print(f"\nWrote {len(all_docs)} rows to {_FINAL_PATH}")
    print(all_docs["quality_flag"].value_counts())
    print()
    print("Instruments covered:", all_docs["instrument_id"].nunique(), "/ 48")
    print()
    print("Category distribution (top 15):")
    print(all_docs["category"].value_counts().head(15))


if __name__ == "__main__":
    run()
