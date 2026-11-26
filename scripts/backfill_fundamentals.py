"""One-time backfill of NSE XBRL fundamentals across the full Nifty 50 universe.

One HTTP request for the filing list + one per XBRL file, per instrument
(unlike delivery data, fundamentals have no whole-exchange-in-one-file
shortcut -- NSE's financial-results API is queried per symbol).

Run with: python3 scripts/backfill_fundamentals.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.fundamentals import (
    cross_validate_eps_against_yfinance,
    ingest_fundamentals_for_instrument,
)
from nifty_analyser.data.instrument_master import get_instrument, ingestible_instrument_ids

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)


def run() -> None:
    ids = ingestible_instrument_ids()
    print(f"Backfilling fundamentals for {len(ids)} instruments...")

    parts = []
    for i, instrument_id in enumerate(ids, 1):
        try:
            fundamentals = ingest_fundamentals_for_instrument(instrument_id)
            if fundamentals.empty:
                print(f"{i:2d}/{len(ids)} {instrument_id:20s} NO FILINGS FOUND")
                continue
            instrument = get_instrument(instrument_id)
            fundamentals = cross_validate_eps_against_yfinance(fundamentals, instrument)
            parts.append(fundamentals)
            counts = fundamentals["quality_flag"].value_counts().to_dict()
            match_count = (fundamentals["yf_cross_check"] == "MATCH").sum()
            print(f"{i:2d}/{len(ids)} {instrument_id:20s} filings={len(fundamentals)} "
                  f"quality={counts} yf_matches={match_count}")
        except Exception as exc:  # noqa: BLE001
            print(f"{i:2d}/{len(ids)} {instrument_id:20s} FAILED: {exc}")

    if not parts:
        print("No fundamentals ingested at all.")
        return

    all_fundamentals = pd.concat(parts, ignore_index=True)
    path = DATA_RAW_DIR / "fundamentals_nifty50.parquet"
    all_fundamentals.to_parquet(path, index=False)
    print(f"\nWrote {len(all_fundamentals)} rows to {path}")
    print(all_fundamentals["quality_flag"].value_counts())
    print()
    print("Cross-validation summary:", all_fundamentals["yf_cross_check"].value_counts().to_dict())


if __name__ == "__main__":
    run()
