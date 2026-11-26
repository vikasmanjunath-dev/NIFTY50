"""Ingest OHLCV for the full survivorship-bias-free pool + the Nifty 50 index.
Run: python3 scripts/backfill_bars.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_analyser.data.ingest import ingest_instrument
from nifty_analyser.data.instrument_master import NIFTY50, NIFTY_INDEX


def run(start: str = "2021-01-01") -> None:
    ok, failed = 0, []
    for inst in [*NIFTY50.values(), NIFTY_INDEX]:
        if not inst.ingestible:
            print(f"SKIP  {inst.instrument_id:22s} (not ingestible)")
            continue
        try:
            df = ingest_instrument(inst, start)
            print(f"OK    {inst.instrument_id:22s} {inst.yahoo_ticker:14s} rows={len(df)} "
                  f"{df.event_time.min().date()}..{df.event_time.max().date()} "
                  f"fail={(df.quality_flag == 'FAIL').sum()} warn={(df.quality_flag == 'WARN').sum()}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {inst.instrument_id:22s} {inst.yahoo_ticker}: {exc}")
            failed.append(inst.instrument_id)
    print(f"\n{ok} ingested, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    run()
