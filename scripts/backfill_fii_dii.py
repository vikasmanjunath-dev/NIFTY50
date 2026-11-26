"""Backfill FII (2021-01 -> today, ~70 monthly requests) and DII (2026-06 -> today) net equity flow.
Run: python3 scripts/backfill_fii_dii.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.fii_dii import ingest_dii_flow, ingest_fii_flow


def run() -> None:
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    fii = ingest_fii_flow("2021-01-01", today)
    fii.to_parquet(DATA_RAW_DIR / "fii_flow_nifty50.parquet", index=False)
    print("FII:", len(fii), fii["event_time"].min(), fii["event_time"].max(), fii["quality_flag"].value_counts().to_dict(), flush=True)
    dii = ingest_dii_flow("2026-05-01", today)
    dii.to_parquet(DATA_RAW_DIR / "dii_flow_nifty50.parquet", index=False)
    print("DII:", len(dii), dii["event_time"].min() if len(dii) else None, dii["event_time"].max() if len(dii) else None, flush=True)


if __name__ == "__main__":
    run()
