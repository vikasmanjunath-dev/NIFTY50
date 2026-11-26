"""Extract SBILIFE/HDFCLIFE fundamentals from their filing PDFs (found via the cached announcements).
Run after scripts/backfill_announcements.py: python3 scripts/backfill_insurance_fundamentals.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.insurance_fundamentals import INSURANCE_INSTRUMENT_IDS, ingest_insurance_fundamentals_for_instrument


def run() -> None:
    ann = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet")
    for c in ("event_time", "available_time"):
        ann[c] = pd.to_datetime(ann[c], utc=True).dt.tz_convert("Asia/Kolkata")
    ann = ann[ann["quality_flag"] == "PASS"]
    frames = []
    for iid in INSURANCE_INSTRUMENT_IDS:
        df = ingest_insurance_fundamentals_for_instrument(iid, ann)
        print(iid, len(df), df["quality_flag"].value_counts().to_dict(), flush=True)
        print(df[["event_time", "revenue", "profit", "eps_basic", "quality_flag"]].to_string(), flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(DATA_RAW_DIR / "insurance_fundamentals_nifty50.parquet", index=False)
    print("Wrote", len(out), "rows")


if __name__ == "__main__":
    run()
