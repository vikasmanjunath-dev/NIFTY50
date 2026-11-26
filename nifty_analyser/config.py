"""Project-wide paths and market constants."""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

IST = ZoneInfo("Asia/Kolkata")
NSE_CLOSE_HOUR_IST = 15
NSE_CLOSE_MINUTE_IST = 30
