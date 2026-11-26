"""NSE delivery-position data from the official ``sec_bhavdata_full`` archive (Blueprint §2.2).

One request per trading day covers the whole exchange (DELIV_QTY / DELIV_PER).

REAL BUG CAUGHT IN THE ORIGINAL BUILD, kept as a guard: for a non-trading date NSE's archive can
return HTTP 200 with the *previous* trading day's file instead of 404. Every file's own DATE1
column is therefore checked against the requested date; mismatches are discarded as non-trading.

``available_time`` = 21:00 IST on the trading date: the report is published after the close
(NSE gives no SLA; same-day files were always present well before that -- a documented
conservative assumption)."""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from nifty_analyser.data.instrument_master import NIFTY50
from nifty_analyser.data.schemas import QualityFlag
from nifty_analyser.data.timestamps import to_ist

SOURCE_ID = "nse_sec_bhavdata_full"
_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}
REPORT_HOUR_IST = 21
DELIVERY_COLUMNS = ["instrument_id", "event_time", "available_time", "source_id", "revision_id",
                    "quality_flag", "missing_reason", "traded_qty", "delivery_qty", "delivery_pct"]


def raw_to_csv_text(raw: bytes) -> str:
    """NSE occasionally serves an .xlsx workbook (a zip, magic bytes ``PK``) under the .csv URL
    (observed for 2022-08-08, a real trading day); convert it rather than drop the day."""
    if raw[:2] == b"PK":
        return pd.read_excel(io.BytesIO(raw)).to_csv(index=False)
    return raw.decode("utf-8", errors="replace")


def fetch_bhavcopy_csv(date: pd.Timestamp, max_retries: int = 3) -> str | None:
    """Raw CSV text, or ``None`` if NSE has no file for the date (HTTP 404)."""
    url = _URL.format(ddmmyyyy=date.strftime("%d%m%Y"))
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS), timeout=30) as r:
                return raw_to_csv_text(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            time.sleep(1.5 * attempt)
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed to fetch bhavcopy for {date.date()}")


def parse_bhavcopy(csv_text: str, requested_date: pd.Timestamp) -> pd.DataFrame:
    """Rows for pool instruments (EQ series) -- empty if the file's own date != requested date."""
    df = pd.read_csv(io.StringIO(csv_text), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    file_date = pd.to_datetime(df["DATE1"].astype(str).str.strip(), format="mixed", dayfirst=True).dt.normalize()
    if (file_date != requested_date.normalize()).any():
        return pd.DataFrame(columns=DELIVERY_COLUMNS)  # stale previous-day file served for a non-trading date
    df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    symbols = {sym: iid for iid, i in NIFTY50.items() for sym in (i.symbol, *i.aliases)}
    df = df[df["SYMBOL"].isin(symbols)]
    num = lambda c: pd.to_numeric(df[c].astype(str).str.strip().replace("-", np.nan), errors="coerce")  # noqa: E731
    qty, dq, dp = num("TTL_TRD_QNTY"), num("DELIV_QTY"), num("DELIV_PER")
    bad = dq.isna() | dp.isna() | (dp < 0) | (dp > 100)
    day = requested_date.normalize()
    return pd.DataFrame({
        "instrument_id": df["SYMBOL"].map(symbols).values,
        "event_time": to_ist(day), "available_time": to_ist(day + pd.Timedelta(hours=REPORT_HOUR_IST)),
        "source_id": SOURCE_ID, "revision_id": "v1",
        "quality_flag": np.where(bad, QualityFlag.FAIL.value, QualityFlag.PASS.value),
        "missing_reason": np.where(bad, "missing or out-of-range delivery data", None),
        "traded_qty": qty.values, "delivery_qty": dq.values, "delivery_pct": dp.values,
    })[DELIVERY_COLUMNS]
