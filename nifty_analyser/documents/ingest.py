"""NSE corporate announcements ingestion (Blueprint §4.3, §4.5).

The document-side analogue of ``data/ingest.py`` (OHLCV) and
``data/fundamentals.py``: same point-in-time discipline, same
investigate-before-trusting posture. Three things were confirmed directly
against the live API before settling on this design:

1. The **per-symbol** endpoint (``&symbol=X``) caps how wide a single
   ``from_date``/``to_date`` window can be (~60 days observed empirically --
   a 90-day request silently returned only its last ~59 days).
2. The **whole-exchange** endpoint (same URL, omit ``&symbol=``) has no such
   cap -- a full 365-day window was requested and honoured in full (179,470
   records, ~64s). This is the same efficiency pattern as the delivery-data
   source (``bhavcopy.py``): one request covers every listed company, not
   one per symbol. Cross-checked directly: filtering a whole-exchange
   response down to one symbol produces an *identical* set of record IDs to
   querying that symbol directly -- so this is a pure speed win, not an
   accuracy trade-off. A first backfill attempt using the naive
   one-request-per-symbol-per-window approach was killed after ~13 minutes
   once this was discovered (it would have taken ~20-25 minutes and ~1900
   requests; the whole-exchange approach needs ~12-13 requests for the same
   5-year, 48-instrument backfill).
3. ``exchdisstime`` (when NSE actually disseminated the announcement) is
   materially more precise than anything used elsewhere in this project so
   far -- to the second, and typically only ~1 second after the filing
   timestamp itself (NSE's own "difference" field). This is used as
   ``available_time`` directly, no assumed lag needed.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request

import pandas as pd

from nifty_analyser.data.instrument_master import NIFTY50, get_instrument
from nifty_analyser.data.schemas import QualityFlag
from nifty_analyser.data.timestamps import to_ist

logger = logging.getLogger(__name__)

SOURCE_ID = "nse_corporate_announcements"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}
_URL_TEMPLATE = (
    "https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}"
    "&from_date={from_date}&to_date={to_date}"
)
_UNIVERSE_URL_TEMPLATE = (
    "https://www.nseindia.com/api/corporate-announcements?index=equities"
    "&from_date={from_date}&to_date={to_date}"
)
# Conservative vs. the ~60-day cap observed empirically on the per-symbol
# endpoint -- kept only for fetch_announcements_window (single-symbol use,
# e.g. an ad-hoc lookup), not for the universe backfill.
_WINDOW_DAYS = 45
# The whole-exchange endpoint has no observed cap (365 days confirmed
# honoured in full); 150 is chosen as a safety margin against an undocumented
# server-side limit and to keep each individual request's payload (tens of
# MB at 180+ days) and failure blast-radius smaller.
_UNIVERSE_WINDOW_DAYS = 150

DOCUMENT_COLUMNS = [
    "instrument_id",
    "doc_id",
    "event_time",
    "available_time",
    "source_id",
    "revision_id",
    "quality_flag",
    "missing_reason",
    "category",
    "text",
    "attachment_url",
    "isin",
]


def _http_get_json(url: str, max_retries: int = 3) -> list[dict]:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.0 * attempt)
    logger.warning("giving up on %s after %d attempts: %s", url, max_retries, last_exc)
    return []


def _date_windows(start: pd.Timestamp, end: pd.Timestamp, window_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    cur = start
    while cur <= end:
        window_end = min(cur + pd.Timedelta(days=window_days - 1), end)
        windows.append((cur, window_end))
        cur = window_end + pd.Timedelta(days=1)
    return windows


def _quality_check(record: dict) -> tuple[QualityFlag, str | None]:
    if not record.get("attchmntText") and not record.get("desc"):
        return QualityFlag.FAIL, "no text content in announcement"
    if not record.get("exchdisstime"):
        return QualityFlag.WARN, "missing exchange dissemination timestamp; falling back to an_dt"
    return QualityFlag.PASS, None


def _to_document_row(record: dict, instrument_id: str) -> dict:
    quality, reason = _quality_check(record)

    event_time_raw = record.get("an_dt")
    available_time_raw = record.get("exchdisstime") or record.get("an_dt")
    event_time = to_ist(pd.to_datetime(event_time_raw, format="%d-%b-%Y %H:%M:%S"))
    available_time = to_ist(pd.to_datetime(available_time_raw, format="%d-%b-%Y %H:%M:%S"))

    text = record.get("attchmntText") or record.get("desc") or ""

    return {
        "instrument_id": instrument_id,
        "doc_id": f"NSE_ANN_{record.get('seq_id')}",
        "event_time": event_time,
        "available_time": available_time,
        "source_id": SOURCE_ID,
        "revision_id": "v1",
        "quality_flag": quality.value,
        "missing_reason": reason,
        "category": record.get("desc") or "Uncategorised",
        "text": text,
        "attachment_url": record.get("attchmntFile"),
        "isin": record.get("sm_isin"),
    }


def fetch_announcements_window(
    symbol: str, from_date: pd.Timestamp, to_date: pd.Timestamp
) -> list[dict]:
    url = _URL_TEMPLATE.format(
        symbol=urllib.parse.quote(symbol, safe=""),
        from_date=from_date.strftime("%d-%m-%Y"),
        to_date=to_date.strftime("%d-%m-%Y"),
    )
    return _http_get_json(url)


def fetch_announcements_window_universe(
    from_date: pd.Timestamp, to_date: pd.Timestamp, timeout: int = 90
) -> list[dict]:
    """Whole-exchange announcements for one date window (every listed company).

    A larger timeout than other fetch functions in this project: a wide
    window's response body can run to tens of MB (confirmed: 180 days ~50MB,
    365 days ~124MB), and the server itself takes tens of seconds to build it.
    """
    url = _UNIVERSE_URL_TEMPLATE.format(
        from_date=from_date.strftime("%d-%m-%Y"), to_date=to_date.strftime("%d-%m-%Y")
    )
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("universe fetch attempt %d/3 failed for %s..%s: %s",
                            attempt, from_date.date(), to_date.date(), exc)
            time.sleep(2.0 * attempt)
    logger.error("giving up on universe window %s..%s: %s", from_date.date(), to_date.date(), last_exc)
    return []


def ingest_announcements_for_instrument(
    instrument_id: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Fetch every announcement for one instrument across [start, end], point-in-time tagged.

    Chunks the request into ``_WINDOW_DAYS``-wide windows internally (see
    module docstring) and de-duplicates by ``doc_id`` across window
    boundaries (a document can appear in two adjacent windows' results if its
    timestamp sits exactly on a boundary).
    """
    instrument = get_instrument(instrument_id)
    rows: list[dict] = []
    seen_doc_ids: set[str] = set()

    for window_start, window_end in _date_windows(start, end, _WINDOW_DAYS):
        records = fetch_announcements_window(instrument.symbol, window_start, window_end)
        for record in records:
            row = _to_document_row(record, instrument_id)
            if row["doc_id"] in seen_doc_ids:
                continue
            seen_doc_ids.add(row["doc_id"])
            rows.append(row)
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame(columns=DOCUMENT_COLUMNS)
    return pd.DataFrame(rows)[DOCUMENT_COLUMNS].sort_values("event_time").reset_index(drop=True)


def ingest_announcements_universe(
    start: pd.Timestamp,
    end: pd.Timestamp,
    on_window_complete=None,
) -> pd.DataFrame:
    """Fetch every announcement for the whole Nifty 50 universe across
    [start, end] in ~(range_days / 150) whole-exchange requests, instead of
    one request per (symbol, 45-day window) -- see module docstring for the
    ~150x request-count reduction this represents and the accuracy check
    that confirmed it changes nothing about the actual data returned.

    ``on_window_complete``, if given, is called after each window with the
    window's own (already-filtered-to-universe) DataFrame -- used by the
    backfill script to checkpoint progress incrementally, so a failure deep
    into a multi-year backfill does not discard everything fetched so far.
    """
    symbol_to_instrument = {sym: iid for iid, inst in NIFTY50.items() for sym in (inst.symbol, *inst.aliases)}
    seen_doc_ids: set[str] = set()
    all_rows: list[dict] = []

    for window_start, window_end in _date_windows(start, end, _UNIVERSE_WINDOW_DAYS):
        records = fetch_announcements_window_universe(window_start, window_end)
        window_rows: list[dict] = []
        for record in records:
            instrument_id = symbol_to_instrument.get(record.get("symbol"))
            if instrument_id is None:
                continue  # not one of our 48 instruments -- the whole point of this endpoint
            row = _to_document_row(record, instrument_id)
            if row["doc_id"] in seen_doc_ids:
                continue
            seen_doc_ids.add(row["doc_id"])
            window_rows.append(row)

        all_rows.extend(window_rows)
        if on_window_complete is not None:
            window_df = (
                pd.DataFrame(window_rows)[DOCUMENT_COLUMNS] if window_rows else pd.DataFrame(columns=DOCUMENT_COLUMNS)
            )
            on_window_complete(window_start, window_end, window_df)

    if not all_rows:
        return pd.DataFrame(columns=DOCUMENT_COLUMNS)
    return pd.DataFrame(all_rows)[DOCUMENT_COLUMNS].sort_values("event_time").reset_index(drop=True)
