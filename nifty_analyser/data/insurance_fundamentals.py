"""Insurance-sector fundamentals via PDF table extraction (SBILIFE, HDFCLIFE).

These IRDAI-regulated insurers file signed PDFs (``hasXbrl: false``), not XBRL, so the general
XBRL fundamentals pipeline returns nothing for them. Filing PDFs are discovered from the cached
announcements corpus ("Financial Result Updates" with an attachment) and parsed with pdfplumber.

Four real bugs were found by reading extracted values against live filings (each regression-tested):
 (1) a row's leading serial-number cell ("2") was read as the value -> only cells AFTER the label;
 (2) pdfplumber splits a label mid-word across cells ('2 Net premium in' | 'come1') -> match on
     whitespace-stripped cells joined together;
 (3) multi-line segment cells ("Net Premium\\nIncome from Investments...") glue into a false match
     when newlines are stripped -> multi-line cells are matched line-by-line, never joined;
 (4) some filings embed the same table twice at different extraction fidelity (the earlier copy
     split a number's last digit into the next cell: '1,437,93' + '8 ...') -> prefer the LAST match.
KNOWN LIMITATION: bespoke 2-company, 3-field (revenue/premium, profit, EPS) extractor."""
from __future__ import annotations

import io
import logging
import re
import time
import urllib.request

import pandas as pd

from nifty_analyser.data.schemas import QualityFlag
from nifty_analyser.data.timestamps import to_ist

logger = logging.getLogger(__name__)
SOURCE_ID = "nse_insurance_pdf_extraction"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}
INSURANCE_INSTRUMENT_IDS = ["NSE:SBILIFE", "NSE:HDFCLIFE"]
_ROW_KEYWORDS = {
    "revenue": ["net premium income", "total income"],
    "profit": ["net profit/ loss after tax", "net profit/loss after tax",
               "profit/(loss) after tax and before extraordinary items",
               "profit/ (loss) after tax and before extraordinary items"],
    "eps_basic": ["basic eps before and after extraordinary items", "a) basic eps"],
}
_PERIOD_END = re.compile(r"period ended (\w+ \d{1,2},? \d{4})", re.IGNORECASE)
INSURANCE_FUNDAMENTALS_COLUMNS = ["instrument_id", "event_time", "available_time", "source_id", "revision_id",
                                  "quality_flag", "missing_reason", "revenue", "profit", "eps_basic"]


def _http_get(url: str, max_retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS), timeout=45) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed to fetch {url} after {max_retries} attempts") from last


def parse_indian_number(text: str | None) -> float | None:
    """Thousands commas, kerning artifacts ('7 .30', '1 ,086'), parenthesised negatives; NA/NIL/- -> None."""
    if text is None:
        return None
    t = text.strip()
    if t in ("", "-", "NA", "NIL"):
        return None
    neg = t.startswith("(") and t.endswith(")")
    t = re.sub(r"\s+", "", t.strip("()")).replace(",", "").replace("₹", "").replace("%", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def extract_all_table_rows(pdf_bytes: bytes) -> list[list[str | None]]:
    import pdfplumber

    rows: list[list[str | None]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                rows.extend(table)
    return rows


def _find_match_cell_index(row: list[str | None], norm_keyword: str) -> int | None:
    concat, ends = "", []
    for i, cell in enumerate(row):
        if not cell or "\n" in cell:
            continue
        concat += re.sub(r"\s+", "", str(cell).lower())
        ends.append((len(concat), i))
    pos = concat.find(norm_keyword)
    if pos != -1:
        idx = next((i for end, i in ends if end >= pos + len(norm_keyword)), None)
        if idx is not None:
            return idx
    for i, cell in enumerate(row):
        if cell and "\n" in cell and any(norm_keyword in re.sub(r"\s+", "", ln.lower()) for ln in cell.split("\n")):
            return i
    return None


def find_first_matching_row_value(rows: list[list[str | None]], keywords: list[str]) -> float | None:
    """Value in the first numeric cell AFTER the label cell, taking the LAST matching row (bug 4)."""
    for kw in keywords:
        nk = re.sub(r"\s+", "", kw.lower())
        for row in reversed(rows):
            idx = _find_match_cell_index(row, nk)
            if idx is None:
                continue
            for cell in row[idx + 1:]:
                v = parse_indian_number(cell) if cell else None
                if v is not None:
                    return v
    return None


def extract_period_end(text: str | None) -> pd.Timestamp | None:
    m = _PERIOD_END.search(text or "")
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1).replace(",", ""), format="%B %d %Y")
    except ValueError:
        return None


def extract_fundamentals_from_filing(pdf_url: str) -> dict[str, float | None]:
    rows = extract_all_table_rows(_http_get(pdf_url))
    return {f: find_first_matching_row_value(rows, kws) for f, kws in _ROW_KEYWORDS.items()}


def _row(iid, event_time, available_time, quality, reason, facts):
    return {"instrument_id": iid, "event_time": event_time, "available_time": available_time, "source_id": SOURCE_ID,
            "revision_id": "v1", "quality_flag": quality.value, "missing_reason": reason, **facts}


def ingest_insurance_fundamentals_for_instrument(instrument_id: str, announcements: pd.DataFrame, pause: float = 2.0) -> pd.DataFrame:
    filings = announcements[(announcements["instrument_id"] == instrument_id)
                            & (announcements["category"] == "Financial Result Updates")
                            & announcements["attachment_url"].notna()]
    empty = {"revenue": None, "profit": None, "eps_basic": None}
    rows = []
    for _, f in filings.iterrows():
        period_end = extract_period_end(f["text"])
        if period_end is None:
            rows.append(_row(instrument_id, f["event_time"], f["available_time"], QualityFlag.FAIL,
                             "could not parse period-end date from announcement text", empty))
            continue
        try:
            facts = extract_fundamentals_from_filing(f["attachment_url"])
            ok = any(v is not None for v in facts.values())
            q, reason = (QualityFlag.PASS, None) if ok else (QualityFlag.FAIL, "no target rows found in PDF tables")
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF extraction failed for %s: %s", f["attachment_url"], exc)
            facts, q, reason = empty, QualityFlag.FAIL, f"PDF fetch/parse error: {exc}"
        rows.append(_row(instrument_id, to_ist(period_end), f["available_time"], q, reason, facts))
        time.sleep(pause)
    if not rows:
        return pd.DataFrame(columns=INSURANCE_FUNDAMENTALS_COLUMNS)
    return pd.DataFrame(rows)[INSURANCE_FUNDAMENTALS_COLUMNS].sort_values("event_time").reset_index(drop=True)
