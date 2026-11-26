"""FII/DII cash-equity net flow from NSDL's official FPI Monitor archive (Blueprint §2.5).

Source investigation (each tested directly): NSE's live page/API (`fiidiiTradeReact`) exposes only
the LATEST trading day; a free third-party aggregator had ~157 days with seeded/estimated rows; NSDL's
own archive (the depository-confirmed source NSE itself points to) has real depth back to 2021.
It is an ASP.NET WebForms page (GET for __VIEWSTATE/__EVENTVALIDATION, POST `btnSubmit1` with the date).
Efficiency: the FII archive returns the whole month-to-date in ONE response, so a month needs 1 request.
NSDL's separate DII archive is one-day-per-request and only has data from 2026-06 (real source limit,
reported as such, never padded).

Real bugs caught by live runs (regression-tested): (a) the table also holds "Total for <Month/Year>"
aggregate rows that match the Equity/Sub-total filter but have no date -> filter on the date format;
(b) a missing value became NaT after DataFrame.iterrows() mixed it with a datetime column, so
``is not None`` missed it -> use pd.isna().
``available_time`` = event day + 1 (NSDL says confirmed data covers "up to the previous trading day(s)"
-- a conservative, documented assumption, not a verified SLA)."""
from __future__ import annotations

import http.cookiejar
import io
import logging
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from nifty_analyser.data.insurance_fundamentals import parse_indian_number
from nifty_analyser.data.schemas import QualityFlag
from nifty_analyser.data.timestamps import to_ist

logger = logging.getLogger(__name__)
SOURCE_ID = "nsdl_fpi_dii_archive"
_FII_URL = "https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx"
_DII_URL = "https://www.fpi.nsdl.co.in/web/Users/DIIGenerateReport.aspx?Rep=A"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}
_LAG_DAYS = 1
_DATE = re.compile(r"^\d{2}-[A-Za-z]{3}-\d{4}$")
FLOW_COLUMNS = ["series_name", "event_time", "available_time", "source_id", "revision_id",
                "quality_flag", "missing_reason", "net_investment_cr"]


def _hidden(html: str, field: str) -> str:
    m = re.search(rf'id="{field}" value="([^"]*)"', html)
    return m.group(1) if m else ""


def _fetch_report_html(opener, url: str, to_date: str, max_retries: int = 3) -> str:
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            html = opener.open(urllib.request.Request(url, headers=_HEADERS), timeout=30).read().decode("utf-8", "replace")
            form = {"__EVENTTARGET": "btnSubmit1", "__EVENTARGUMENT": "", "txtDate": to_date, "hdnDate": to_date,
                    **{k: _hidden(html, k) for k in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION")}}
            req = urllib.request.Request(url, data=urllib.parse.urlencode(form).encode(),
                                         headers={**_HEADERS, "Content-Type": "application/x-www-form-urlencoded", "Referer": url})
            return opener.open(req, timeout=30).read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed to fetch {url} (to_date={to_date})") from last


def _opener():
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def fetch_fii_month(to_date: str) -> pd.DataFrame:
    """Every trading day's FII/FPI equity net flow from the month start through ``to_date`` (DD-Mon-YYYY)."""
    html = _fetch_report_html(_opener(), _FII_URL, to_date)
    if "No Data Found" in html:
        return pd.DataFrame(columns=["event_time", "net_investment_cr"])
    t = pd.read_html(io.StringIO(html))[1]
    t.columns = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
    t = t[(t["Debt/Equity"] == "Equity") & (t["Investment Route"] == "Sub-total")]
    t = t[t["Reporting Date"].astype(str).str.match(_DATE)]
    return pd.DataFrame({"event_time": [pd.to_datetime(d, format="%d-%b-%Y") for d in t["Reporting Date"]],
                         "net_investment_cr": [parse_indian_number(str(v)) for v in t["Net Investment (Rs Crore)"]]})


def fetch_dii_day(to_date: str) -> float | None:
    html = _fetch_report_html(_opener(), _DII_URL, to_date)
    if "No Data Found" in html:
        return None
    t = pd.read_html(io.StringIO(html))[2]
    t.columns = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
    row = t[(t["Type of DII"] == "Total") & (t["Investment Route"] == "Equity Total")]
    return None if row.empty else parse_indian_number(str(row.iloc[0]["Net Investment (Rs. Crore)"]))


def _flow_rows(series_name: str, items: list[tuple[pd.Timestamp, float | None]]) -> pd.DataFrame:
    rows = []
    for day, net in items:
        ev = to_ist(pd.Timestamp(day).normalize())
        missing = pd.isna(net)
        rows.append({"series_name": series_name, "event_time": ev, "available_time": ev + pd.Timedelta(days=_LAG_DAYS),
                     "source_id": SOURCE_ID, "revision_id": "v1",
                     "quality_flag": (QualityFlag.FAIL if missing else QualityFlag.PASS).value,
                     "missing_reason": "no Equity Sub-total row found" if missing else None,
                     "net_investment_cr": None if missing else net})
    return pd.DataFrame(rows, columns=FLOW_COLUMNS)


def ingest_fii_flow(start: str, end: str) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    for month_end in [*pd.date_range(s, e, freq="M"), e]:
        try:
            parts.append(fetch_fii_month(month_end.strftime("%d-%b-%Y")))
        except Exception as exc:  # noqa: BLE001
            logger.warning("FII month fetch failed for %s: %s", month_end.date(), exc)
        time.sleep(1.0)
    if not parts:
        return pd.DataFrame(columns=FLOW_COLUMNS)
    df = pd.concat(parts, ignore_index=True).drop_duplicates("event_time")
    df = df[(df["event_time"] >= s) & (df["event_time"] <= e)].sort_values("event_time")
    return _flow_rows("fii_net_equity_flow", list(zip(df["event_time"], df["net_investment_cr"])))


def ingest_dii_flow(start: str, end: str) -> pd.DataFrame:
    items = []
    for day in pd.bdate_range(start, end):
        try:
            net = fetch_dii_day(day.strftime("%d-%b-%Y"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("DII day fetch failed for %s: %s", day.date(), exc)
            continue
        if net is not None:
            items.append((day, net))
        time.sleep(0.5)
    return _flow_rows("dii_net_equity_flow", items)
