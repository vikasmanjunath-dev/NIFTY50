"""Fundamentals ingestion (Blueprint §2.4) — NSE's own XBRL quarterly filings.

Closes Phase 2's last data-source gap. yfinance's fundamental history for
Indian stocks is far too sparse to be useful (3-5 quarters only, checked
directly during development) -- NSE's own `corporates-financial-results` API
is the real source: it lists every quarterly filing with an exact
``broadCastDate``/``filingDate`` timestamp (to the second) and a link to the
filing's raw XBRL, which contains standard IND-AS-taxonomy-tagged figures
(Revenue, Profit, EPS, ...).

THE HARD PART, investigated before writing a line of extraction code: a
single company's XBRL filing contains the SAME tag (e.g.
``RevenueFromOperations``) under *multiple* ``contextRef`` values whose raw
period dates can be identical in the XML even though one represents the
single quarter and another represents the year-to-date cumulative figure --
there is no way to tell them apart from the context's own dates alone in at
least some filers' XBRL. This was resolved empirically, not assumed: fetched
Reliance's Q3-FY25 *Consolidated* filing (a case with an independently
plausible, well-known-order-of-magnitude quarterly revenue) and confirmed the
*smaller* of the two candidate context values matches the real quarterly
figure, while the larger matches roughly the multi-quarter cumulative. That
"smallest value per tag" heuristic is what ``_select_quarterly_value`` uses.

BECAUSE this heuristic is not logically guaranteed to hold for every company
in every quarter (e.g. a quarter with an unusual loss inside a
cumulatively-positive year could invert it), every extracted quarterly EPS
value is cross-checked against yfinance's own (short but independently
sourced) quarterly EPS history wherever the two overlap, and flagged
WARN/FAIL rather than silently trusted when they disagree beyond a tolerance
-- turning the residual extraction risk into something measured and visible,
not hidden.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from nifty_analyser.data.instrument_master import Instrument, get_instrument
from nifty_analyser.data.schemas import QualityFlag
from nifty_analyser.data.timestamps import to_ist

logger = logging.getLogger(__name__)

SOURCE_ID = "nse_xbrl_financial_results"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}
_LIST_URL = "https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={symbol}&period=Quarterly"

# IND-AS XBRL tag names for the four fields the blueprint's fundamental-feature
# table calls out (§2.4): revenue, profit, EPS. Debt/Equity and ROE are
# deliberately NOT extracted from XBRL in this first pass -- a spot check
# found the `DebtEquityRatio` tag populated as an untrustworthy 0.00 for a
# real company with genuine debt, i.e. that specific tag is not reliably
# filled across issuers, so it is left out rather than shipped silently wrong.
#
# Banks file under a DIFFERENT set of tag names within the same `in-bse-fin`
# namespace (confirmed directly against ICICI Bank's real filing after every
# non-financial-sector tag came back empty for every single banking stock in
# the full-universe backfill -- a systematic, not random, failure pattern).
# Each field therefore lists candidate tags in priority order; the first one
# present in a given filing wins. Insurers (SBILIFE, HDFCLIFE) returned "no
# filings found" entirely rather than a taxonomy mismatch and are a separate,
# still-open gap -- see TRACKER.md.
_TAGS = {
    "revenue": [
        "in-bse-fin:RevenueFromOperations",
        "in-bse-fin:InterestEarned",  # banking taxonomy
    ],
    "profit": [
        "in-bse-fin:ProfitLossForPeriod",
        "in-bse-fin:ProfitLossForThePeriod",  # banking taxonomy (note "ForThePeriod")
    ],
    "eps_basic": [
        "in-bse-fin:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        "in-bse-fin:BasicEarningsPerShareBeforeExtraordinaryItems",  # banking taxonomy
    ],
}

FUNDAMENTALS_COLUMNS = [
    "instrument_id",
    "event_time",
    "available_time",
    "source_id",
    "revision_id",
    "quality_flag",
    "missing_reason",
    "consolidated",
    "revenue",
    "profit",
    "eps_basic",
]


def _http_get(url: str, max_retries: int = 3) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.0 * attempt)
    raise RuntimeError(f"failed to fetch {url} after {max_retries} attempts") from last_exc


def fetch_financial_results_list(symbol: str, preferred_consolidation: str = "Consolidated") -> pd.DataFrame:
    """List every quarterly filing NSE has on record for one symbol, with exact
    filing timestamps and the XBRL link, filtered to one consolidation type.

    ``preferred_consolidation`` defaults to "Consolidated" -- the more
    representative figure for a diversified conglomerate and what most
    published fundamental ratios (P/E, EPS growth) are quoted against; falls
    back to "Non-Consolidated" for the (rare, single-business) issuer that
    only files that type.
    """
    # Real bug found in the full-universe backfill: NSE:M&M's symbol contains
    # "&", which breaks the query string if not encoded (it gets parsed as an
    # extra, bogus query parameter, silently truncating/corrupting the request
    # and returning zero results). url-quote every symbol defensively.
    url = _LIST_URL.format(symbol=urllib.parse.quote(symbol, safe=""))
    raw = json.loads(_http_get(url))
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    available_types = set(df["consolidated"].unique())
    consolidation = preferred_consolidation if preferred_consolidation in available_types else "Non-Consolidated"
    df = df[df["consolidated"] == consolidation].copy()

    # A majority of older records carry no real XBRL -- NOT represented as a
    # bare "-" or empty string as might be expected, but as a full URL whose
    # path ends in a lone dash (e.g. ".../corporate/xbrl/-"), confirmed
    # directly against the live API (77/130 records for a single large-cap
    # issuer). Skip anything that isn't an actual .xml link before ever
    # attempting a fetch, rather than retrying a URL that can never succeed.
    df = df[df["xbrl"].notna() & df["xbrl"].str.endswith(".xml")].copy()

    df["from_date"] = pd.to_datetime(df["fromDate"], format="%d-%b-%Y")
    df["to_date"] = pd.to_datetime(df["toDate"], format="%d-%b-%Y")
    df["filing_timestamp"] = pd.to_datetime(df["broadCastDate"], format="%d-%b-%Y %H:%M:%S")
    return df.sort_values("to_date").reset_index(drop=True)


def _select_quarterly_value(xml_text: str, tag: str) -> float | None:
    """Extract one tag's "this quarter, not cumulative" value.

    See module docstring: when a tag appears under multiple contextRefs, the
    smallest-magnitude value is taken as the single-quarter figure (empirically
    validated against an independently-plausible real number; cumulative
    year-to-date figures are always larger in magnitude for a profitable,
    growing line item).
    """
    pattern = rf"<{re.escape(tag)}[^>]*>([^<]*)</{re.escape(tag)}>"
    matches = re.findall(pattern, xml_text)
    values = []
    for m in matches:
        try:
            values.append(float(m))
        except ValueError:
            continue
    if not values:
        return None
    return min(values, key=abs)


def _select_first_matching_tag(xml_text: str, candidate_tags: list[str]) -> float | None:
    for tag in candidate_tags:
        value = _select_quarterly_value(xml_text, tag)
        if value is not None:
            return value
    return None


def parse_xbrl_facts(xbrl_url: str) -> dict[str, float | None]:
    xml_text = _http_get(xbrl_url).decode("utf-8", errors="replace")
    return {name: _select_first_matching_tag(xml_text, tags) for name, tags in _TAGS.items()}


def ingest_fundamentals_for_instrument(instrument_id: str) -> pd.DataFrame:
    """Fetch every quarterly filing for one instrument, parse its XBRL, and
    point-in-time tag the result. ``available_time`` is the filing's own exact
    broadcast timestamp -- genuine point-in-time correctness, not an assumption
    (contrast with the delivery-report and macro-series adapters, which must
    assume a publication-time cutoff because NSE does not expose one).
    """
    instrument = get_instrument(instrument_id)
    filings = fetch_financial_results_list(instrument.symbol)
    if filings.empty:
        return pd.DataFrame(columns=FUNDAMENTALS_COLUMNS)

    rows = []
    for _, filing in filings.iterrows():
        try:
            facts = parse_xbrl_facts(filing["xbrl"])
            quality, reason = QualityFlag.PASS, None
            if all(v is None for v in facts.values()):
                quality, reason = QualityFlag.FAIL, "no target tags found in XBRL"
        except Exception as exc:  # noqa: BLE001
            logger.warning("XBRL parse failed for %s %s: %s", instrument_id, filing["xbrl"], exc)
            facts = {name: None for name in _TAGS}
            quality, reason = QualityFlag.FAIL, f"XBRL fetch/parse error: {exc}"

        rows.append(
            {
                "instrument_id": instrument_id,
                "event_time": to_ist(filing["to_date"]),
                "available_time": to_ist(filing["filing_timestamp"]),
                "source_id": SOURCE_ID,
                "revision_id": "v1",
                "quality_flag": quality.value,
                "missing_reason": reason,
                "consolidated": filing["consolidated"],
                **facts,
            }
        )
        time.sleep(0.2)

    return pd.DataFrame(rows)[FUNDAMENTALS_COLUMNS]


def cross_validate_eps_against_yfinance(
    fundamentals: pd.DataFrame, instrument: Instrument, tolerance: float = 0.25
) -> pd.DataFrame:
    """Flag WARN any quarter where our extracted EPS disagrees materially with
    yfinance's independently-sourced quarterly EPS, wherever both cover the
    same period. This is the safety net for the context-selection heuristic
    documented in the module docstring: it cannot prove every quarter is
    right, but it proves the ones it CAN check, and visibly flags the rest.
    """
    import yfinance as yf

    out = fundamentals.copy()
    try:
        qis = yf.Ticker(instrument.yahoo_ticker).quarterly_income_stmt
        if qis is None or qis.empty or "Basic EPS" not in qis.index:
            out["yf_cross_check"] = "no_yfinance_data"
            return out
        yf_eps = qis.loc["Basic EPS"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance cross-check unavailable for %s: %s", instrument.instrument_id, exc)
        out["yf_cross_check"] = "yfinance_unavailable"
        return out

    out["yf_cross_check"] = "no_overlap"
    for idx, row in out.iterrows():
        period_end = pd.Timestamp(row["event_time"]).tz_localize(None).normalize()
        match = yf_eps[yf_eps.index.normalize() == period_end]
        if len(match) == 0 or pd.isna(row["eps_basic"]) or row["eps_basic"] is None:
            continue
        # Real bug found in the full-universe backfill: the date can match
        # while yfinance's own value for that date is itself NaN (that line
        # item just wasn't populated that quarter). Comparing against NaN
        # silently fails the `<= tolerance` check and was being misreported
        # as a genuine MISMATCH -- must be treated as "nothing to compare".
        if pd.isna(match.iloc[0]):
            continue
        yf_value = float(match.iloc[0])
        our_value = float(row["eps_basic"])
        if yf_value == 0:
            continue
        relative_diff = abs(our_value - yf_value) / abs(yf_value)
        if relative_diff <= tolerance:
            out.at[idx, "yf_cross_check"] = "MATCH"
        else:
            out.at[idx, "yf_cross_check"] = f"MISMATCH (ours={our_value}, yfinance={yf_value})"
            out.at[idx, "quality_flag"] = QualityFlag.WARN.value
            out.at[idx, "missing_reason"] = "EPS disagrees with yfinance beyond tolerance"
    return out
