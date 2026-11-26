"""Structured extraction: turning documents into features (Blueprint §4.7).

"The most useful way to connect RAG to predictive ML is to convert retrieved
evidence into structured, time-stamped event features rather than feeding a
giant free-form document into a neural predictor."

Materiality classification here is grounded in NSE's own ``category`` field
(the raw "desc" from the announcements API) rather than an NLP model's guess
at importance -- deliberately: NSE's own taxonomy is itself the authoritative
signal about what kind of disclosure this is, and using it directly avoids
inventing a confidence score for something that already has a ground-truth
label. A sentiment/tone model over the free text is a further refinement
(flagged in TRACKER.md), not attempted here.

KEYWORD matching, not exact-string matching, and this is a real, measured
design correction: an initial version used an exact-match set curated from a
single company's ~6-month sample. Once the full 48-instrument, 5-year corpus
was backfilled, it turned out NSE's real category taxonomy has **171 unique
strings** for what is conceptually a much smaller set of event types --
riddled with case variants ("General Updates" / "General updates"),
pluralisation ("Loss of Share Certificates" / "Loss of share certificate"),
and granular sub-types ("Buyback", "Buyback - Open Market", "Post Buyback
Public Announcement", "Closure of Buyback" all being the same underlying
event class). An exact-match set built from one company's vocabulary
silently under-classified most of the real universe as Low. Case-insensitive
substring/keyword matching against the same conceptual event classes closes
that gap and was verified against the real, full category list before being
trusted (see TRACKER.md for the before/after coverage numbers).
"""
from __future__ import annotations

import pandas as pd

# Keyword lists checked in order (High first) against the lower-cased
# category string. A category matching ANY keyword in a tier is classified
# at that tier; first match wins, so list a keyword only once across tiers.
HIGH_MATERIALITY_KEYWORDS = [
    "financial result", "outcome of board meeting", "acquisition", "amalgamation",
    "merger", "buyback", "bonus", "stock split", "split of shares", "dividend",
    "credit rating", "change in management", "change in director", "resignation",
    "demise", "rights issue", "preferential issue", "preferential placement",
    "qualified institutional placement", "scheme of arrangement", "demerger",
    "delisting", "fund raising", "utilisation of funds", "restructuring",
    "disinvestment", "insolvency", "liquidation", "capacity addition",
    "product launch", "commencement of commercial production",
]
MEDIUM_MATERIALITY_KEYWORDS = [
    "board meeting", "investor presentation", "analyst", "institutional investor meet",
    "annual general meeting", "extra ordinary meeting", "postal ballot",
    "related party transaction", "monthly business update", "appointment",
    "re-appointment", "cessation", "auditor", "litigation", "dispute",
    "material event", "material issue", "agreement", "arrangement",
]
# Everything else (e.g. "Loss of Share Certificates", "Newspaper
# Advertisements", "Certificate under SEBI (Depositories...)") defaults to Low.


def classify_materiality(category: str) -> str:
    lowered = category.lower()
    if any(keyword in lowered for keyword in HIGH_MATERIALITY_KEYWORDS):
        return "High"
    if any(keyword in lowered for keyword in MEDIUM_MATERIALITY_KEYWORDS):
        return "Medium"
    return "Low"


def add_materiality(documents: pd.DataFrame) -> pd.DataFrame:
    out = documents.copy()
    out["materiality"] = out["category"].map(classify_materiality)
    return out


def compute_daily_event_features(documents_with_materiality: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-instrument, per-day event counts and a "days since last
    high-materiality event" novelty feature.

    One row per (instrument_id, event_date) that had at least one document;
    days with zero announcements are absent (the merge into the main feature
    store, ``features/events.py``, is responsible for filling those days'
    counts with 0 rather than NaN, since "no announcement today" is known
    information, not missing information).
    """
    df = documents_with_materiality.copy()
    # event_date is a calendar-day identifier, not a moment in time -- kept
    # deliberately tz-naive (a recurring bug class this session: comparing a
    # tz-aware normalized timestamp against a plain tz-naive pd.Timestamp
    # silently fails rather than raising, exactly as it did in bhavcopy.py).
    df["event_date"] = pd.to_datetime(df["event_time"]).dt.tz_localize(None).dt.normalize()

    parts = []
    for instrument_id, g in df.groupby("instrument_id", sort=False):
        daily = (
            g.groupby("event_date")
            .agg(
                announcement_count=("doc_id", "count"),
                high_materiality_count=("materiality", lambda s: (s == "High").sum()),
                medium_materiality_count=("materiality", lambda s: (s == "Medium").sum()),
                available_time=("available_time", "max"),
            )
            .reset_index()
            .sort_values("event_date")
        )
        daily["instrument_id"] = instrument_id

        last_high_event_date = pd.NaT
        days_since_high = []
        for _, row in daily.iterrows():
            if pd.notna(last_high_event_date):
                days_since_high.append((row["event_date"] - last_high_event_date).days)
            else:
                days_since_high.append(pd.NA)
            if row["high_materiality_count"] > 0:
                last_high_event_date = row["event_date"]
        daily["days_since_high_materiality_event"] = days_since_high

        parts.append(daily)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=[
            "event_date", "announcement_count", "high_materiality_count",
            "medium_materiality_count", "available_time", "instrument_id",
            "days_since_high_materiality_event",
        ]
    )
