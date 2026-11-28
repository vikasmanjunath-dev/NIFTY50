"""Tests for structured event extraction (nifty_analyser/documents/extraction.py)."""
from __future__ import annotations

import pandas as pd


from nifty_analyser.documents.extraction import (
    add_materiality,
    classify_materiality,
    compute_daily_event_features,
)


def test_classify_materiality_high():
    assert classify_materiality("Financial Results") == "High"
    assert classify_materiality("Acquisition") == "High"


def test_classify_materiality_medium():
    assert classify_materiality("Board Meeting") == "Medium"


def test_classify_materiality_defaults_to_low():
    assert classify_materiality("Loss of Share Certificates") == "Low"
    assert classify_materiality("Some Totally Unknown Category") == "Low"


def test_classify_materiality_handles_real_taxonomy_variants():
    """Regression test for a real design correction: an initial exact-match
    version, built from a single company's narrow sample, silently
    misclassified most of NSE's real 171-variant category taxonomy as Low.
    These are actual strings observed in the full 48-instrument backfill.
    """
    # Case/pluralisation variants of the same underlying event, both must
    # classify identically (neither should silently fall through to Low).
    assert classify_materiality("Loss of Share Certificates") == classify_materiality("Loss of share certificate")
    assert classify_materiality("General Updates") == classify_materiality("General updates")

    # Granular sub-types of the same conceptual High-materiality event.
    for variant in ["Buyback", "Buyback - Open Market", "Post Buyback Public Announcement",
                    "Closure of Buyback", "Public Announcement - Buyback of Shares"]:
        assert classify_materiality(variant) == "High", f"{variant!r} should be High"

    # A real, specific miss from the original exact-match version: the
    # "outcome" of a board meeting (decisions actually made) is meaningfully
    # different from a mere meeting notice, but both are clearly board-related.
    assert classify_materiality("Outcome of Board Meeting") == "High"
    assert classify_materiality("Board Meeting Intimation") == "Medium"

    # Genuinely routine/administrative categories should still default Low.
    assert classify_materiality("Certificate under SEBI (Depositories and Participants) Regulations, 2018") == "Low"
    assert classify_materiality("Newspaper Advertisements") == "Low"
    assert classify_materiality("Copy of Newspaper Publication") == "Low"


def _docs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": ["NSE:AAA"] * 5,
            "doc_id": ["d0", "d1", "d2", "d3", "d4"],
            "event_time": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-15", "2024-02-01", "2024-02-01"]
            ).tz_localize("Asia/Kolkata"),
            "available_time": pd.to_datetime(
                ["2024-01-01 10:05", "2024-01-01 15:05", "2024-01-15 10:05", "2024-02-01 10:05", "2024-02-01 10:10"]
            ).tz_localize("Asia/Kolkata"),
            "category": ["Financial Results", "Loss of Share Certificates", "Board Meeting", "Acquisition", "Updates"],
        }
    )


def test_add_materiality_column():
    out = add_materiality(_docs())
    assert out.loc[out.doc_id == "d0", "materiality"].iloc[0] == "High"
    assert out.loc[out.doc_id == "d1", "materiality"].iloc[0] == "Low"


def test_daily_aggregation_counts_correctly():
    docs = add_materiality(_docs())
    daily = compute_daily_event_features(docs)
    jan1 = daily[daily.event_date == pd.Timestamp("2024-01-01")]
    assert jan1["announcement_count"].iloc[0] == 2
    assert jan1["high_materiality_count"].iloc[0] == 1


def test_days_since_high_materiality_event_is_na_before_first_high_event():
    docs = pd.DataFrame(
        {
            "instrument_id": ["NSE:AAA"],
            "doc_id": ["d0"],
            "event_time": pd.to_datetime(["2024-01-01"]).tz_localize("Asia/Kolkata"),
            "available_time": pd.to_datetime(["2024-01-01 10:00"]).tz_localize("Asia/Kolkata"),
            "category": ["Board Meeting"],  # medium, not high
        }
    )
    docs = add_materiality(docs)
    daily = compute_daily_event_features(docs)
    assert pd.isna(daily["days_since_high_materiality_event"].iloc[0])


def test_days_since_high_materiality_event_counts_correctly():
    docs = add_materiality(_docs())
    daily = compute_daily_event_features(docs).sort_values("event_date").reset_index(drop=True)
    # Jan 1 has a High event (Financial Results); Jan 15 is 14 days later
    jan15 = daily[daily.event_date == pd.Timestamp("2024-01-15")]
    assert jan15["days_since_high_materiality_event"].iloc[0] == 14
    # Feb 1 is 17 days after Jan 15's last-high-event-date (still Jan 1, since
    # Jan 15 was only Medium) -> 31 days since Jan 1
    feb1 = daily[daily.event_date == pd.Timestamp("2024-02-01")]
    assert feb1["days_since_high_materiality_event"].iloc[0] == 31


def test_multiple_instruments_are_independent():
    docs = pd.concat(
        [
            add_materiality(_docs()),
            add_materiality(_docs()).assign(instrument_id="NSE:BBB"),
        ],
        ignore_index=True,
    )
    daily = compute_daily_event_features(docs)
    assert set(daily["instrument_id"]) == {"NSE:AAA", "NSE:BBB"}
    a_counts = daily[daily.instrument_id == "NSE:AAA"]["announcement_count"].sum()
    b_counts = daily[daily.instrument_id == "NSE:BBB"]["announcement_count"].sum()
    assert a_counts == b_counts == 5


def test_empty_input_returns_empty_frame_not_crash():
    empty = add_materiality(_docs()).iloc[0:0]
    daily = compute_daily_event_features(empty)
    assert daily.empty
