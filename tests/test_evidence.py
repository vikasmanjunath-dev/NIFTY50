"""Tests for evidence pack construction (nifty_analyser/documents/evidence.py)."""
from __future__ import annotations

import pandas as pd


from nifty_analyser.documents.evidence import build_evidence_pack


def _retrieved_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": ["d0", "d1"],
            "instrument_id": ["NSE:RELIANCE", "NSE:RELIANCE"],
            "source_id": ["nse_corporate_announcements", "nse_corporate_announcements"],
            "available_time": pd.to_datetime(["2024-01-10 10:00", "2024-01-12 11:00"]).tz_localize(
                "Asia/Kolkata"
            ),
            "category": ["Updates", "Acquisition"],
            "text": ["Some announcement text.", "Another announcement text."],
            "retrieval_score": [0.9, 0.5],
        }
    )


def test_evidence_pack_has_one_item_per_row():
    as_of = pd.Timestamp("2024-01-15", tz="Asia/Kolkata")
    pack = build_evidence_pack("query", as_of, _retrieved_frame(), max_items=8)
    assert len(pack.evidence_items) == 2
    assert not pack.is_empty


def test_evidence_pack_respects_max_items():
    as_of = pd.Timestamp("2024-01-15", tz="Asia/Kolkata")
    pack = build_evidence_pack("query", as_of, _retrieved_frame(), max_items=1)
    assert len(pack.evidence_items) == 1


def test_evidence_pack_empty_input_gives_empty_pack():
    as_of = pd.Timestamp("2024-01-15", tz="Asia/Kolkata")
    empty = _retrieved_frame().iloc[0:0]
    pack = build_evidence_pack("query", as_of, empty, max_items=8)
    assert pack.is_empty
    assert pack.coverage_score == 0.0


def test_evidence_ids_are_unique_and_traceable_to_doc_id():
    as_of = pd.Timestamp("2024-01-15", tz="Asia/Kolkata")
    pack = build_evidence_pack("query", as_of, _retrieved_frame(), max_items=8)
    ids = [item.evidence_id for item in pack.evidence_items]
    assert len(ids) == len(set(ids))
    assert all(item.doc_id in item.evidence_id for item in pack.evidence_items)


def test_content_hash_is_deterministic_and_content_sensitive():
    as_of = pd.Timestamp("2024-01-15", tz="Asia/Kolkata")
    pack1 = build_evidence_pack("query", as_of, _retrieved_frame(), max_items=8)
    pack2 = build_evidence_pack("query", as_of, _retrieved_frame(), max_items=8)
    assert pack1.evidence_items[0].content_hash == pack2.evidence_items[0].content_hash
    assert pack1.evidence_items[0].content_hash != pack1.evidence_items[1].content_hash


def test_defensive_temporal_check_flags_a_smuggled_future_item():
    """Even though hybrid_retrieve is supposed to filter first, build_evidence_pack
    has its own defensive re-check -- prove it actually catches a violation.
    """
    docs = _retrieved_frame()
    docs.loc[1, "available_time"] = pd.Timestamp("2024-06-01", tz="Asia/Kolkata")  # smuggle a future doc
    as_of = pd.Timestamp("2024-01-15", tz="Asia/Kolkata")

    pack = build_evidence_pack("query", as_of, docs, max_items=8)
    assert pack.temporal_valid is False
    assert len(pack.evidence_items) == 1  # the future item is excluded from the pack itself


def test_evidence_pack_serializes_to_dict_cleanly():
    as_of = pd.Timestamp("2024-01-15", tz="Asia/Kolkata")
    pack = build_evidence_pack("query", as_of, _retrieved_frame(), max_items=8)
    d = pack.to_dict()
    assert d["query"] == "query"
    assert isinstance(d["evidence_items"], list)
    assert "content_hash" in d["evidence_items"][0]
