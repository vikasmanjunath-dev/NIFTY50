"""Tests for the point-in-time document contract (nifty_analyser/documents/schemas.py)."""
from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from nifty_analyser.data.schemas import QualityFlag
from nifty_analyser.documents.schemas import DocumentRecord


def _doc(**overrides) -> dict:
    base = dict(
        instrument_id="NSE:RELIANCE",
        doc_id="NSE_ANN_123",
        event_time=pd.Timestamp("2026-09-17 19:50:37"),
        available_time=pd.Timestamp("2026-09-17 19:50:38"),
        source_id="nse_corporate_announcements",
        category="Updates",
        text="Some announcement text.",
    )
    base.update(overrides)
    return base


def test_valid_document_constructs():
    doc = DocumentRecord(**_doc())
    assert doc.doc_id == "NSE_ANN_123"
    assert doc.quality_flag == QualityFlag.PASS


def test_blank_doc_id_rejected():
    with pytest.raises(ValidationError):
        DocumentRecord(**_doc(doc_id="  "))


def test_available_before_event_rejected():
    with pytest.raises(ValidationError):
        DocumentRecord(
            **_doc(
                event_time=pd.Timestamp("2026-09-17 19:50:38"),
                available_time=pd.Timestamp("2026-09-17 19:50:00"),
            )
        )


def test_pass_quality_requires_nonempty_text():
    with pytest.raises(ValidationError):
        DocumentRecord(**_doc(text=""))


def test_fail_quality_allows_empty_text_with_reason():
    doc = DocumentRecord(**_doc(text="", quality_flag=QualityFlag.FAIL, missing_reason="no content"))
    assert doc.quality_flag == QualityFlag.FAIL


def test_document_is_immutable():
    doc = DocumentRecord(**_doc())
    with pytest.raises(ValidationError):
        doc.text = "changed"
