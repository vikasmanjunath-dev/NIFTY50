"""Tests for chunking (nifty_analyser/documents/chunking.py)."""
from __future__ import annotations

from nifty_analyser.documents.chunking import chunk_text


def test_short_text_becomes_one_chunk():
    chunks = chunk_text("doc1", "This is a short announcement.")
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc1_c0"
    assert chunks[0].chunk_index == 0


def test_empty_text_returns_no_chunks():
    assert chunk_text("doc1", "") == []
    assert chunk_text("doc1", "   ") == []


def test_long_text_splits_into_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text("doc1", text, max_words=150, overlap_words=20)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.doc_id == "doc1"


def test_chunk_ids_are_stable_and_unique():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text("doc1", text, max_words=150, overlap_words=20)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    # re-chunking the same text must give the same IDs (stability)
    chunks2 = chunk_text("doc1", text, max_words=150, overlap_words=20)
    assert [c.chunk_id for c in chunks2] == ids


def test_chunks_overlap_by_the_configured_amount():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text("doc1", text, max_words=100, overlap_words=20)
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-20:] == second_words[:20]


def test_all_words_are_covered_no_gaps():
    words = [f"w{i}" for i in range(350)]
    text = " ".join(words)
    chunks = chunk_text("doc1", text, max_words=100, overlap_words=10)
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(words)
