"""Tests for hybrid retrieval and the point-in-time firewall (nifty_analyser/documents/retrieval.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd


from nifty_analyser.documents.retrieval import (
    BM25Index,
    filter_as_of,
    hybrid_retrieve,
    reciprocal_rank_fusion,
)


# ---------- BM25 ----------

def test_bm25_scores_exact_term_match_higher():
    docs = ["the quick brown fox", "a completely unrelated sentence about nothing"]
    bm25 = BM25Index.build(["d0", "d1"], docs)
    scores = bm25.score("quick fox")
    assert scores["d0"] > scores["d1"]


def test_bm25_zero_score_for_no_overlap():
    docs = ["apple banana cherry", "date fig grape"]
    bm25 = BM25Index.build(["d0", "d1"], docs)
    scores = bm25.score("zzz nonexistent term")
    assert scores["d0"] == 0.0
    assert scores["d1"] == 0.0


def test_bm25_rewards_term_frequency_but_with_diminishing_returns():
    docs = ["fox fox fox fox fox", "fox and other words here too indeed yes"]
    bm25 = BM25Index.build(["d0", "d1"], docs)
    scores = bm25.score("fox")
    assert scores["d0"] > scores["d1"]  # more occurrences -> higher score


def test_bm25_penalizes_long_documents_relative_to_short_ones():
    short_doc = "fox"
    long_doc = "fox " + " ".join(["filler"] * 200)
    bm25 = BM25Index.build(["short", "long"], [short_doc, long_doc])
    scores = bm25.score("fox")
    assert scores["short"] > scores["long"]


# ---------- Reciprocal Rank Fusion ----------

def test_rrf_top_item_appearing_in_both_lists_wins():
    list_a = ["x", "y", "z"]
    list_b = ["x", "w", "y"]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0][0] == "x"


def test_rrf_handles_disjoint_lists():
    list_a = ["a", "b"]
    list_b = ["c", "d"]
    fused = reciprocal_rank_fusion([list_a, list_b])
    ids = [f[0] for f in fused]
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_rank_order_matters_not_just_presence():
    list_a = ["a", "b", "c"]
    list_b = ["a", "b", "c"]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert [f[0] for f in fused] == ["a", "b", "c"]


# ---------- Point-in-time firewall ----------

def _doc_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": ["d0", "d1", "d2"],
            "text": ["quarterly results announcement", "board meeting notice", "quarterly results update"],
            "available_time": pd.to_datetime(
                ["2024-01-10 10:00", "2024-01-15 10:00", "2024-01-20 10:00"]
            ).tz_localize("Asia/Kolkata"),
        }
    )


def test_filter_as_of_excludes_future_documents():
    docs = _doc_frame()
    result = filter_as_of(docs, pd.Timestamp("2024-01-15 10:00", tz="Asia/Kolkata"))
    assert set(result["doc_id"]) == {"d0", "d1"}


def test_filter_as_of_boundary_is_inclusive():
    docs = _doc_frame()
    result = filter_as_of(docs, pd.Timestamp("2024-01-10 10:00", tz="Asia/Kolkata"))
    assert set(result["doc_id"]) == {"d0"}


def test_filter_as_of_before_any_document_returns_empty():
    docs = _doc_frame()
    result = filter_as_of(docs, pd.Timestamp("2024-01-01", tz="Asia/Kolkata"))
    assert result.empty


class _FakeEmbeddingCache:
    """Deterministic fake embedding: a bag-of-words vector, so hybrid_retrieve
    can be tested end-to-end without a real Ollama call.
    """

    _VOCAB = ["quarterly", "results", "announcement", "board", "meeting", "notice", "update"]

    def get(self, text: str):
        vec = np.array([1.0 if w in text.lower() else 0.0 for w in self._VOCAB])
        return vec

    def put(self, *_a, **_k):
        pass


def test_hybrid_retrieve_never_returns_a_document_after_as_of(monkeypatch):
    """The load-bearing test: even if a future document would score highest on
    both BM25 and dense similarity, it must never appear in the result.
    """
    import nifty_analyser.documents.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module, "embed_text", lambda text, cache=None: _FakeEmbeddingCache().get(text))
    monkeypatch.setattr(
        retrieval_module,
        "embed_many",
        lambda texts, cache=None: np.stack([_FakeEmbeddingCache().get(t) for t in texts]),
    )

    docs = _doc_frame()
    as_of = pd.Timestamp("2024-01-15 10:00", tz="Asia/Kolkata")  # d2 (best text match) is NOT yet available
    result = retrieval_module.hybrid_retrieve("quarterly results", docs, as_of, top_k=5)

    assert "d2" not in result["doc_id"].tolist()
    assert set(result["doc_id"]) <= {"d0", "d1"}


def test_hybrid_retrieve_empty_when_nothing_available_yet(monkeypatch):
    import nifty_analyser.documents.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module, "embed_text", lambda text, cache=None: _FakeEmbeddingCache().get(text))
    monkeypatch.setattr(
        retrieval_module,
        "embed_many",
        lambda texts, cache=None: np.stack([_FakeEmbeddingCache().get(t) for t in texts]),
    )

    docs = _doc_frame()
    result = retrieval_module.hybrid_retrieve(
        "quarterly results", docs, pd.Timestamp("2023-01-01", tz="Asia/Kolkata")
    )
    assert result.empty


def test_hybrid_retrieve_ranks_relevant_document_first(monkeypatch):
    import nifty_analyser.documents.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module, "embed_text", lambda text, cache=None: _FakeEmbeddingCache().get(text))
    monkeypatch.setattr(
        retrieval_module,
        "embed_many",
        lambda texts, cache=None: np.stack([_FakeEmbeddingCache().get(t) for t in texts]),
    )

    docs = _doc_frame()
    as_of = pd.Timestamp("2024-01-25", tz="Asia/Kolkata")  # everything available
    result = retrieval_module.hybrid_retrieve("quarterly results", docs, as_of, top_k=3)
    assert result.iloc[0]["doc_id"] in {"d0", "d2"}  # one of the two genuinely relevant docs
