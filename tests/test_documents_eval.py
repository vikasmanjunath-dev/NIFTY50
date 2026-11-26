"""Tests for retrieval evaluation metrics (nifty_analyser/documents/eval.py)."""
from __future__ import annotations

import pytest

from nifty_analyser.documents.eval import (
    RetrievalMetrics,
    evaluate_ranking,
    mean_recall_at_k,
    mean_reciprocal_rank,
)


def test_perfect_ranking_scores_perfectly():
    m = evaluate_ranking(["a", "b", "c"], {"a", "b", "c"})
    assert m.recall_at_k == pytest.approx(1.0)
    assert m.precision_at_k == pytest.approx(1.0)
    assert m.reciprocal_rank == pytest.approx(1.0)


def test_relevant_item_first_gives_reciprocal_rank_one():
    m = evaluate_ranking(["a", "x", "y"], {"a"})
    assert m.reciprocal_rank == pytest.approx(1.0)


def test_relevant_item_third_gives_reciprocal_rank_one_third():
    m = evaluate_ranking(["x", "y", "a"], {"a"})
    assert m.reciprocal_rank == pytest.approx(1.0 / 3.0)


def test_no_relevant_item_retrieved_gives_zero_reciprocal_rank():
    m = evaluate_ranking(["x", "y", "z"], {"a"})
    assert m.reciprocal_rank == 0.0
    assert m.recall_at_k == 0.0


def test_recall_at_k_partial():
    m = evaluate_ranking(["a", "x", "y"], {"a", "b"})  # only 1 of 2 relevant retrieved
    assert m.recall_at_k == pytest.approx(0.5)


def test_precision_at_k_respects_k_truncation():
    m = evaluate_ranking(["a", "x", "y", "z"], {"a"}, k=2)
    assert m.n_retrieved == 2
    assert m.precision_at_k == pytest.approx(0.5)  # 1 relevant out of top-2


def test_no_relevant_docs_at_all_gives_nan_recall():
    m = evaluate_ranking(["a", "b"], set())
    assert m.recall_at_k != m.recall_at_k  # NaN


def test_empty_retrieved_list_gives_zero_precision_not_crash():
    m = evaluate_ranking([], {"a"})
    assert m.precision_at_k == 0.0
    assert m.reciprocal_rank == 0.0


def test_mean_reciprocal_rank_averages_across_queries():
    metrics = [
        evaluate_ranking(["a"], {"a"}),  # RR = 1
        evaluate_ranking(["x", "a"], {"a"}),  # RR = 0.5
    ]
    assert mean_reciprocal_rank(metrics) == pytest.approx(0.75)


def test_mean_recall_at_k_ignores_nan_queries():
    metrics = [
        evaluate_ranking(["a"], {"a"}),  # recall = 1
        evaluate_ranking(["x"], set()),  # recall = NaN, must be excluded from the mean
    ]
    assert mean_recall_at_k(metrics) == pytest.approx(1.0)


def test_mean_reciprocal_rank_empty_list_is_nan():
    assert mean_reciprocal_rank([]) != mean_reciprocal_rank([])
