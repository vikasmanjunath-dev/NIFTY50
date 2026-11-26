"""Retrieval evaluation (Blueprint §4.10, §27).

"Do not report 'RAG accuracy' as a single number. Report which layer
improved, on which time window, with what cost." This module implements the
standard retrieval-quality metrics the blueprint names explicitly
(Recall@k, Precision@k, MRR), each computed against a *relevance* definition
supplied by the caller -- deliberately not baked in, since what counts as
"relevant" depends on the query.

For this project's evaluation (see ``scripts/run_phase3_pipeline.py``),
relevance is defined operationally and auditably: a document is relevant to
an instrument-targeted query if it belongs to that instrument. This avoids
inventing subjective relevance judgments while still giving a real,
falsifiable measure of whether hybrid retrieval finds the right company's
documents instead of a different one's -- a meaningful, if narrow, slice of
what "relevant" means here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    n_relevant_total: int
    n_retrieved: int


def evaluate_ranking(retrieved_ids: list[str], relevant_ids: set[str], k: int | None = None) -> RetrievalMetrics:
    """Compute Recall@k, Precision@k and reciprocal rank for one query's ranked results.

    ``retrieved_ids`` must already be in rank order (best first). ``k``
    defaults to ``len(retrieved_ids)`` (i.e. evaluate the whole given list).
    """
    if k is None:
        k = len(retrieved_ids)
    top_k = retrieved_ids[:k]

    n_relevant_total = len(relevant_ids)
    n_relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)

    recall = n_relevant_retrieved / n_relevant_total if n_relevant_total > 0 else float("nan")
    precision = n_relevant_retrieved / len(top_k) if top_k else 0.0

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    return RetrievalMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=reciprocal_rank,
        n_relevant_total=n_relevant_total,
        n_retrieved=len(top_k),
    )


def mean_reciprocal_rank(per_query_metrics: list[RetrievalMetrics]) -> float:
    if not per_query_metrics:
        return float("nan")
    return sum(m.reciprocal_rank for m in per_query_metrics) / len(per_query_metrics)


def mean_recall_at_k(per_query_metrics: list[RetrievalMetrics]) -> float:
    valid = [m.recall_at_k for m in per_query_metrics if m.recall_at_k == m.recall_at_k]  # drop NaN
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-gain nDCG@k; NaN when the query has no relevant documents."""
    import math
    if not relevant_ids:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved_ids[:k]) if d in relevant_ids)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / ideal
