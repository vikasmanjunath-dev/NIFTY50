"""Hybrid retrieval with a point-in-time firewall (Blueprint §4.4, §10).

Two things are non-negotiable per the blueprint, and this module exists
specifically to enforce both mechanically rather than by convention:

1. **The point-in-time firewall (§4.4)**: "for a backtest at time t, the
   retrieval system must behave as though it lived at time t." Every
   retrieval call takes an explicit ``as_of`` timestamp and filters the
   candidate pool to ``available_time <= as_of`` *before* any scoring
   happens -- never after. ``retrieve()`` cannot accidentally see a future
   document because it is structurally incapable of scoring one that was
   filtered out first.
2. **Hybrid retrieval (§4.6)**: lexical (BM25, hand-rolled here -- the
   project's established style of implementing auditable formulas directly
   rather than adding a dependency for something this contained) finds exact
   terminology; dense (Ollama embeddings) bridges paraphrases; Reciprocal
   Rank Fusion combines the two candidate sets without needing to calibrate
   their very different raw score scales against each other.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nifty_analyser.documents.embeddings import EmbeddingCache, cosine_similarity, embed_many, embed_text


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.split() if t.isalnum() or any(c.isalnum() for c in t)]


@dataclass
class BM25Index:
    """A minimal, from-scratch BM25 implementation (Okapi BM25).

    Kept deliberately small and auditable: this is the same "hand-implement
    rather than add a dependency" choice made for technical indicators in
    Part 2 of the blueprint's own build.
    """

    doc_ids: list[str]
    doc_term_freqs: list[Counter]
    doc_lengths: list[int]
    avg_doc_length: float
    doc_freq: dict[str, int]
    n_docs: int
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, doc_ids: list[str], texts: list[str]) -> "BM25Index":
        doc_term_freqs = [Counter(_tokenize(t)) for t in texts]
        doc_lengths = [sum(tf.values()) for tf in doc_term_freqs]
        avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

        doc_freq: dict[str, int] = defaultdict(int)
        for tf in doc_term_freqs:
            for term in tf:
                doc_freq[term] += 1

        return cls(
            doc_ids=doc_ids,
            doc_term_freqs=doc_term_freqs,
            doc_lengths=doc_lengths,
            avg_doc_length=avg_len,
            doc_freq=dict(doc_freq),
            n_docs=len(doc_ids),
        )

    def _idf(self, term: str) -> float:
        n_containing = self.doc_freq.get(term, 0)
        # +1 smoothing keeps IDF finite (and non-negative under BM25's usual
        # convention) even for a term that appears in every document.
        return math.log((self.n_docs - n_containing + 0.5) / (n_containing + 0.5) + 1.0)

    def score(self, query: str, candidate_indices: list[int] | None = None) -> dict[str, float]:
        query_terms = _tokenize(query)
        indices = candidate_indices if candidate_indices is not None else range(self.n_docs)
        scores: dict[str, float] = {}
        for i in indices:
            tf = self.doc_term_freqs[i]
            doc_len = self.doc_lengths[i]
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                idf = self._idf(term)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_length or 1.0))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores[self.doc_ids[i]] = score
        return scores


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists of doc/chunk IDs into one fused ranking.

    RRF score for an item = sum over lists of 1/(k + rank). Chosen (per the
    blueprint's own recommendation, §10) specifically because it needs no
    calibration between BM25's unbounded scores and cosine similarity's
    [-1, 1] range -- only each list's *rank order* matters.
    """
    scores: dict[str, float] = defaultdict(float)
    weights = weights if weights is not None else [1.0] * len(ranked_lists)          # default = classic unweighted RRF
    for w, ranked_list in zip(weights, ranked_lists):
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] += w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def filter_as_of(documents: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """THE point-in-time firewall: keep only documents knowable at ``as_of``.

    This must run before any retrieval scoring, not after -- filtering after
    scoring would still leak information indirectly (e.g. through a
    fitted/cached index built over future documents, or IDF statistics
    computed including them). ``hybrid_retrieve`` below always calls this
    first for that reason.
    """
    return documents[documents["available_time"] <= as_of].copy()


def hybrid_retrieve(
    query: str,
    documents: pd.DataFrame,
    as_of: pd.Timestamp,
    top_k: int = 8,
    candidate_k: int = 50,
    embedding_cache: EmbeddingCache | None = None,
) -> pd.DataFrame:
    """Point-in-time-safe hybrid retrieval: filter -> BM25 + dense -> RRF fuse -> top_k.

    ``documents`` must have columns: doc_id, text, available_time (plus
    whatever metadata the caller wants carried through to the result).
    """
    eligible = filter_as_of(documents, as_of)
    if eligible.empty:
        return eligible.assign(retrieval_score=pd.Series(dtype=float))

    doc_ids = eligible["doc_id"].tolist()
    texts = eligible["text"].tolist()

    bm25 = BM25Index.build(doc_ids, texts)
    bm25_scores = bm25.score(query)
    bm25_ranked = [doc_id for doc_id, _ in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_k]]

    query_vec = embed_text(query, cache=embedding_cache)
    doc_vecs = embed_many(texts, cache=embedding_cache)
    dense_scores = cosine_similarity(query_vec, doc_vecs)
    dense_ranked_idx = np.argsort(-dense_scores)[:candidate_k]
    dense_ranked = [doc_ids[i] for i in dense_ranked_idx]

    fused = reciprocal_rank_fusion([bm25_ranked, dense_ranked])[:top_k]
    fused_ids = [doc_id for doc_id, _ in fused]
    fused_scores = dict(fused)

    result = eligible[eligible["doc_id"].isin(fused_ids)].copy()
    result["retrieval_score"] = result["doc_id"].map(fused_scores)
    return result.sort_values("retrieval_score", ascending=False).reset_index(drop=True)
