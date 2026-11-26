"""Evidence packs: the contract between retrieval and generation (Blueprint §4.6, §14).

"Every evidence item must be independently addressable... keep the original
text span so a human can audit the claim... keep the document version hash so
a later correction cannot silently alter the basis of an old decision."

This is deliberately a plain, serialisable structure (not a class with
behaviour) -- it is meant to be logged verbatim as part of the audit trail
(Blueprint §32), so the LLM layer (``llm.py``) and any future consumer only
ever sees this contract, never a raw retrieval result.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    doc_id: str
    instrument_id: str
    source_id: str
    publication_time: str  # ISO string -- evidence packs are meant to be JSON-serialised
    category: str
    claim_span: str
    relevance_score: float
    content_hash: str
    rerank_llm: float | None = None   # 0-10 LLM relevance of this item to the query when a reranker ran (None otherwise)


@dataclass
class EvidencePack:
    query: str
    as_of_time: str
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    coverage_score: float = 0.0
    temporal_valid: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def is_empty(self) -> bool:
        return len(self.evidence_items) == 0


def _content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_evidence_pack(
    query: str, as_of: pd.Timestamp, retrieved: pd.DataFrame, max_items: int = 8
) -> EvidencePack:
    """Turn a hybrid-retrieval result frame into a structured, citable evidence pack.

    ``retrieved`` is expected to already be point-in-time filtered (i.e. the
    output of ``retrieval.hybrid_retrieve``, never a raw, unfiltered frame) --
    ``temporal_valid`` below is a defensive re-check, not the primary control.
    """
    items = []
    temporal_valid = True
    for _, row in retrieved.head(max_items).iterrows():
        publication_time = pd.Timestamp(row["available_time"])
        if publication_time > as_of:
            temporal_valid = False  # should be unreachable if the caller filtered correctly
            continue
        items.append(
            EvidenceItem(
                evidence_id=f"EV_{row['doc_id']}",
                doc_id=row["doc_id"],
                instrument_id=row.get("instrument_id", ""),
                source_id=row.get("source_id", "unknown"),
                publication_time=publication_time.isoformat(),
                category=row.get("category", "Uncategorised"),
                claim_span=row["text"],
                relevance_score=float(row.get("retrieval_score", 0.0)),
                content_hash=_content_hash(row["text"]),
                rerank_llm=float(row["rerank_llm"]) if "rerank_llm" in row and pd.notna(row["rerank_llm"]) else None,
            )
        )

    coverage = len(items) / max_items if max_items > 0 else 0.0
    return EvidencePack(
        query=query,
        as_of_time=as_of.isoformat(),
        evidence_items=items,
        coverage_score=coverage,
        temporal_valid=temporal_valid,
    )
