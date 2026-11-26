"""Deterministic evidence digest: an answer composed ONLY of what the retrieved announcements state (dates, categories, quoted text), used when the
LLM refuses although the reranker rates the evidence highly relevant. Faithful by construction (every word comes from the evidence), so it carries no
hallucination risk; the gate (reranker score) prevents answering on irrelevant evidence. Always labelled ``answer_source = digest`` by the service."""
from __future__ import annotations

import re

from nifty_analyser.documents.evidence import EvidencePack

DIGEST_MIN_RERANK = 7.0       # chosen on half of the explain-study questions, confirmed on the other half (reports/digest_eval.json)


def best_rerank(pack: EvidencePack) -> float | None:
    s = [i.rerank_llm for i in pack.evidence_items if i.rerank_llm is not None]
    return max(s) if s else None


def build_digest(pack: EvidencePack, max_items: int = 4, snippet_chars: int = 150) -> str:
    items = sorted(pack.evidence_items, key=lambda i: i.publication_time, reverse=True)[:max_items]
    if not items:
        return ""
    cats = sorted({i.category for i in items})
    lines = [f"The retrieved announcements ({', '.join(cats)}) establish the following, newest first:"]
    for n, i in enumerate(items, 1):
        text = re.sub(r"\s+", " ", i.claim_span).strip()
        lines.append(f"[{n}] {i.publication_time[:10]} ({i.category}): {text[:snippet_chars]}{'…' if len(text) > snippet_chars else ''}")
    lines.append("No further details beyond these filings are available in the evidence.")
    return "\n".join(lines)
