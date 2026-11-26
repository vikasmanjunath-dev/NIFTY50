"""Chunking (Blueprint §4.5, §8 micro-detail).

NSE announcement text (via ``attchmntText``) is typically already a short,
self-contained summary -- a handful of sentences, not a multi-page filing --
so most documents become exactly one chunk. This module still implements
general word-count-bounded chunking with a stable per-chunk ID, both because
some announcements *are* long (board resolutions, detailed disclosures), and
because a future extension to parsing full attachment PDFs (out of scope
here, see TRACKER.md) would need it immediately.

Chunk IDs are stable for a given (doc_id, chunk_index) pair, matching the
blueprint's requirement (§8) that "chunk-level IDs remain stable for the same
source version" -- needed so the evidence pack in evidence.py can cite a
chunk and have that citation still resolve correctly later.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str


def chunk_text(doc_id: str, text: str, max_words: int = 150, overlap_words: int = 20) -> list[Chunk]:
    """Split ``text`` into word-bounded chunks with a small overlap.

    Splits on whitespace-delimited words rather than sentences, deliberately
    simple: NSE announcement text does not reliably use consistent sentence
    punctuation (see real examples with run-on clauses), so a robust sentence
    splitter would add complexity without a clear benefit at this corpus's
    typical length.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= max_words:
        return [Chunk(chunk_id=f"{doc_id}_c0", doc_id=doc_id, chunk_index=0, text=text.strip())]

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = max_words - overlap_words
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text_value = " ".join(words[start:end])
        chunks.append(Chunk(chunk_id=f"{doc_id}_c{index}", doc_id=doc_id, chunk_index=index, text=chunk_text_value))
        if end == len(words):
            break
        start += step
        index += 1
    return chunks
