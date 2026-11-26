"""Dense embeddings via a local Ollama server (Blueprint §30 local deployment).

Confirmed directly, before writing this module, that Ollama is running
locally with ``mxbai-embed-large`` already pulled (1024-dim). This is used
in preference to a hosted embedding API for exactly the reason the blueprint
gives for local deployment: privacy, cost and deterministic infrastructure --
and in preference to installing `sentence-transformers`/`torch` in this
environment, since two earlier real dependency incidents this session
(scipy/numpy ABI conflicts) argued for not adding another heavy, versioned ML
dependency stack when a working local alternative already exists.

Embeddings are cached on disk by a hash of the input text, because the same
announcement text is looked up repeatedly (once per retrieval query it's a
candidate for) and re-embedding is the most expensive step in the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.request

import numpy as np

from nifty_analyser.config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "mxbai-embed-large"
EMBEDDING_DIM = 1024

_CACHE_PATH = DATA_PROCESSED_DIR / "embedding_cache.json"


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server cannot be reached at all."""


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """A simple on-disk JSON cache: text-hash -> embedding vector.

    JSON (not Parquet/pickle) deliberately, so the cache is human-inspectable
    and diffable -- it is small (one 1024-float list per unique text) and
    correctness here matters more than raw load speed.
    """

    def __init__(self, path=_CACHE_PATH):
        self.path = path
        self._data: dict[str, list[float]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def get(self, text: str) -> np.ndarray | None:
        vec = self._data.get(_text_hash(text))
        return np.array(vec, dtype=np.float32) if vec is not None else None

    def put(self, text: str, embedding: np.ndarray) -> None:
        self._data[_text_hash(text)] = embedding.tolist()

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data))

    def __len__(self) -> int:
        return len(self._data)


def embed_text(text: str, cache: EmbeddingCache | None = None, max_retries: int = 2) -> np.ndarray:
    """Embed one piece of text via Ollama, using the cache when available."""
    if cache is not None:
        cached = cache.get(text)
        if cached is not None:
            return cached

    payload = json.dumps({"model": EMBEDDING_MODEL, "prompt": text}).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            embedding = np.array(result["embedding"], dtype=np.float32)
            if cache is not None:
                cache.put(text, embedding)
            return embedding
        except urllib.error.URLError as exc:
            last_exc = exc
            time.sleep(0.5 * attempt)
    raise OllamaUnavailableError(
        f"could not reach Ollama at {OLLAMA_URL} after {max_retries} attempts: {last_exc}"
    ) from last_exc


def embed_many(texts: list[str], cache: EmbeddingCache | None = None) -> np.ndarray:
    """Embed a list of texts, returning an (n, EMBEDDING_DIM) array."""
    return np.stack([embed_text(t, cache=cache) for t in texts])


def cosine_similarity(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and every row of a corpus matrix."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    corpus_norms = corpus_matrix / (np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-10)
    return corpus_norms @ query_norm
