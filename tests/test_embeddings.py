"""Tests for the Ollama embedding client (nifty_analyser/documents/embeddings.py)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from nifty_analyser.documents.embeddings import (
    EmbeddingCache,
    OllamaUnavailableError,
    cosine_similarity,
    embed_text,
)


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    corpus = np.array([v])
    sim = cosine_similarity(v, corpus)
    assert sim[0] == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    query = np.array([1.0, 0.0])
    corpus = np.array([[0.0, 1.0]])
    sim = cosine_similarity(query, corpus)
    assert sim[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    query = np.array([1.0, 0.0])
    corpus = np.array([[-1.0, 0.0]])
    sim = cosine_similarity(query, corpus)
    assert sim[0] == pytest.approx(-1.0)


def test_cosine_similarity_ranks_more_similar_higher():
    query = np.array([1.0, 1.0, 0.0])
    corpus = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, -1.0, 0.0]])
    sims = cosine_similarity(query, corpus)
    assert sims[0] > sims[1] > sims[2]


def test_embedding_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache = EmbeddingCache(path=cache_path)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cache.put("hello world", vec)
    cache.save()

    reloaded = EmbeddingCache(path=cache_path)
    got = reloaded.get("hello world")
    np.testing.assert_allclose(got, vec)


def test_embedding_cache_miss_returns_none(tmp_path):
    cache = EmbeddingCache(path=tmp_path / "empty_cache.json")
    assert cache.get("never seen this text") is None


def test_embed_text_uses_cache_without_network_call(monkeypatch, tmp_path):
    """If the text is already cached, embed_text must not attempt any network call."""
    import nifty_analyser.documents.embeddings as embeddings_module

    def _fail_if_called(*_a, **_k):
        raise AssertionError("network should not be called when cache hits")

    monkeypatch.setattr(embeddings_module.urllib.request, "urlopen", _fail_if_called)

    cache = EmbeddingCache(path=tmp_path / "cache.json")
    vec = np.array([1.0, 2.0], dtype=np.float32)
    cache.put("cached text", vec)

    result = embed_text("cached text", cache=cache)
    np.testing.assert_allclose(result, vec)


def test_embed_text_raises_clear_error_when_ollama_unreachable(monkeypatch):
    import urllib.error

    import nifty_analyser.documents.embeddings as embeddings_module

    def _raise_connection_error(*_a, **_k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(embeddings_module.urllib.request, "urlopen", _raise_connection_error)

    with pytest.raises(OllamaUnavailableError):
        embed_text("some text", max_retries=1)
