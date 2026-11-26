"""Trained cross-encoder rerankers (Blueprint §4.6) behind the same scorer interface as ``rerank.ollama_scorer``.
Models are downloaded once from Hugging Face (safetensors weights) and run locally; scores are sigmoid(logit)*10 so they drop into
``rerank.rerank`` (LLM-score slot, 0-10) unchanged. Heavy imports are deferred so the package imports without torch."""
from __future__ import annotations

from typing import Callable

import numpy as np

MODELS = {"minilm": "cross-encoder/ms-marco-MiniLM-L6-v2",       # ~90 MB, fast
          "bge": "BAAI/bge-reranker-v2-m3"}                       # ~2.3 GB, multilingual, strongest of the two


def sigmoid10(logits) -> list[float]:
    return (10.0 / (1.0 + np.exp(-np.asarray(logits, float)))).tolist()


def cross_encoder_scorer(name: str = "bge", max_length: int = 256, batch_size: int = 16) -> Callable[[str, list[str]], list[float]]:
    from sentence_transformers import CrossEncoder
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CrossEncoder(MODELS.get(name, name), max_length=max_length, device=device)

    def score(query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return sigmoid10(model.predict([(query, t) for t in texts], batch_size=batch_size, show_progress_bar=False))
    return score
