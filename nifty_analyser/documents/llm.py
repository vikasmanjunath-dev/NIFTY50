"""LLM explanation layer via a local Ollama model (Blueprint §4.9, §11 of v2).

"The LLM should never be the numerical source of truth. It should explain
and orchestrate evidence produced by deterministic analytics and live data
tools." This module's job is narrow on purpose: given an already-built
``EvidencePack`` (never a raw document, never a raw question with no
evidence attached), ask a local model to summarise *only* what's in that
pack, then hand the raw response to ``guardrails.py`` for verification before
anything is treated as trustworthy output.

The model itself is never trusted to self-report groundedness -- that would
just move the hallucination risk one level up. Grounding is checked
mechanically in ``guardrails.py`` by testing whether the model's output text
actually overlaps the evidence it was given.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from nifty_analyser.documents.embeddings import OllamaUnavailableError
from nifty_analyser.documents.evidence import EvidencePack

logger = logging.getLogger(__name__)

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"  # reports/explain_eval*.json: answers 50% of answerable questions vs 29% for llama3.2:3b, no unfaithful answers released under the judge

_SYSTEM_PROMPT = (
    "You are a financial research assistant. You will be given a question and a "
    "numbered list of evidence items, each from a dated, sourced NSE corporate "
    "announcement. Rules, which you must follow exactly:\n"
    "1. Answer using ONLY the evidence provided. Never use outside knowledge.\n"
    "2. Every factual claim in your answer must be traceable to a specific evidence "
    "item number.\n"
    "3. If the evidence does not contain enough information to answer, say exactly: "
    "\"Insufficient evidence to answer.\" Do not guess or speculate. BUT if the evidence shows that announcements on the question's topic exist, "
    "answer with what they establish (their dates and categories, the facts they state) and say what is not stated; reserve the exact sentence "
    "for the case where NO item relates to the question.\n"
    "4. Do not invent prices, dates, or figures not present in the evidence.\n"
    "5. Keep the answer under 150 words.\n"
    "6. Items are listed NEWEST FIRST. When several items answer the question (for example several dividends or result filings), "
    "answer from the most recent item published before the question date, unless the question names a different date."
)


def _format_evidence_for_prompt(pack: EvidencePack) -> str:
    lines = []
    for i, item in enumerate(sorted(pack.evidence_items, key=lambda e: e.publication_time, reverse=True), start=1):
        lines.append(
            f"[{i}] ({item.publication_time}, {item.category}, doc={item.doc_id}): {item.claim_span}"
        )
    return "\n".join(lines)


def build_prompt(question: str, pack: EvidencePack) -> str:
    if pack.is_empty:
        evidence_block = "(no evidence items)"
    else:
        evidence_block = _format_evidence_for_prompt(pack)
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Question (asked at {pack.as_of_time}): {question}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Answer:"
    )


def generate_explanation(
    question: str, pack: EvidencePack, model: str = DEFAULT_MODEL, max_retries: int = 2
) -> str:
    """Ask the local LLM to answer ``question`` using only ``pack``'s evidence.

    If the pack is empty, the model is not even called -- there is nothing it
    could ground an answer in, so the abstention is enforced before
    generation, not left to the model's discretion (Blueprint §4.9: "no
    evidence -> no claim").
    """
    if pack.is_empty:
        return "Insufficient evidence to answer."

    prompt = build_prompt(question, pack)
    body = {"model": model, "prompt": prompt, "stream": False}
    if model.startswith("qwen3"):
        body["think"] = False          # otherwise the reply starts with a <think> block that would be fed to the guardrails
    payload = json.dumps(body).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_GENERATE_URL, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            return result.get("response", "").strip()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
    raise OllamaUnavailableError(
        f"could not reach Ollama generate endpoint after {max_retries} attempts: {last_exc}"
    ) from last_exc
