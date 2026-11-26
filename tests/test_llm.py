"""Tests for the LLM explanation layer (nifty_analyser/documents/llm.py).

Network calls to Ollama are mocked here for CI-safety and speed; a separate
real end-to-end smoke test against the actual local Ollama server was run
manually during development (see TRACKER.md) rather than made part of the
permanent suite, since it depends on external local infrastructure being up.
"""
from __future__ import annotations

import json

import pytest

from nifty_analyser.documents.embeddings import OllamaUnavailableError
from nifty_analyser.documents.evidence import EvidenceItem, EvidencePack
from nifty_analyser.documents.llm import build_prompt, generate_explanation


def _pack() -> EvidencePack:
    item = EvidenceItem(
        evidence_id="EV_1",
        doc_id="d1",
        instrument_id="NSE:RELIANCE",
        source_id="nse_corporate_announcements",
        publication_time="2024-01-10T10:00:00",
        category="Financial Results",
        claim_span="The board approved unaudited results showing revenue of 2438.65 crore.",
        relevance_score=0.9,
        content_hash="sha256:abc",
    )
    return EvidencePack(query="What were the results?", as_of_time="2024-01-15T00:00:00", evidence_items=[item])


def test_build_prompt_includes_evidence_and_question():
    prompt = build_prompt("What were the results?", _pack())
    assert "What were the results?" in prompt
    assert "2438.65 crore" in prompt
    assert "[1]" in prompt


def test_build_prompt_handles_empty_pack():
    from nifty_analyser.documents.evidence import EvidencePack

    empty = EvidencePack(query="q", as_of_time="2024-01-15T00:00:00", evidence_items=[])
    prompt = build_prompt("q", empty)
    assert "no evidence items" in prompt


def test_generate_explanation_does_not_call_model_when_pack_empty(monkeypatch):
    """Enforced abstention: an empty pack must never even reach the model --
    there is nothing to ground an answer in.
    """
    import nifty_analyser.documents.llm as llm_module

    def _fail_if_called(*_a, **_k):
        raise AssertionError("the model must not be called with an empty evidence pack")

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", _fail_if_called)

    from nifty_analyser.documents.evidence import EvidencePack

    empty = EvidencePack(query="q", as_of_time="2024-01-15T00:00:00", evidence_items=[])
    result = generate_explanation("q", empty)
    assert result == "Insufficient evidence to answer."


def test_generate_explanation_returns_model_response(monkeypatch):
    import nifty_analyser.documents.llm as llm_module

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"response": "Revenue was 2438.65 crore, per [1]."}).encode("utf-8")

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())

    result = generate_explanation("What were the results?", _pack())
    assert "2438.65" in result


def test_generate_explanation_raises_clear_error_when_ollama_down(monkeypatch):
    import urllib.error

    import nifty_analyser.documents.llm as llm_module

    def _raise(*_a, **_k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", _raise)

    with pytest.raises(OllamaUnavailableError):
        generate_explanation("q", _pack(), max_retries=1)


def test_prompt_lists_evidence_newest_first_and_states_question_time():
    from nifty_analyser.documents.evidence import EvidenceItem, EvidencePack
    from nifty_analyser.documents.llm import build_prompt
    mk = lambda i, t: EvidenceItem(f"EV_{i}", i, "NSE:X", "s", t, "Dividend", f"text {i}", 1.0, "h")
    p = build_prompt("q?", EvidencePack("q?", "2026-06-01T00:00:00+05:30", [mk("old", "2024-01-01T00:00:00+05:30"), mk("new", "2026-01-01T00:00:00+05:30")]))
    assert p.index("doc=new") < p.index("doc=old") and "asked at 2026-06-01" in p and "NEWEST FIRST" in p
