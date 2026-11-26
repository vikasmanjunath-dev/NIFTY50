"""Hallucination control (Blueprint §4.9, §23).

"The LLM is the language surface, not the source of truth." Every guardrail
here is a *mechanical*, auditable check against the evidence pack the model
was given -- never a second LLM call asking the model to grade itself (which
would just relocate the hallucination risk, not remove it).

Implemented checks (mapped to the blueprint's guardrail table, §23):

- ``check_abstention``          -> "No evidence -> no claim"
- ``check_citation_references`` -> "Citation guard"
- ``check_numeric_grounding``   -> "Numeric guard"
- ``check_claim_overlap``       -> "Citation guard" (faithfulness, coarse)
- ``check_prompt_injection``    -> "Prompt-injection guard"
- ``verify_response``           combines all of the above into one PASS/BLOCK decision.

KNOWN SIMPLIFICATION: ``check_claim_overlap`` is a lexical (word-overlap)
faithfulness check, not a trained natural-language-inference/entailment
model -- it can be fooled by a paraphrase that shares no vocabulary with its
supporting evidence, or fail to catch a subtly unsupported inference stated
in evidence-adjacent language. It is still a real, auditable check that
catches the common case (numbers or entities invented wholesale), and is
documented as a target for a real NLI model as a further refinement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from nifty_analyser.documents.evidence import EvidencePack

MIN_OVERLAP = 0.25      # chosen from the guardrail probe's pass/block trade-off (reports/guardrail_probe.json)
ABSTENTION_TEXT = "Insufficient evidence to answer."

_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|above|prior) instructions",
    r"you are now",
    r"system\s*:",
    r"disregard (all |the )?(previous|above)",
    r"new instructions?:",
]

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_NUMBER_PATTERN = re.compile(r"\b\d[\d,]*\.?\d*%?\b")
# A bare 1-2 digit integer with no decimal point and no thousands separator is
# overwhelmingly likely to be a numbered-list marker ("1.", "2.", "3." at the
# start of an LLM-formatted answer) rather than a genuine financial figure.
# Found via a real end-to-end test: llama3.2:3b formatted a well-grounded
# answer as a numbered list, and the naive version of this check flagged
# "1", "2", "3" as invented numbers -- a false positive that would have
# blocked a genuinely correct, fully-grounded response.
_LIST_MARKER_PATTERN = re.compile(r"^\d{1,2}$")


@dataclass
class VerificationResult:
    passed: bool
    reason: str
    unsupported_citations: list[int] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    injection_warnings: list[str] = field(default_factory=list)
    claim_overlap_score: float = 1.0
    abstained: bool = False


_ABSTAIN_PATTERN = re.compile(r"insufficient (evidence|information)|(cannot|can't|unable to|not able to) (answer|find|confirm|determine)|"
                              r"(does|do|did) not (mention|provide|contain|include|have)|no (relevant )?(information|evidence)", re.I)


def is_abstaining(response_text: str) -> bool:
    """The answer declines to answer. True when it OPENS with a refusal/insufficiency statement, or contains a refusal phrase and no concrete content
    (digits, ignoring [n] citation markers). A partial answer that states facts and then notes what is missing ("announced X on 2024-07-18; details
    were not disclosed") is an ANSWER, not an abstention - the first version of this check matched the trailing phrase and mislabelled such answers
    ABSTAIN, understating the answer rate. Abstentions skip only the claim-overlap check; citation and number grounding always apply."""
    t = (response_text or "").strip()
    if not _ABSTAIN_PATTERN.search(t):
        return False
    opens_with_refusal = bool(_ABSTAIN_PATTERN.search(t[:70]))
    no_content = not re.search(r"\d", re.sub(r"\[\d+\]", "", t))
    return opens_with_refusal or no_content


def check_abstention(response_text: str, pack: EvidencePack) -> bool:
    """No evidence -> no claim. If the pack is empty, the response MUST be the
    exact abstention string, not a hedged-but-still-substantive answer.
    """
    if pack.is_empty:
        return response_text.strip() == ABSTENTION_TEXT
    return True


def check_citation_references(response_text: str, pack: EvidencePack) -> list[int]:
    """Return any [N] citation markers in the response that don't correspond to
    an actual evidence item -- a fabricated citation, not a mis-numbering.
    """
    n_items = len(pack.evidence_items)
    cited = {int(m) for m in _CITATION_PATTERN.findall(response_text)}
    return sorted(c for c in cited if c < 1 or c > n_items)


def _full_evidence_text(pack: EvidencePack) -> str:
    """The evidence text as the guardrails should see it -- which MUST match
    what the LLM was actually shown in ``llm._format_evidence_for_prompt``,
    not just the raw ``claim_span``.

    Real bug found and fixed via an end-to-end test against the actual local
    model: the prompt shown to the LLM includes each item's
    ``publication_time`` (e.g. "2024-09-17T17:58:28"), and a real response
    correctly read "2024" from that timestamp and reported it -- which the
    numeric guardrail then flagged as an "invented" number, because it only
    checked against ``claim_span`` and never saw the timestamp the model was
    actually given. The guardrail's definition of "evidence" must be exactly
    what the model saw, not a subset of it, or it cannot judge groundedness
    correctly.
    """
    shown = " ".join(f"{item.publication_time} {item.category} {item.claim_span}" for item in pack.evidence_items)
    return f"{shown} {pack.as_of_time}"      # the prompt also states the question date, which a correct answer may quote


def check_numeric_grounding(response_text: str, pack: EvidencePack) -> list[str]:
    """Every number the response states must appear somewhere in the evidence
    text -- catches an invented figure not present in any retrieved document.
    """
    if pack.is_empty:
        return []
    evidence_numbers = set(_NUMBER_PATTERN.findall(_full_evidence_text(pack)))
    response_numbers = {
        n for n in _NUMBER_PATTERN.findall(response_text) if not _LIST_MARKER_PATTERN.match(n)
    }
    return sorted(response_numbers - evidence_numbers)


# Words that summarise or hedge ABOUT evidence rather than assert a fact (a faithful answer is full of them, short boilerplate evidence never contains them).
_GENERIC = set("""made make makes latest recent recently most earliest first last next previous prior multiple several occasions occasion instances instance times time
specific details detail information info provided provide stated state states specify specifies specified nature kind type types however but not none nothing there their they them
evidence items item announcement announcements announced announce announces company companys informed exchange regarding about date dates including include includes such also
same repeated twice once followed then only other others any further available mentioned mention listed list shows show indicates indicate suggests suggest appears appear seems seem
likely possibly possible which whose since during between after before within per each every all some many few more less one two three four five six seven eight nine ten
disclosed disclose disclosure disclosures reported report reports reporting regarding related relates relating concerning concern concerns while although though thus therefore hence
contains contain contained refers refer referring pertain pertains pertaining amount number count total overall various different given based according can could would should may might will
insufficient exact answer additional earlier beyond no under categorized categorised category categories target targets purpose subject topic relevant relevance indicating whether unspecified unclear unknown unavailable description descriptions content contents text
has have had been being are was were is be do does did get got see seen note noted notably mainly mostly primarily generally typically sometimes often again still yet already just very
new old high low large small major minor significant key main whether if when where what who whom how why because so as than that this these those with without into onto from to of in on at by for and or the a an it its""".split())


def check_claim_overlap(response_text: str, pack: EvidencePack, min_overlap: float = 0.15) -> float:
    """Grounding of the response's FACTUAL CONTENT words: fraction found in the evidence text the model was shown. Generic summarising/hedging vocabulary,
    the question's own words (ticker, topic), citation markers and numbers (checked separately) are exempt; a response with no content words left makes no
    new factual claim (1.0). Invented events, categories, companies or places are novel content words and drive the score down.
    (History: a raw word-set overlap blocked ~all faithful prose about short boilerplate evidence; found by scripts/run_guardrail_probe.py.)"""
    if pack.is_empty:
        return 1.0 if response_text.strip() == ABSTENTION_TEXT else 0.0

    def stem(w: str) -> str:            # light plural/tense stemming so "acquisitions" matches "Acquisition"
        for suf in ("ings", "ing", "ies", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)] + ("y" if suf == "ies" else ""); break
        return w[:-1] if len(w) > 4 and w.endswith("e") else w      # change / changes / changed -> chang
    tok = lambda t: [stem(w) for w in re.findall(r"[a-z][a-z&\-]{1,}", t.lower())]
    exempt = {stem(w) for w in _GENERIC} | set(_GENERIC) | set(tok(pack.query or ""))
    content = {w for w in tok(re.sub(r"\[[^\]]*\]", " ", response_text)) if w not in exempt}
    if not content:
        return 1.0
    evidence_words = set(tok(_full_evidence_text(pack)))
    return sum(1 for w in content if w in evidence_words) / len(content)


def check_prompt_injection(pack: EvidencePack) -> list[str]:
    """Scan the evidence itself for embedded instruction-like text -- retrieved
    documents are untrusted input, and a document containing "ignore previous
    instructions" style text must be flagged, not silently followed.
    """
    warnings = []
    for item in pack.evidence_items:
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, item.claim_span, re.IGNORECASE):
                warnings.append(f"{item.evidence_id}: matched pattern {pattern!r}")
    return warnings


def verify_response(response_text: str, pack: EvidencePack) -> VerificationResult:
    """Run every guardrail and return one PASS/BLOCK decision with full detail."""
    if not check_abstention(response_text, pack):
        return VerificationResult(
            passed=False,
            reason="pack was empty but response did not abstain",
        )

    injection_warnings = check_prompt_injection(pack)
    unsupported_citations = check_citation_references(response_text, pack)
    unsupported_numbers = check_numeric_grounding(response_text, pack)
    overlap = check_claim_overlap(response_text, pack)

    if unsupported_citations:
        return VerificationResult(
            passed=False,
            reason=f"response cites evidence item(s) that don't exist: {unsupported_citations}",
            unsupported_citations=unsupported_citations,
            injection_warnings=injection_warnings,
            claim_overlap_score=overlap,
        )
    if unsupported_numbers:
        return VerificationResult(
            passed=False,
            reason=f"response contains number(s) not found in any evidence: {unsupported_numbers}",
            unsupported_numbers=unsupported_numbers,
            injection_warnings=injection_warnings,
            claim_overlap_score=overlap,
        )
    abstained = is_abstaining(response_text)
    if overlap < MIN_OVERLAP and not pack.is_empty and not abstained:
        return VerificationResult(
            passed=False,
            reason=f"claim overlap with evidence too low ({overlap:.2f} < {MIN_OVERLAP})",
            injection_warnings=injection_warnings,
            claim_overlap_score=overlap,
        )

    return VerificationResult(
        passed=True,
        reason="model abstained (no claims to verify beyond citations/numbers)" if abstained else "all guardrails passed",
        injection_warnings=injection_warnings,
        claim_overlap_score=overlap,
        abstained=abstained,
    )
