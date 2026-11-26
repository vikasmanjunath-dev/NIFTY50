"""``/v3/evidence`` - point-in-time document retrieval as a service (Blueprint §4 evidence packs, no LLM text generation).

Same guarantees as ``/v3/analyse``: a future ``as_of`` is rejected; only announcements with ``available_time <= as_of`` can be returned
(enforced inside retrieval, re-checked by ``build_evidence_pack``); unknown instruments are REFUSED; every call is written to the audit
chain (query, as_of, returned doc ids + content hashes) so a result can be reproduced. It degrades instead of failing: if the embedder
is down it falls back to BM25-only, if the LLM reranker is down or absent it returns the first-stage order, and says so in ``warnings``."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field

from nifty_analyser.data.instrument_master import FORMER_NAMES, NIFTY50
from nifty_analyser.data.timestamps import to_ist
from nifty_analyser.documents.digest import DIGEST_MIN_RERANK, best_rerank, build_digest
from nifty_analyser.documents.evidence import EvidenceItem, EvidencePack, build_evidence_pack
from nifty_analyser.documents.guardrails import verify_response
from nifty_analyser.documents.index import VectorIndex, hybrid_retrieve_indexed, retrieve
from nifty_analyser.documents.embeddings import OllamaUnavailableError as EmbedderUnavailable
from nifty_analyser.documents.llm import OllamaUnavailableError as LLMUnavailable
from nifty_analyser.ops.audit import AuditLog, digest
from nifty_analyser.ops.service import FutureAsOfError
from nifty_analyser.ops.temporal import has_explicit_time, parse_time_anchor

_DOWN = (EmbedderUnavailable, LLMUnavailable)       # the embedder and the LLM modules define separate exception classes
LATEST_RE = re.compile(r"\b(latest|most recent|recent(ly)?|current(ly)?|last|newest|newly)\b", re.I)
RECENCY_DEFAULT_ON = True   # company-specific questions only: judged relevance unchanged there (0.555 -> 0.549), but market-wide topical nDCG@5 dipped 0.774 -> 0.749
ROUTE_TOP_N, ROUTE_WEIGHT = 2, 3.0     # category routing; n=3 was worse, weight 3 = 10 on the judged benchmark (reports/judged_eval.json)
RECENCY_WEIGHT = 0.5      # chosen on a tune sample, confirmed on a holdout (reports/recency_retrieval.json)
DISCLAIMER = "Retrieved company announcements for research; relevance is model-estimated (see TRACKER.md for measured accuracy). Not investment advice."


class EvidenceRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    instrument_id: str | None = Field(None, pattern=r"^NSE:[A-Z0-9&\-]+$", examples=["NSE:RELIANCE"])
    as_of: datetime | None = Field(None, description="Decision time (default now). Naive = IST. Future values are rejected.")
    top_k: int = Field(5, ge=1, le=10)
    rerank: bool = Field(False, description="LLM-rerank the first-stage candidates (slower; measured +0.09 nDCG@5).")
    window_days: int | None = Field(None, ge=1, le=3650, description="Only consider announcements published within this many days before as_of.")
    prefer_recent: bool | None = Field(None, description="Blend a recency boost into retrieval (weight 0.5; hit-rate for 'latest filing' questions 0.90 -> 1.00 on a 200-question holdout). "
                                       "None = automatic: on when the query says latest/most recent/current/last/newest OR (for company-specific queries) names no time period at all; a named month/year restricts the search to that period instead. Leave off for questions about a specific past filing.")
    strict_faithfulness: bool = Field(False, description="/v3/explain only: also veto answers a pretrained NLI model cannot entail from the evidence (needs the server started with --nli). "
                                      "Blocks 99.9% of planted fabrications vs 95.7% for the lexical check, but in a replay on the study's real packs it withheld 3 of 21 faithful fact-stating answers (14%): off by default.")
    request_id: str | None = Field(None, max_length=64)


class EvidenceResponse(BaseModel):
    contract_version: Literal["v3"] = "v3"
    request_id: str
    status: Literal["OK", "EMPTY", "REFUSED"]
    query: str
    instrument_id: str | None = None
    as_of: datetime
    retrieval: str = ""
    reranked: bool = False
    pack: dict | None = None
    warnings: list[str] = []
    disclaimer: str = DISCLAIMER
    audit_seq: int | None = None
    audit_hash: str | None = None


@dataclass
class DocsContext:
    documents: pd.DataFrame                        # doc_id, text, available_time (IST), instrument_id, category, source_id
    index: VectorIndex
    audit: AuditLog
    scorer: Callable[[str, list[str]], list[float]] | None = None
    router: object | None = None                 # CategoryRouter: question -> likely NSE categories (measured nDCG@5 +0.028 +/- 0.010); None disables
    nli_fn: Callable[[str, EvidencePack], dict] | None = None       # documents.nli.nli_check; advisory unless the request sets strict_faithfulness
    digest: bool = True                           # allow the gated digest fallback (needs a reranker run on the request)
    generate: Callable[[str, EvidencePack], str] | None = None     # (question, pack) -> answer text; None disables /v3/explain
    now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="Asia/Kolkata")


def evidence(req: EvidenceRequest, ctx: DocsContext) -> EvidenceResponse:
    now = ctx.now()
    as_of = to_ist(req.as_of) if req.as_of else now
    if as_of > now + pd.Timedelta(seconds=1):
        raise FutureAsOfError("as_of is in the future; evidence can only use information available now")
    rw = 0.0
    rid, warnings, pack, status, mode, reranked = req.request_id or uuid.uuid4().hex[:16], [], None, "OK", "", False
    if req.instrument_id is not None and req.instrument_id not in {i.instrument_id for i in NIFTY50.values()}:
        status, warnings = "REFUSED", [f"unknown instrument {req.instrument_id}"]
    else:
        docs = ctx.documents
        if req.window_days:
            docs = docs[docs["available_time"] > as_of - pd.Timedelta(days=req.window_days)]
        kw = dict(documents=docs, as_of=as_of, index=ctx.index, instrument_id=req.instrument_id, top_k=req.top_k)
        anchor = parse_time_anchor(req.query, as_of)
        if anchor is not None:                       # the question names a period: search only that period, no recency boost
            docs = docs[(docs["available_time"] >= anchor[0]) & (docs["available_time"] <= anchor[1])]
            warnings.append(f"time anchor applied: {anchor[0].date()} .. {anchor[1].date()}")
        kw["documents"] = docs
        explicit = anchor is not None or has_explicit_time(req.query)
        # default: no explicit time reference => 'as of now' semantics => freshest information wins (prefer_recent=False/True override)
        rw = RECENCY_WEIGHT if (req.prefer_recent or (req.prefer_recent is None and (LATEST_RE.search(req.query) or (RECENCY_DEFAULT_ON and not explicit and req.instrument_id is not None)))) else 0.0
        kw["recency_weight"] = rw
        if ctx.router is not None:
            kw.update(router=ctx.router, route_top_n=ROUTE_TOP_N, route_weight=ROUTE_WEIGHT)
        want_rerank = req.rerank and ctx.scorer is not None
        if req.rerank and ctx.scorer is None:
            warnings.append("rerank requested but no reranker configured; returning first-stage order")
        try:
            try:
                res, mode, reranked = retrieve(req.query, scorer=ctx.scorer if want_rerank else None, **kw), "hybrid", want_rerank
            except _DOWN:
                if not want_rerank:
                    raise
                warnings.append("reranker unavailable; returning first-stage order")
                res, mode = retrieve(req.query, **kw), "hybrid"
        except _DOWN:
            warnings.append("embedder unavailable; BM25-only fallback")
            res, mode = hybrid_retrieve_indexed(req.query, docs, as_of, ctx.index, top_k=req.top_k, instrument_id=req.instrument_id, mode="bm25"), "bm25"
        pack = build_evidence_pack(req.query, as_of, res, max_items=req.top_k)
        if not pack.temporal_valid:                      # unreachable if retrieval is correct; fail closed if it ever happens
            pack, status, warnings = None, "REFUSED", warnings + ["temporal validation failed"]
        elif pack.is_empty:
            status = "EMPTY"
    entry = ctx.audit.append("evidence", {"request_id": rid, "query": req.query, "instrument_id": req.instrument_id, "as_of": as_of.isoformat(), "status": status,
                                          "retrieval": mode, "reranked": reranked, "recency_weight": rw,
                                          "evidence": [(e.doc_id, e.content_hash) for e in pack.evidence_items] if pack else []})
    return EvidenceResponse(request_id=rid, status=status, query=req.query, instrument_id=req.instrument_id, as_of=as_of, retrieval=mode, reranked=reranked,
                            pack=pack.to_dict() if pack else None, warnings=warnings, audit_seq=entry["seq"], audit_hash=entry["hash"])


class ExplainResponse(BaseModel):
    contract_version: Literal["v3"] = "v3"
    request_id: str
    status: Literal["OK", "ABSTAIN", "BLOCKED", "DEGRADED", "REFUSED"]
    question: str
    answer_source: Literal["llm", "digest"] | None = None   # digest = deterministic summary of the evidence, used when the LLM refuses/is down but the reranker rates the top filing >= 7/10
    answer: str | None = None                      # present only when every guardrail passed (or the answer is the enforced abstention)
    verification: dict | None = None
    evidence: EvidenceResponse
    warnings: list[str] = []
    disclaimer: str = DISCLAIMER
    audit_seq: int | None = None
    audit_hash: str | None = None


def explain(req: EvidenceRequest, ctx: DocsContext) -> ExplainResponse:
    """Grounded answer: retrieve -> LLM answers ONLY from the pack -> guardrails (citations exist, numbers appear in evidence, claim overlap,
    abstention when the pack is empty). A failed guardrail withholds the text (BLOCKED) - the caller gets the evidence, never an unverified answer.
    No evidence -> the model is never called. LLM down -> DEGRADED with the evidence only."""
    ev = evidence(req, ctx)
    warnings, answer, ver, status = list(ev.warnings), None, None, ev.status
    if ev.status == "REFUSED" or ctx.generate is None:
        status = "REFUSED" if ev.status == "REFUSED" else "DEGRADED"
        if ctx.generate is None and ev.status != "REFUSED":
            warnings.append("no generator configured; returning evidence only")
    else:
        pack = EvidencePack(**{**ev.pack, "evidence_items": [EvidenceItem(**i) for i in ev.pack["evidence_items"]]})
        try:
            names = FORMER_NAMES.get(req.instrument_id or "", ())
            question = req.query + (f" (Note: this company was previously named {' / '.join(names)}; evidence may use the earlier name.)" if names else "")
            text = "Insufficient evidence to answer." if pack.is_empty else ctx.generate(question, pack)     # no evidence -> the model is never called
        except _DOWN:
            status = "DEGRADED"; warnings.append("LLM unavailable; returning evidence only")
        else:
            v = verify_response(text, pack)
            ver = {"passed": v.passed, "reason": v.reason, "claim_overlap": round(v.claim_overlap_score, 3), "unsupported_citations": v.unsupported_citations,
                   "abstained": v.abstained, "unsupported_numbers": v.unsupported_numbers, "injection_warnings": v.injection_warnings}
            if v.passed and ctx.nli_fn is not None and not pack.is_empty and not v.abstained:
                n = ctx.nli_fn(text, pack); ver["nli"] = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in n.items() if k != "passed"}
                if req.strict_faithfulness and n["n_claims"] and (n["max_contra"] > 0.9 or n["min_entail"] < 0.003):
                    v.passed = False; v.reason = f"NLI veto: a claim is not entailed by the evidence ({n['worst_claim'][:80]!r})"; ver["passed"], ver["reason"] = False, v.reason
            if v.passed:
                answer, status = text, ("ABSTAIN" if pack.is_empty or v.abstained else "OK")
            else:
                status = "BLOCKED"; warnings.append(f"answer withheld: {v.reason}")
    source = "llm" if (answer and status == "OK") else None
    if ctx.digest and ev.pack and status in ("ABSTAIN", "DEGRADED"):        # LLM refused / unavailable: fall back to the evidence digest, gated by relevance
        pk = EvidencePack(**{**ev.pack, "evidence_items": [EvidenceItem(**i) for i in ev.pack["evidence_items"]]})
        br = best_rerank(pk)
        if not pk.is_empty and br is not None and br >= DIGEST_MIN_RERANK:
            answer, source, status = build_digest(pk), "digest", "OK"
            ver = {**(ver or {}), "passed": True, "reason": f"digest fallback: top filing reranked {br:.0f}/10", "answer_source": "digest"}
            warnings.append("answer is a deterministic digest of the retrieved filings (the model did not answer)")
    entry = ctx.audit.append("explain", {"request_id": ev.request_id, "evidence_audit_seq": ev.audit_seq, "status": status, "answer_source": source,
                                         "answer_sha256": digest(answer) if answer else None, "verification": ver})
    return ExplainResponse(request_id=ev.request_id, status=status, question=req.query, answer_source=source, answer=answer, verification=ver, evidence=ev, warnings=warnings,
                           audit_seq=entry["seq"], audit_hash=entry["hash"])
