import numpy as np, pandas as pd, pytest
from fastapi.testclient import TestClient
from nifty_analyser.documents.index import VectorIndex
from nifty_analyser.documents.llm import OllamaUnavailableError
from nifty_analyser.ops.api import create_app
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.evidence_service import DocsContext, EvidenceRequest, evidence
from nifty_analyser.ops.service import FutureAsOfError
import tests.test_ops as T

TZ = "Asia/Kolkata"


class Emb:
    def __init__(self): self.fail = False
    def __call__(self, texts):
        if self.fail: raise OllamaUnavailableError("down")
        out = np.zeros((len(texts), 26), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z": out[i, ord(ch) - 97] += 1
        return out


NOW = pd.Timestamp("2026-06-01 12:00", tz=TZ)


def mk(tmp_path, scorer=None):
    t = pd.date_range("2026-01-01", periods=6, freq="10D", tz=TZ)          # last doc published 2026-02-20; one doc is in the "future" below
    d = pd.DataFrame({"doc_id": [f"d{i}" for i in range(6)], "text": ["dividend declared board", "loss of share certificate", "resignation of director",
                      "dividend record date fixed", "merger approved scheme", "quarterly results profit"], "available_time": t,
                      "instrument_id": ["NSE:RELIANCE", "NSE:RELIANCE", "NSE:TCS", "NSE:RELIANCE", "NSE:TCS", "NSE:TCS"], "category": "c", "source_id": "s"})
    d.loc[5, "available_time"] = NOW + pd.Timedelta(days=3)               # published after "now": must never be served
    emb = Emb(); ix = VectorIndex(tmp_path / "ix", emb); ix.add(d)
    return DocsContext(d, ix, AuditLog(tmp_path / "a.jsonl"), scorer, now=lambda: NOW), emb


def test_returns_only_published_docs_for_instrument_and_audits(tmp_path):
    ctx, _ = mk(tmp_path)
    r = evidence(EvidenceRequest(query="dividend declared", instrument_id="NSE:RELIANCE", top_k=5), ctx)
    ids = [e["doc_id"] for e in r.pack["evidence_items"]]
    assert r.status == "OK" and set(ids) <= {"d0", "d1", "d3"} and ids[0] in {"d0", "d3"} and r.audit_seq == 1
    assert ctx.audit.verify()[0] and ctx.audit.read()[0]["payload"]["evidence"][0][0] == ids[0]


def test_as_of_in_the_past_hides_later_docs_and_future_as_of_is_rejected(tmp_path):
    ctx, _ = mk(tmp_path)
    r = evidence(EvidenceRequest(query="dividend", as_of=pd.Timestamp("2026-01-05", tz=TZ).to_pydatetime()), ctx)
    assert {e["doc_id"] for e in r.pack["evidence_items"]} == {"d0"}
    with pytest.raises(FutureAsOfError):
        evidence(EvidenceRequest(query="dividend", as_of=(NOW + pd.Timedelta(days=1)).to_pydatetime()), ctx)
    allr = evidence(EvidenceRequest(query="quarterly results profit", top_k=10), ctx)
    assert "d5" not in {e["doc_id"] for e in allr.pack["evidence_items"]}      # the not-yet-published doc


def test_unknown_instrument_refused_and_empty_status(tmp_path):
    ctx, _ = mk(tmp_path)
    assert evidence(EvidenceRequest(query="dividend", instrument_id="NSE:NOPE"), ctx).status == "REFUSED"
    assert evidence(EvidenceRequest(query="dividend", window_days=1), ctx).status == "EMPTY"


def test_degrades_when_embedder_or_reranker_is_down(tmp_path):
    def bad(q, t): raise OllamaUnavailableError("rerank down")
    ctx, emb = mk(tmp_path, scorer=bad)
    r = evidence(EvidenceRequest(query="dividend declared", rerank=True), ctx)
    assert r.status == "OK" and not r.reranked and any("reranker unavailable" in w for w in r.warnings)
    emb.fail = True
    r2 = evidence(EvidenceRequest(query="dividend declared"), ctx)
    assert r2.retrieval == "bm25" and r2.status == "OK" and any("BM25-only" in w for w in r2.warnings)
    assert any("no reranker" in w for w in evidence(EvidenceRequest(query="dividend", rerank=True), mk(tmp_path / "x", None)[0]).warnings)


def test_http_endpoint_exists_only_when_docs_configured(tmp_path):
    sctx = T.mk(tmp_path / "s") if hasattr(T, "mk") else None
    dctx, _ = mk(tmp_path / "d")
    on = TestClient(create_app(sctx, docs=dctx)); off = TestClient(create_app(sctx))
    assert on.post("/v3/evidence", json={"query": "dividend declared"}).json()["status"] == "OK"
    assert on.post("/v3/evidence", json={"query": "x"}).status_code == 422                   # min length
    assert on.post("/v3/evidence", json={"query": "dividend", "as_of": "2999-01-01T00:00:00"}).status_code == 422
    assert off.post("/v3/evidence", json={"query": "dividend declared"}).status_code == 404


def _gen_ctx(tmp_path, gen):
    ctx, emb = mk(tmp_path); ctx.generate = gen; return ctx


def test_explain_ok_blocked_degraded_and_abstain(tmp_path):
    from nifty_analyser.ops.evidence_service import explain
    seen = []
    def good(q, pack): seen.append(q); return "A dividend was declared by the board [EV_d0]."
    r = explain(EvidenceRequest(query="dividend declared", instrument_id="NSE:RELIANCE"), _gen_ctx(tmp_path / "a", good))
    assert r.status == "OK" and r.answer and r.verification["passed"] and r.audit_seq == 2 and r.evidence.audit_seq == 1
    bad = lambda q, pack: "The board declared a dividend of 99999 rupees per share [EV_d0]."          # number not in evidence
    r2 = explain(EvidenceRequest(query="dividend declared", instrument_id="NSE:RELIANCE"), _gen_ctx(tmp_path / "b", bad))
    assert r2.status == "BLOCKED" and r2.answer is None and r2.evidence.pack["evidence_items"] and "withheld" in r2.warnings[0]
    def down(q, pack): raise OllamaUnavailableError("llm down")
    r3 = explain(EvidenceRequest(query="dividend declared"), _gen_ctx(tmp_path / "c", down))
    assert r3.status == "DEGRADED" and r3.answer is None and r3.evidence.status == "OK"
    called = []
    def spy(q, pack): called.append(1); return "Insufficient evidence to answer."
    r4 = explain(EvidenceRequest(query="dividend", window_days=1), _gen_ctx(tmp_path / "d", spy))     # nothing in window -> EMPTY pack
    assert r4.evidence.status == "EMPTY" and r4.status == "ABSTAIN" and r4.answer.startswith("Insufficient") and called == []     # generator never invoked


def test_explain_endpoint_gating_and_refusal(tmp_path):
    sctx = T.mk(tmp_path / "s"); dctx, _ = mk(tmp_path / "d")
    assert TestClient(create_app(sctx, docs=dctx)).post("/v3/explain", json={"query": "dividend declared"}).status_code == 404       # no generator
    dctx.generate = lambda q, p: "Dividend declared [EV_d0]."
    c = TestClient(create_app(sctx, docs=dctx))
    assert c.post("/v3/explain", json={"query": "dividend declared", "instrument_id": "NSE:RELIANCE"}).json()["status"] == "OK"
    assert c.post("/v3/explain", json={"query": "dividend", "instrument_id": "NSE:NOPE"}).json()["status"] == "REFUSED"
    assert c.post("/v3/explain", json={"query": "dividend", "as_of": "2999-01-01T00:00:00"}).status_code == 422


def test_recency_boost_prefers_newer_filing_and_auto_detects_latest_keyword(tmp_path):
    ctx, _ = mk(tmp_path)          # d0 (Jan 1) and d3 (Jan 31) are both RELIANCE dividend filings; d3 is newer
    req = lambda **k: EvidenceRequest(query="dividend", instrument_id="NSE:RELIANCE", top_k=1, **k)
    assert evidence(req(prefer_recent=True), ctx).pack["evidence_items"][0]["doc_id"] == "d3"
    assert evidence(EvidenceRequest(query="latest dividend", instrument_id="NSE:RELIANCE", top_k=1), ctx).audit_seq and ctx.audit.read()[-1]["payload"]["recency_weight"] == 0.5
    evidence(req(), ctx); assert ctx.audit.read()[-1]["payload"]["recency_weight"] == 0.5           # no time reference at all -> 'as of now' semantics -> on by default
    evidence(EvidenceRequest(query="dividend", top_k=1), ctx); assert ctx.audit.read()[-1]["payload"]["recency_weight"] == 0.0     # market-wide, no keyword -> off
    evidence(req(prefer_recent=False), ctx); assert ctx.audit.read()[-1]["payload"]["recency_weight"] == 0.0


def test_explicit_time_reference_restricts_period_and_disables_boost(tmp_path):
    ctx, _ = mk(tmp_path)          # RELIANCE dividend filings: d0 = 2026-01-01, d3 = 2026-01-31; add an older Dec-2025 one
    import pandas as pd
    r = evidence(EvidenceRequest(query="dividend declared in January 2026", instrument_id="NSE:RELIANCE", top_k=5), ctx)
    assert {e["doc_id"] for e in r.pack["evidence_items"]} <= {"d0", "d1", "d3"} and any("time anchor" in w for w in r.warnings) and ctx.audit.read()[-1]["payload"]["recency_weight"] == 0.0
    r2 = evidence(EvidenceRequest(query="dividend declared in March 2025", instrument_id="NSE:RELIANCE", top_k=5), ctx)
    assert r2.status == "EMPTY"                                                                # nothing published in that period


def test_parse_time_anchor_forms_and_future_clipping():
    import pandas as pd
    from nifty_analyser.ops.temporal import has_explicit_time, parse_time_anchor
    now = pd.Timestamp("2026-06-15", tz="Asia/Kolkata")
    a = parse_time_anchor("record date announced in Nov 2023", now); assert a[0] == pd.Timestamp("2023-11-01", tz="Asia/Kolkata") and a[1].month == 11 and a[1].day == 30
    y = parse_time_anchor("results during 2022", now); assert y[0].year == y[1].year == 2022
    c = parse_time_anchor("dividend in June 2026", now); assert c[1] == now                    # clipped at as_of: never reaches into the future
    assert parse_time_anchor("dividend in January 2099", now) is None and parse_time_anchor("latest dividend", now) is None
    assert has_explicit_time("what happened last quarter") and not has_explicit_time("record date of the dividend")


def test_category_router_routes_by_name_similarity_and_service_accepts_it(tmp_path):
    from nifty_analyser.documents.index import CategoryRouter
    emb = Emb(); r = CategoryRouter(["Dividend", "Loss of Share Certificate", "Credit Rating", ""], emb)
    assert r.categories == ["Credit Rating", "Dividend", "Loss of Share Certificate"]        # blank dropped, sorted case-insensitively
    assert r.route("dividend declared by the board", top_n=1) == ["Dividend"] and len(r.route("anything", top_n=2)) == 2
    ctx, _ = mk(tmp_path); ctx.documents = ctx.documents.assign(category=["Dividend", "Loss of Share Certificate", "Personnel", "Dividend", "Acquisition", "Results"]); ctx.router = r
    res = evidence(EvidenceRequest(query="dividend declared", instrument_id="NSE:RELIANCE", top_k=2), ctx)
    assert res.status == "OK" and res.pack["evidence_items"][0]["doc_id"] in {"d0", "d3"}


def test_strict_faithfulness_nli_veto_is_opt_in(tmp_path):
    from nifty_analyser.ops.evidence_service import explain
    ctx, _ = mk(tmp_path); ctx.generate = lambda q, p: "A dividend was declared by the board [EV_d0]."
    ctx.nli_fn = lambda a, p: {"passed": False, "n_claims": 1, "min_entail": 0.0001, "max_contra": 0.2, "worst_claim": "A dividend was declared by the board."}
    base = EvidenceRequest(query="dividend declared", instrument_id="NSE:RELIANCE")
    r = explain(base, ctx); assert r.status == "OK" and r.verification["nli"]["min_entail"] == 0.0001           # advisory by default
    r2 = explain(base.model_copy(update={"strict_faithfulness": True}), ctx); assert r2.status == "BLOCKED" and "NLI veto" in r2.warnings[0] and r2.answer is None


def test_digest_fallback_is_gated_by_rerank_score_and_labelled(tmp_path):
    from nifty_analyser.ops.evidence_service import explain
    refuse = lambda q, p: "Insufficient evidence to answer."
    req = lambda **k: EvidenceRequest(query="dividend declared", instrument_id="NSE:RELIANCE", rerank=True, **k)
    ctx, _ = mk(tmp_path / "hi", scorer=lambda q, texts: [9.0] * len(texts)); ctx.generate = refuse
    r = explain(req(), ctx)
    assert r.status == "OK" and r.answer_source == "digest" and "newest first" in r.answer and r.answer.count("[") >= 1 and ctx.audit.read()[-1]["payload"]["answer_source"] == "digest"
    ctx2, _ = mk(tmp_path / "lo", scorer=lambda q, texts: [2.0] * len(texts)); ctx2.generate = refuse
    assert explain(req(), ctx2).status == "ABSTAIN"                                                   # irrelevant by the reranker: stays an abstention
    ctx3, _ = mk(tmp_path / "none"); ctx3.generate = refuse
    assert explain(req(), ctx3).status == "ABSTAIN"                                                   # no reranker ran: no gate, no digest
    ctx4, _ = mk(tmp_path / "off", scorer=lambda q, texts: [9.0] * len(texts)); ctx4.generate = refuse; ctx4.digest = False
    assert explain(req(), ctx4).status == "ABSTAIN"
    def down(q, p): raise OllamaUnavailableError("llm down")
    ctx5, _ = mk(tmp_path / "dn", scorer=lambda q, texts: [9.0] * len(texts)); ctx5.generate = down
    assert explain(req(), ctx5).answer_source == "digest"                                             # LLM down + relevant evidence still answers
