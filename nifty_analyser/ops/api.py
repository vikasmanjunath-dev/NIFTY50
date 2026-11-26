"""FastAPI app for the v3 contract. Admin endpoints are DISABLED unless STM_ADMIN_TOKEN is set
(secure by default) and use constant-time token comparison."""
from __future__ import annotations

import hmac
import os

from datetime import date

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from nifty_analyser.ops.dashboard import render_dashboard
from pydantic import Field
from nifty_analyser.documents.event_classifier import mask_text
from nifty_analyser.ops.evidence_service import DocsContext, EvidenceRequest, EvidenceResponse, ExplainResponse, evidence, explain
from nifty_analyser.ops.service import AnalyseRequest, AnalyseResponse, FutureAsOfError, ServingContext, analyse, replay


class ClassifyRequest(BaseModel):          # module level: FastAPI cannot resolve a request model defined inside create_app under postponed annotations
    text: str = Field(min_length=3, max_length=4000)
    category: str | None = Field(None, max_length=200, description="NSE category if known; it is masked out before classification, as in training.")


class TripRequest(BaseModel):
    reason: str
    by: str


class ResetRequest(BaseModel):
    by: str


def create_app(ctx: ServingContext, psi_provider=lambda: None, metrics: dict | None = None, docs: DocsContext | None = None, classifier=None,
               volatility: dict | None = None) -> FastAPI:
    app = FastAPI(title="Nifty 50 analyser", version="3.0")

    def admin(x_admin_token: str | None = Header(default=None)):
        token = os.environ.get("STM_ADMIN_TOKEN")
        if not token:
            raise HTTPException(403, "admin endpoints are disabled (STM_ADMIN_TOKEN not set)")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, token):
            raise HTTPException(401, "invalid admin token")

    @app.post("/v3/analyse", response_model=AnalyseResponse)
    def post_analyse(req: AnalyseRequest):
        try:
            return analyse(req, ctx)
        except FutureAsOfError as exc:
            raise HTTPException(422, str(exc)) from exc

    if docs is not None:                      # evidence retrieval is opt-in: needs the announcement corpus and the vector index
        @app.post("/v3/evidence", response_model=EvidenceResponse)
        def post_evidence(req: EvidenceRequest):
            try:
                return evidence(req, docs)
            except FutureAsOfError as exc:
                raise HTTPException(422, str(exc)) from exc

    if docs is not None and docs.generate is not None:
        @app.post("/v3/explain", response_model=ExplainResponse)
        def post_explain(req: EvidenceRequest):
            try:
                return explain(req, docs)
            except FutureAsOfError as exc:
                raise HTTPException(422, str(exc)) from exc

    if classifier is not None:
        @app.post("/v3/classify")
        def post_classify(req: ClassifyRequest):
            m = mask_text(req.text, req.category)
            if len(m) < 25:
                return {"contract_version": "v3", "status": "ABSTAIN", "label": None, "confidence": None, "reason": "too little content beyond company name/category"}
            (label, conf), = classifier.predict([m])
            return {"contract_version": "v3", "status": "OK" if label else "ABSTAIN", "label": label, "confidence": round(conf, 4), "threshold": classifier.threshold,
                    "classes": classifier.classes_, "model": classifier.meta.get("n_train")}

    if volatility is not None:
        @app.get("/v3/volatility")
        def get_volatility(instrument_id: str, horizon_days: int = 5):
            block = volatility.get(str(horizon_days))
            if block is None:
                raise HTTPException(422, f"horizon_days must be one of {sorted(map(int, volatility))}")
            hit = block["forecasts"].get(instrument_id)
            age = (ctx.now_fn().date() - date.fromisoformat(block["data_date"])).days
            if ctx.killswitch.manual().get("tripped"):
                status, hit = "REFUSED", None
            elif hit is None:
                status = "REFUSED"
            elif age > ctx.stale_days:
                status, hit = "ABSTAIN", None
            else:
                status = "OK"
            out = {"contract_version": "v3", "instrument_id": instrument_id, "horizon_days": horizon_days,
                   "data_date": block["data_date"], "status": status, "forecast": hit,
                   "disclaimer": "Forecasts the size of price moves, not their direction. Not investment advice."}
            ctx.audit.append("volatility", out)
            return out

    @app.get("/v3/health")
    def get_health():
        h = ctx.killswitch.effective(ctx.health_fn())
        return {**h.to_dict(), "model_version": ctx.version, "manual_switch": ctx.killswitch.manual()}

    @app.post("/v3/admin/kill", dependencies=[Depends(admin)])
    def kill(body: TripRequest):
        state = ctx.killswitch.trip(body.reason, body.by)
        ctx.audit.append("kill_switch", {"action": "trip", **body.model_dump()})
        return state

    @app.post("/v3/admin/reset", dependencies=[Depends(admin)])
    def reset(body: ResetRequest):
        state = ctx.killswitch.reset(body.by)
        ctx.audit.append("kill_switch", {"action": "reset", **body.model_dump()})
        return state

    @app.get("/v3/audit/verify")
    def audit_verify():
        ok, bad = ctx.audit.verify()
        return {"ok": ok, "first_bad_seq": bad, "entries": len(ctx.audit.read())}

    @app.post("/v3/audit/replay/{seq}")
    def audit_replay(seq: int):
        hit = [e for e in ctx.audit.read() if e["seq"] == seq and e["kind"] == "analyse"]
        if not hit:
            raise HTTPException(404, "no analyse entry with that seq")
        return replay(hit[0], ctx)

    @app.get("/v3/dashboard", response_class=HTMLResponse)
    def dashboard():
        h = ctx.killswitch.effective(ctx.health_fn())
        ok, _ = ctx.audit.verify()
        return render_dashboard(h.to_dict(), psi_provider(), ctx.meta, ctx.audit.read(), ok, metrics)

    return app
