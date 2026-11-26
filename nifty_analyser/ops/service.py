"""``/v3/analyse`` contract and serving core (Blueprint §5.4).

Guarantees: (1) point-in-time -- only the newest feature row whose ``available_time <= as_of`` is
used; a future ``as_of`` is rejected; (2) fail-safe -- kill switch RED returns DEGRADED with no
signal, low confidence / stale data returns ABSTAIN, unknown or data-less instruments REFUSED;
(3) every call is audited (incl. degraded ones) with the model version and a feature digest, and
any signal can be deterministically replayed (``replay``) from its audit entry."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from nifty_analyser.data.index_membership import instrument_ids_as_of
from nifty_analyser.data.instrument_master import NIFTY50
from nifty_analyser.data.timestamps import to_ist
from nifty_analyser.ops.audit import AuditLog, digest
from nifty_analyser.ops.kill_switch import RED, Health, KillSwitch

CONTRACT_VERSION = "v3"


class FutureAsOfError(ValueError):
    pass


class AnalyseRequest(BaseModel):
    instrument_id: str = Field(pattern=r"^NSE:[A-Z0-9&\-]+$", examples=["NSE:RELIANCE"])
    as_of: datetime | None = Field(None, description="Decision time (default now). Naive = IST. Future values are rejected.")
    request_id: str | None = Field(None, max_length=64)


class Signal(BaseModel):
    prob_up: float
    direction: Literal["UP", "DOWN"]
    confidence: float
    horizon_days: int


class AnalyseResponse(BaseModel):
    contract_version: Literal["v3"] = "v3"
    request_id: str
    instrument_id: str
    as_of: datetime
    data_as_of: datetime | None = None
    model_version: str
    status: Literal["OK", "ABSTAIN", "DEGRADED", "REFUSED"]
    signal: Signal | None = None
    regime: dict[str, str | None] = {}
    in_universe: bool | None = None
    health: dict = {}
    warnings: list[str] = []
    disclaimer: str = ""
    audit_seq: int | None = None
    audit_hash: str | None = None


class TTLCached:
    """Cache an expensive health computation for ``seconds``."""

    def __init__(self, fn: Callable[[], Health], seconds: float = 300.0, clock=time.monotonic):
        self.fn, self.seconds, self.clock, self._v, self._t = fn, seconds, clock, None, None

    def __call__(self) -> Health:
        if self._v is None or self.clock() - self._t >= self.seconds:
            self._v, self._t = self.fn(), self.clock()
        return self._v


@dataclass
class ServingContext:
    features: pd.DataFrame                 # unlabelled point-in-time feature rows (incl. the newest day)
    cols: list[str]
    model: object                          # has predict_proba_up(np.ndarray)
    meta: dict
    killswitch: KillSwitch
    audit: AuditLog
    health_fn: Callable[[], Health]
    abstain_threshold: float = 0.10
    stale_days: int = 5
    horizon_days: int = 5
    now_fn: Callable[[], pd.Timestamp] = lambda: to_ist(pd.Timestamp.now(tz="Asia/Kolkata"))
    _by_inst: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        f = self.features.sort_values(["instrument_id", "available_time"])
        self._by_inst = {k: g.reset_index(drop=True) for k, g in f.groupby("instrument_id")}

    @property
    def version(self) -> str:
        return self.meta.get("version", "unversioned")


def _compute(req: AnalyseRequest, ctx: ServingContext, as_of: pd.Timestamp) -> dict:
    """The deterministic, health-independent part: same inputs + same model => same output."""
    out = {"status": "OK", "signal": None, "regime": {}, "in_universe": None, "warnings": [], "data_as_of": None, "feature_hash": None}
    if req.instrument_id not in NIFTY50:
        return {**out, "status": "REFUSED", "warnings": [f"unknown instrument {req.instrument_id}"]}
    g = ctx._by_inst.get(req.instrument_id)
    pos = -1 if g is None else int(g["available_time"].searchsorted(as_of, side="right")) - 1
    if pos < 0:
        member = req.instrument_id in instrument_ids_as_of(as_of)
        why = ("no feature data available at as_of" if member else
               "instrument was not a Nifty 50 constituent at as_of; serving data exists for index members only")
        return {**out, "status": "REFUSED", "in_universe": member, "warnings": [why]}
    row = g.iloc[pos]
    out["data_as_of"] = row["available_time"]
    out["regime"] = {"trend": row.get("trend_regime"), "vol": row.get("vol_regime")}
    out["in_universe"] = req.instrument_id in instrument_ids_as_of(as_of)
    if not out["in_universe"]:
        out["warnings"].append("instrument was not a Nifty 50 constituent at as_of; the model was trained on index members only")
    vec = row[ctx.cols].to_numpy(dtype=float)
    out["feature_hash"] = digest([None if np.isnan(v) else float(v) for v in vec])
    if (as_of - row["available_time"]) > timedelta(days=ctx.stale_days):
        return {**out, "status": "ABSTAIN", "warnings": [*out["warnings"], f"newest data is older than {ctx.stale_days} days at as_of"]}
    p = float(ctx.model.predict_proba_up(vec.reshape(1, -1))[0])
    conf = abs(p - 0.5) * 2
    if conf < ctx.abstain_threshold:
        return {**out, "status": "ABSTAIN", "warnings": [*out["warnings"], f"confidence {conf:.3f} below abstention threshold {ctx.abstain_threshold}"]}
    out["signal"] = {"prob_up": round(p, 6), "direction": "UP" if p > 0.5 else "DOWN", "confidence": round(conf, 6), "horizon_days": ctx.horizon_days}
    return out


def _signal_digest(core: dict, version: str) -> str:
    keep = {k: core[k] for k in ("status", "signal", "regime", "in_universe", "data_as_of", "feature_hash")}
    return digest({**keep, "model_version": version})


def analyse(req: AnalyseRequest, ctx: ServingContext) -> AnalyseResponse:
    now = ctx.now_fn()
    as_of = to_ist(req.as_of) if req.as_of else now
    if as_of > now + timedelta(minutes=1):
        raise FutureAsOfError("as_of is in the future; a decision can only use information available now")
    rid = req.request_id or uuid.uuid4().hex[:16]
    health = ctx.killswitch.effective(ctx.health_fn())
    if health.state == RED:
        core = {"status": "DEGRADED", "signal": None, "regime": {}, "in_universe": None, "data_as_of": None, "feature_hash": None,
                "warnings": ["serving is halted by the kill switch / health checks; no signal produced", *health.reasons]}
    else:
        core = _compute(req, ctx, as_of)
        if health.state != "GREEN":
            core["warnings"] = [*core["warnings"], *[f"health {health.state}: {r}" for r in health.reasons]]
    sig_digest = _signal_digest(core, ctx.version)
    entry = ctx.audit.append("analyse", {
        "request_id": rid, "instrument_id": req.instrument_id, "as_of": as_of.isoformat(), "model_version": ctx.version,
        "status": core["status"], "signal": core["signal"], "regime": core["regime"], "in_universe": core["in_universe"],
        "data_as_of": core["data_as_of"], "feature_hash": core["feature_hash"], "signal_digest": sig_digest,
        "health": health.to_dict(), "warnings": core["warnings"]})
    return AnalyseResponse(
        request_id=rid, instrument_id=req.instrument_id, as_of=as_of, data_as_of=core["data_as_of"], model_version=ctx.version,
        status=core["status"], signal=core["signal"], regime=core["regime"], in_universe=core["in_universe"],
        health=health.to_dict(), warnings=core["warnings"], disclaimer=ctx.meta.get("disclaimer", ""),
        audit_seq=entry["seq"], audit_hash=entry["hash"])


def replay(entry: dict, ctx: ServingContext) -> dict:
    """Re-derive an audited signal from current stores. Matches only if the model version, the
    feature row and the output are all identical -- a changed model or edited features is reported."""
    p = entry["payload"]
    if p["status"] == "DEGRADED":
        return {"match": None, "note": "no signal was computed (serving halted); nothing to replay"}
    req = AnalyseRequest(instrument_id=p["instrument_id"])
    core = _compute(req, ctx, to_ist(pd.Timestamp(p["as_of"])))
    now_digest = _signal_digest(core, ctx.version)
    return {"match": now_digest == p["signal_digest"], "logged_model_version": p["model_version"], "current_model_version": ctx.version,
            "feature_hash_match": core["feature_hash"] == p["feature_hash"], "recomputed_signal": core["signal"], "logged_signal": p["signal"]}
