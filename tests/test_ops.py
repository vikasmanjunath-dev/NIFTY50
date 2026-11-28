"""Phase 4 tests: audit log, kill switch/health, registry, /v3/analyse core, replay, API."""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nifty_analyser.ops.api import create_app
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.kill_switch import AMBER, GREEN, RED, Health, KillSwitch, Thresholds, compute_live_health, evaluate_health
from nifty_analyser.ops.registry import RegistryError, load_model, save_model
from nifty_analyser.ops.service import AnalyseRequest, FutureAsOfError, ServingContext, TTLCached, analyse, replay

IST = "Asia/Kolkata"
NOW = pd.Timestamp("2026-03-03 18:00", tz=IST)


class FakeModel:
    def predict_proba_up(self, X):
        return 1 / (1 + np.exp(-X[:, 0]))


def _features():
    rows = []
    for iid, start, end in (("NSE:RELIANCE", "2022-01-03", "2026-03-03"), ("NSE:IOC", "2022-01-03", "2022-06-30")):
        d = pd.bdate_range(start, end, tz=IST)
        rows.append(pd.DataFrame({"instrument_id": iid, "event_time": d, "available_time": d + pd.Timedelta(hours=15, minutes=30),
                                  "f0": 1.5, "f1": 0.1, "trend_regime": "Uptrend", "vol_regime": "LowVol"}))
    return pd.concat(rows, ignore_index=True)


def mk(tmp_path, health=Health(GREEN), features=None, model=None, **kw):
    return ServingContext(features=_features() if features is None else features, cols=["f0", "f1"], model=model or FakeModel(),
                          meta={"version": "v-test", "disclaimer": "no demonstrated edge", "feature_cols": ["f0", "f1"], "trained_through": "2025-01-01"},
                          killswitch=KillSwitch(tmp_path / "ks.json"), audit=AuditLog(tmp_path / "audit.jsonl"),
                          health_fn=lambda: health, now_fn=lambda: NOW, **kw)


# ---------------- audit ----------------
def test_audit_chain_verifies_and_detects_edit_deletion_and_reorder(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(5):
        log.append("x", {"i": i})
    assert log.verify() == (True, None)
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    (tmp_path / "a.jsonl").write_text("\n".join(lines).replace('"i":2', '"i":9') + "\n")
    assert log.verify() == (False, 3)
    (tmp_path / "a.jsonl").write_text("\n".join(lines[:2] + lines[3:]) + "\n")
    assert log.verify()[0] is False
    (tmp_path / "a.jsonl").write_text("\n".join([lines[1], lines[0], *lines[2:]]) + "\n")
    assert log.verify()[0] is False


def test_audit_resumes_chain_after_reopen_and_is_thread_safe(tmp_path):
    p = tmp_path / "a.jsonl"
    AuditLog(p).append("x", {})
    log = AuditLog(p)
    threads = [threading.Thread(target=lambda: [log.append("t", {"n": 1}) for _ in range(20)]) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert log.verify() == (True, None) and len(log.read()) == 81


# ---------------- health / kill switch ----------------
def _psi(vals):
    return pd.DataFrame({"feature": [f"f{i}" for i in range(len(vals))], "psi": vals})


def test_health_rules():
    ok = _psi([0.01] * 20)
    assert evaluate_health(1.0, ok).state == GREEN
    assert evaluate_health(4.5, ok).state == AMBER and evaluate_health(8, ok).state == RED and evaluate_health(None, ok).state == RED
    assert evaluate_health(1, _psi([0.3] * 4 + [0.0] * 16)).state == AMBER      # 20% significant
    assert evaluate_health(1, _psi([0.3] * 10 + [0.0] * 10)).state == RED       # 50% significant
    assert evaluate_health(1, _psi([0.0] * 3)).state == AMBER and evaluate_health(1, None).state == AMBER   # can't assess != healthy


def test_manual_trip_persists_overrides_green_and_needs_explicit_reset(tmp_path):
    ks = KillSwitch(tmp_path / "ks.json")
    assert ks.effective(Health(GREEN)).state == GREEN
    ks.trip("bad data vendor", "ops")
    assert KillSwitch(tmp_path / "ks.json").effective(Health(GREEN)).state == RED     # survives restart
    assert "bad data vendor" in KillSwitch(tmp_path / "ks.json").effective(Health(GREEN)).reasons[0]
    ks.reset("ops")
    assert ks.effective(Health(GREEN)).state == GREEN


def test_live_health_flags_drift_and_staleness_and_refuses_tiny_samples():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2023-01-02", periods=400, tz=IST)
    d = days.repeat(30)                                   # 30 instruments per day, like the real panel
    cols = [f"f{i}" for i in range(12)]
    f = pd.concat([pd.DataFrame({"event_time": d, "available_time": d + pd.Timedelta(hours=15)}),
                   pd.DataFrame(rng.normal(size=(len(d), 12)), columns=cols)], axis=1)
    now = days[-1] + pd.Timedelta(days=1)
    h, t = compute_live_health(f, cols, days[250], now)
    assert h.state == GREEN and len(t) == 12
    drifted = f.copy()
    drifted.loc[drifted["event_time"] > now - pd.Timedelta(days=60), cols] += 4
    assert compute_live_health(drifted, cols, days[250], now)[0].state == RED
    assert compute_live_health(f, cols, days[250], now + pd.Timedelta(days=10))[0].state == RED
    tiny = f[f["event_time"] > days[-6]]
    tiny = pd.concat([f[f["event_time"] <= days[250]], tiny])
    h2, t2 = compute_live_health(tiny, cols, days[250], now)
    assert (t2["status"] == "too few current rows").all() and h2.state == AMBER and "could not be assessed" in h2.reasons[0]


def test_ttl_cache_computes_once_within_window():
    calls, t = [], [0.0]
    c = TTLCached(lambda: calls.append(1) or Health(GREEN), 10, clock=lambda: t[0])
    c(), c()
    t[0] = 11
    c()
    assert len(calls) == 2


# ---------------- registry ----------------
def test_registry_roundtrip_and_tamper_refused(tmp_path):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), [0, 0, 1, 1])
    v = save_model(m, {"trained_through": "2026-01-01"}, tmp_path)
    loaded, meta = load_model(tmp_path)
    assert meta["version"] == v and loaded.predict(np.array([[3.0]]))[0] == 1
    (tmp_path / f"model_{v}.joblib").write_bytes(b"tampered")
    with pytest.raises(RegistryError, match="integrity"):
        load_model(tmp_path)
    with pytest.raises(RegistryError):
        load_model(tmp_path / "empty")


# ---------------- analyse ----------------
def req(iid="NSE:RELIANCE", as_of=None):
    return AnalyseRequest(instrument_id=iid, as_of=as_of)


def test_ok_signal_and_audit_entry(tmp_path):
    ctx = mk(tmp_path)
    r = analyse(req(), ctx)
    assert r.status == "OK" and r.signal.direction == "UP" and r.signal.prob_up == pytest.approx(1 / (1 + np.exp(-1.5)), abs=1e-5)
    assert r.in_universe is True and r.regime == {"trend": "Uptrend", "vol": "LowVol"} and r.disclaimer == "no demonstrated edge"
    e = ctx.audit.read()[-1]
    assert e["seq"] == r.audit_seq and e["hash"] == r.audit_hash and e["payload"]["model_version"] == "v-test"
    assert ctx.audit.verify()[0]


def test_low_confidence_abstains(tmp_path):
    f = _features().assign(f0=0.02)
    r = analyse(req(), mk(tmp_path, features=f))
    assert r.status == "ABSTAIN" and r.signal is None and "confidence" in r.warnings[0]


def test_point_in_time_never_uses_a_row_not_yet_available(tmp_path):
    f = _features()
    late = f[f.instrument_id == "NSE:RELIANCE"].iloc[[-1]].assign(f0=-9.0, available_time=NOW - pd.Timedelta(minutes=5))
    ctx = mk(tmp_path, features=pd.concat([f, late], ignore_index=True))
    before = analyse(req(as_of=(NOW - pd.Timedelta(hours=1)).to_pydatetime()), ctx)   # the -9.0 row is published 5 min before NOW
    after = analyse(req(), ctx)
    assert before.signal.direction == "UP" and after.signal.direction == "DOWN"


def test_refused_cases_and_future_as_of(tmp_path):
    ctx = mk(tmp_path)
    assert analyse(req("NSE:NOPE"), ctx).status == "REFUSED"
    early = analyse(req(as_of=pd.Timestamp("2000-01-01", tz=IST).to_pydatetime()), ctx)
    assert early.status == "REFUSED" and early.in_universe is True and "no feature data" in early.warnings[0]
    pre_join = analyse(req("NSE:TRENT", pd.Timestamp("2023-06-01", tz=IST).to_pydatetime()), ctx)   # joined the index 2024-09-30
    assert pre_join.status == "REFUSED" and pre_join.in_universe is False and "not a Nifty 50 constituent" in pre_join.warnings[0]
    with pytest.raises(FutureAsOfError):
        analyse(req(as_of=(NOW + pd.Timedelta(days=2)).to_pydatetime()), ctx)


def test_membership_warning_and_stale_abstain(tmp_path):
    ctx = mk(tmp_path)
    member = analyse(req("NSE:IOC", pd.Timestamp("2022-03-01 18:00", tz=IST).to_pydatetime()), ctx)
    nonmember = analyse(req("NSE:IOC", pd.Timestamp("2022-06-29 18:00", tz=IST).to_pydatetime()), ctx)
    stale = analyse(req("NSE:IOC"), ctx)   # IOC data ends 2022-06-30; now is 2026
    assert member.in_universe is True and member.status == "OK"
    assert nonmember.in_universe is False and any("not a Nifty 50 constituent" in w for w in nonmember.warnings)
    assert stale.status == "ABSTAIN" and any("older than" in w for w in stale.warnings)


def test_kill_switch_degrades_with_no_signal_but_is_still_audited(tmp_path):
    ctx = mk(tmp_path)
    ctx.killswitch.trip("test", "ops")
    r = analyse(req(), ctx)
    assert r.status == "DEGRADED" and r.signal is None and r.health["state"] == RED
    assert ctx.audit.read()[-1]["payload"]["status"] == "DEGRADED"
    ctx2 = mk(tmp_path / "b", health=Health(RED, ["data stale"]))
    assert analyse(req(), ctx2).status == "DEGRADED"


def test_amber_health_still_serves_with_warnings(tmp_path):
    r = analyse(req(), mk(tmp_path, health=Health(AMBER, ["drift rising"])))
    assert r.status == "OK" and any("AMBER" in w for w in r.warnings)


# ---------------- replay ----------------
def test_replay_matches_then_detects_edited_features_and_changed_model(tmp_path):
    ctx = mk(tmp_path)
    analyse(req(), ctx)
    entry = ctx.audit.read()[-1]
    assert replay(entry, ctx)["match"] is True
    edited = _features().assign(f0=0.9)
    bad = replay(entry, mk(tmp_path / "e", features=edited))
    assert bad["match"] is False and bad["feature_hash_match"] is False
    other = mk(tmp_path / "m")
    other.meta["version"] = "v-other"
    assert replay(entry, other)["match"] is False
    ctx.killswitch.trip("x", "ops")
    analyse(req(), ctx)
    assert replay(ctx.audit.read()[-1], ctx)["match"] is None


# ---------------- API ----------------
def client(tmp_path, **kw):
    return TestClient(create_app(mk(tmp_path, **kw)))


def test_api_analyse_contract_and_validation(tmp_path):
    c = client(tmp_path)
    j = c.post("/v3/analyse", json={"instrument_id": "NSE:RELIANCE"}).json()
    assert j["contract_version"] == "v3" and j["status"] == "OK" and j["signal"]["direction"] == "UP" and j["audit_seq"] == 1
    assert c.post("/v3/analyse", json={"instrument_id": "reliance; DROP"}).status_code == 422
    assert c.post("/v3/analyse", json={"instrument_id": "NSE:RELIANCE", "as_of": "2031-01-01T00:00:00+05:30"}).status_code == 422


def test_admin_disabled_without_token_and_requires_valid_token(tmp_path, monkeypatch):
    monkeypatch.delenv("STM_ADMIN_TOKEN", raising=False)
    c = client(tmp_path)
    assert c.post("/v3/admin/kill", json={"reason": "x", "by": "me"}).status_code == 403
    monkeypatch.setenv("STM_ADMIN_TOKEN", "s3cret")
    assert c.post("/v3/admin/kill", json={"reason": "x", "by": "me"}).status_code == 401
    assert c.post("/v3/admin/kill", json={"reason": "x", "by": "me"}, headers={"X-Admin-Token": "wrong"}).status_code == 401
    h = {"X-Admin-Token": "s3cret"}
    assert c.post("/v3/admin/kill", json={"reason": "vendor outage", "by": "ops"}, headers=h).status_code == 200
    assert c.post("/v3/analyse", json={"instrument_id": "NSE:RELIANCE"}).json()["status"] == "DEGRADED"
    assert c.get("/v3/health").json()["state"] == RED
    assert c.post("/v3/admin/reset", json={"by": "ops"}, headers=h).status_code == 200
    assert c.post("/v3/analyse", json={"instrument_id": "NSE:RELIANCE"}).json()["status"] == "OK"
    assert c.get("/v3/audit/verify").json() == {"ok": True, "first_bad_seq": None, "entries": 4}


def test_api_replay_dashboard_and_xss_escaping(tmp_path):
    c = client(tmp_path, health=Health(AMBER, ["<script>alert(1)</script>"]))
    seq = c.post("/v3/analyse", json={"instrument_id": "NSE:RELIANCE"}).json()["audit_seq"]
    assert c.post(f"/v3/audit/replay/{seq}").json()["match"] is True
    assert c.post("/v3/audit/replay/999").status_code == 404
    html = c.get("/v3/dashboard").text
    assert "AMBER" in html and "Chain integrity: <b>OK</b>" in html and "<script>alert(1)" not in html and "&lt;script&gt;" in html


@pytest.mark.parametrize("when", ["0001-01-01", "9999-12-31"])
def test_out_of_range_as_of_is_a_validation_error_not_a_crash(when):
    from pydantic import ValidationError

    from nifty_analyser.ops.evidence_service import EvidenceRequest
    from nifty_analyser.ops.service import AnalyseRequest

    for model, fields in ((AnalyseRequest, {"instrument_id": "NSE:INFY"}), (EvidenceRequest, {"query": "dividend"})):
        with pytest.raises(ValidationError):
            model(**fields, as_of=when)
        assert model(**fields, as_of="2026-01-02T10:00:00").as_of.year == 2026


def test_audit_log_survives_a_torn_line_and_a_second_writer(tmp_path):
    from nifty_analyser.ops.audit import AuditLog

    path = tmp_path / "a.jsonl"
    first, second = AuditLog(path), AuditLog(path)  # two processes hold the same file
    first.append("a", {})
    second.append("b", {})
    first.append("c", {})
    assert [e["seq"] for e in first.read()] == [1, 2, 3] and first.verify() == (True, None)
    path.open("a").write('{"seq": 4, "tor')  # crash mid-write
    assert [e["kind"] for e in AuditLog(path).read()] == ["a", "b", "c"]
    assert AuditLog(path).verify() == (False, 4)  # damage is reported, not hidden
    AuditLog(path).append("d", {})
    assert [e["kind"] for e in AuditLog(path).read()] == ["a", "b", "c", "d"]
    assert AuditLog(path).verify()[0] is False


def test_only_one_refresh_at_a_time(tmp_path):
    from nifty_analyser.ops.refresh import single_instance

    with single_instance(tmp_path / "r.lock") as first:
        with single_instance(tmp_path / "r.lock") as second:
            assert first and not second
    with single_instance(tmp_path / "r.lock") as third:
        assert third
