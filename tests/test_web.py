import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nifty_analyser.ops.api import create_app
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.kill_switch import Health, KillSwitch
from nifty_analyser.ops.web import WEB_DIR, WebData, attach_web


class Ctx:
    stale_days, version = 5, "test"

    def __init__(self, tmp):
        self.killswitch, self.audit = KillSwitch(tmp / "k.json"), AuditLog(tmp / "a.jsonl")
        self.now = pd.Timestamp("2026-01-10", tz="Asia/Kolkata")

    def now_fn(self):
        return self.now

    def health_fn(self):
        return Health("GREEN")


@pytest.fixture
def site(tmp_path):
    forecast = {"prob_above_usual": 0.7, "forecast_vol_annualised": 0.2, "usual_vol_annualised": 0.18}
    (tmp_path / "volatility_latest.json").write_text(json.dumps({"5": {"data_date": "2026-01-09", "forecasts": {"NSE:INFY": forecast}}}))
    pd.DataFrame({"instrument_id": ["NSE:INFY", "NSE:TCS"], "available_time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                  "category": ["Dividend", "Updates"], "text": ["<script>x</script> dividend declared", "other news"],
                  "attachment_url": ["javascript:alert(1)", "https://example.com/a.pdf"]}).to_parquet(tmp_path / "announcements_nifty50.parquet")
    ctx = Ctx(tmp_path)
    return ctx, TestClient(attach_web(create_app(ctx), ctx, WebData(ctx, tmp_path, tmp_path, tmp_path)))


def test_overview_serves_then_withholds_when_stale_or_halted(site):
    ctx, client = site
    body = client.get("/api/overview").json()
    assert body["status"] == "OK" and body["rows"][0]["symbol"] == "INFY" and body["rows"][0]["h5"]["prob_above_usual"] == 0.7
    ctx.now = pd.Timestamp("2026-02-01", tz="Asia/Kolkata")
    assert client.get("/api/overview").json() == {**body, "status": "STALE", "data_date": None, "rows": []}
    ctx.killswitch.trip("test", "t")
    assert client.get("/api/overview").json()["status"] == "HALTED"


def test_filings_filter_and_drop_unsafe_links(site):
    _, client = site
    hits = client.get("/api/filings", params={"q": "infy dividend"}).json()
    assert [h["symbol"] for h in hits] == ["INFY"] and hits[0]["url"] is None
    assert client.get("/api/filings", params={"q": "(["}).json() == []  # searched literally, not as a pattern
    assert client.get("/api/filings").json()[0]["url"] == "https://example.com/a.pdf"


def test_pages_static_files_and_system(site):
    _, client = site
    assert "Nifty Analyser" in client.get("/").text
    for name in ("app.js", "app.css", "manifest.webmanifest", "icon.svg", "icon-180.png"):
        assert client.get(f"/static/{name}").status_code == 200
    worker = client.get("/sw.js")
    assert worker.headers["content-type"].startswith("text/javascript") and "/api/" not in worker.text.split("const SHELL")[1].split(";")[0]
    assert client.get("/api/system").json()["audit_ok"] and client.get("/api/model").json()["rounds"]
    assert client.get("/api/stock/NSE:NOPE").status_code == 404
    assert "innerHTML =" not in (WEB_DIR / "app.js").read_text()


def test_portfolio_risk_and_input_validation(site, monkeypatch, tmp_path):
    import numpy as np

    from nifty_analyser.ops import web

    ctx, client = site
    t = pd.date_range("2025-01-01", periods=300, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300)))
    monkeypatch.setattr(web, "load_cached", lambda i: pd.DataFrame({"event_time": t, "close": close, "quality_flag": "PASS"}))
    body = client.get("/api/portfolio", params={"holdings": "NSE:INFY:1000,NSE:INFY:3000"}).json()
    assert body["total"] == 4000 and body["volatility"] == pytest.approx(0.2) and body["parts"][0]["risk_share"] == pytest.approx(1.0)
    assert body["typical_move"] == pytest.approx(0.2 * (5 / 252) ** 0.5 * 4000)
    for bad in ("NSE:INFY:-5", "NSE:INFY:nan", "garbage", "NSE:ZZZ:10"):
        assert client.get("/api/portfolio", params={"holdings": bad}).status_code == 422
    assert body["large_move"] == pytest.approx(1.645 / 1.0 * body["typical_move"]) and body["large_move_held"] is None
    (tmp_path / "portfolio_backtest.json").write_text(json.dumps({"5": {"multiplier_for_90pct": 2.0, "second_half_within_calibrated": 0.89}}))
    again = client.get("/api/portfolio", params={"holdings": "NSE:INFY:4000"}).json()
    assert again["large_move"] == pytest.approx(2 * again["typical_move"]) and again["large_move_held"] == 0.89
    ctx.killswitch.trip("test", "t")
    assert client.get("/api/portfolio", params={"holdings": "NSE:INFY:1"}).json() == {"status": "HALTED"}
