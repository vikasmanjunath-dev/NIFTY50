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
    assert client.get("/api/system").json()["audit_ok"] and client.get("/api/model").json()["rounds"]
    assert client.get("/api/stock/NSE:NOPE").status_code == 404
    assert "innerHTML =" not in (WEB_DIR / "app.js").read_text()
