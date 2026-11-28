import json
import shutil
import subprocess
import sys
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nifty_analyser.ops.api import create_app
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.kill_switch import Health, KillSwitch
from nifty_analyser.ops.refresh import Refresher
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
    return ctx, TestClient(attach_web(create_app(ctx), ctx, WebData(ctx, tmp_path, tmp_path, tmp_path), Refresher([sys.executable, "-c", "print('ok')"])),
                           base_url="http://127.0.0.1")


def test_overview_serves_then_withholds_when_stale_or_halted(site):
    ctx, client = site
    body = client.get("/api/overview").json()
    assert body["age_days"] == 1 and body["status"] == "OK" and body["rows"][0]["symbol"] == "INFY" and body["rows"][0]["h5"]["prob_above_usual"] == 0.7
    ctx.now = pd.Timestamp("2026-02-01", tz="Asia/Kolkata")
    assert client.get("/api/overview").json() == {**body, "status": "STALE", "data_date": None, "rows": [], "age_days": None}
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
    for path in ("/", "/static/app.js", "/api/overview"):
        assert client.get(path).headers["cache-control"] == "no-cache"
    worker = client.get("/sw.js")
    assert worker.headers["content-type"].startswith("text/javascript") and "/api/" not in worker.text.split("const SHELL")[1].split(";")[0]
    assert client.get("/api/system").json()["audit_ok"] and client.get("/api/model").json()["rounds"]
    assert client.get("/api/stock/NSE:NOPE").status_code == 404
    script = (WEB_DIR / "app.js").read_text()
    assert "innerHTML =" not in script and ".flat()" not in script  # shallow flattening silently drops nested sections


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


def test_ask_page_numbers_sources_like_the_prompt_does():
    import inspect

    from nifty_analyser.documents import llm

    assert "sorted(pack.evidence_items, key=lambda e: e.publication_time, reverse=True)" in inspect.getsource(llm)
    assert "String(b.publication_time).localeCompare(String(a.publication_time))" in (WEB_DIR / "app.js").read_text()


def test_refresh_only_from_the_app_itself_and_foreign_hosts_are_rejected(site):
    ctx, client = site
    mine = {"X-Requested-With": "nifty-web"}
    assert client.post("/api/refresh").status_code == 403
    assert client.post("/api/refresh", headers={**mine, "Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/overview", headers={"Host": "evil.example"}).status_code == 400
    assert client.post("/api/refresh", headers={**mine, "Origin": "http://127.0.0.1"}).status_code == 200
    for _ in range(100):
        state = client.get("/api/refresh").json()
        if not state["running"]:
            break
        time.sleep(0.05)
    assert state == {"running": False, "ok": True, "message": "ok"}
    assert ctx.audit.read()[-1]["kind"] == "refresh"


def test_refresher_runs_one_at_a_time_and_reports_failure():
    r = Refresher([sys.executable, "-c", "import time,sys; time.sleep(0.3); sys.exit('boom')"])
    assert r.start() and not r.start()
    for _ in range(100):
        if not r.state["running"]:
            break
        time.sleep(0.05)
    assert r.state["ok"] is False and r.state["message"] == "boom"


def test_track_record_and_reliability(site, tmp_path):
    _, client = site
    assert client.get("/api/model").json()["reliability"] == {}
    n = 400
    pd.DataFrame({"instrument_id": "NSE:INFY", "event_time": pd.date_range("2025-01-01", periods=n), "y": [1, 0, 1, 1] * (n // 4),
                  "lv": -3.5, "p": 0.75, "r": -3.5}).to_parquet(tmp_path / "vol_oos_h5.parquet")
    assert client.get("/api/model").json()["reliability"]["5"] == [{"said": 0.75, "happened": 0.75, "n": n}]
    from nifty_analyser.ops.web import WebData

    record = WebData(None, tmp_path, tmp_path, tmp_path).track_record("NSE:INFY")["5"]
    assert len(record["dates"]) == 250 and record["forecast"] == record["actual"]


def test_upcoming_meetings_show_only_published_future_dates(site, tmp_path):
    _, client = site
    ist = lambda *d: pd.to_datetime(list(d)).tz_localize("Asia/Kolkata")
    pd.DataFrame({"instrument_id": "NSE:INFY", "meeting_date": ist("2026-01-20", "2026-01-02", "2026-01-25", "2026-03-30"),
                  "available_time": ist("2026-01-05", "2025-12-20", "2026-01-12", "2026-01-05"), "is_results": True}).to_parquet(tmp_path / "board_meetings.parquet")
    # "now" is 10 Jan: the past meeting, the not-yet-published one and the one beyond 30 days are all excluded
    assert [u["date"] for u in client.get("/api/overview").json()["upcoming"]] == ["2026-01-20"]


def test_changes_compare_with_the_previous_recorded_date(site, tmp_path):
    _, client = site
    assert client.get("/api/overview").json()["changes"] == {}
    ledger = AuditLog(tmp_path / "vol_ledger.jsonl")
    for day, prob in (("2026-01-07", 0.2), ("2026-01-08", 0.5), ("2026-01-09", 0.9)):  # the same-day entry must be ignored
        ledger.append("vol_forecast", {"date": day, "horizon_days": 5, "forecasts": [{"instrument_id": "NSE:INFY", "prob": prob}]})
    change = client.get("/api/overview").json()["changes"]["5"]
    assert change["since"] == "2026-01-08" and change["rows"] == [{"id": "NSE:INFY", "symbol": "INFY", "now": 0.7, "before": 0.5}]


def test_filings_filter_by_category(site):
    _, client = site
    assert sorted(client.get("/api/filings/categories").json()) == ["Dividend", "Updates"]
    assert [h["symbol"] for h in client.get("/api/filings", params={"category": "Updates"}).json()] == ["TCS"]


def test_saved_forecasts_are_pending_then_graded_and_backed_up(site, tmp_path):
    import io
    import zipfile

    ctx, client = site
    ledger = AuditLog(tmp_path / "vol_ledger.jsonl")
    for day in ("2026-01-01", "2026-01-08"):
        ledger.append("vol_forecast", {"date": day, "horizon_days": 5, "forecasts": [{"instrument_id": "NSE:INFY", "prob": 0.8, "med_rv": 0.001}]})
    days = pd.date_range("2026-01-01", periods=8, freq="B", tz="Asia/Kolkata")
    bars = pd.DataFrame({"event_time": days, "close": [100, 110, 100, 110, 100, 110, 100, 110], "quality_flag": "PASS"})
    new, old = WebData(ctx, tmp_path, tmp_path, tmp_path).forward("NSE:INFY", bars)
    assert old == {"date": "2026-01-01", "horizon": 5, "prob": 0.8, "above_usual": True, "inside_range": None, "days_left": 0}
    assert new["above_usual"] is None and new["days_left"] == 3
    response = client.get("/api/backup")
    z = zipfile.ZipFile(io.BytesIO(response.content))
    assert "attachment" in response.headers["content-disposition"]
    assert z.read("vol_ledger.jsonl") == (tmp_path / "vol_ledger.jsonl").read_bytes()
    assert json.loads(z.read("chain_status.json")) == {"vol_ledger.jsonl": True}


def test_freshness_flags_missing_and_lagging_sources(site, tmp_path):
    ctx, _ = site  # "now" is 10 Jan 2026
    ist = lambda d: pd.to_datetime([d]).tz_localize("Asia/Kolkata")
    pd.DataFrame({"event_time": ist("2026-01-09")}).to_parquet(tmp_path / "NSE_NIFTY50_INDEX.parquet")
    pd.DataFrame({"event_time": ist("2025-12-01")}).to_parquet(tmp_path / "macro_india_vix.parquet")
    rows = {r["source"]: r for r in WebData(ctx, tmp_path, tmp_path, tmp_path).freshness()}
    assert (rows["Prices"]["age_days"], rows["Prices"]["late"]) == (1, False)
    assert rows["India VIX"]["late"] and rows["Board meetings"]["newest"] is None and rows["Board meetings"]["late"]
    assert rows["Company filings"]["newest"] == "2026-01-02" and rows["Company filings"]["late"]


def test_filings_for_several_instruments(site):
    _, client = site
    both = client.get("/api/filings", params={"instrument": "NSE:INFY,NSE:TCS"}).json()
    assert [h["symbol"] for h in both] == ["TCS", "INFY"]
    assert [h["symbol"] for h in client.get("/api/filings", params={"instrument": "NSE:INFY"}).json()] == ["INFY"]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_holdings_paste_parser():
    source = (WEB_DIR / "app.js").read_text()
    function = source[source.index("function parseHoldings"): source.index("async function portfolio")]
    script = function + """
const r = parseHoldings("infy, 1,00,000\\nTCS 500\\nTCS:250\\nNOPE 5\\nINFY -3\\n\\nM&M, ₹ 10.5", new Map([["INFY","a"],["TCS","b"],["M&M","c"]]));
console.log(JSON.stringify([[...r.found], r.skipped]));"""
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [[["a", 100000], ["b", 750], ["c", 10.5]], ["NOPE 5", "INFY -3"]]


def test_results_moves_and_sector_peers(site, tmp_path):
    ctx, _ = site
    days = pd.date_range("2025-01-01", periods=100, freq="B", tz="Asia/Kolkata")
    close = [100.0] * 100
    close[51:] = [110.0] * 49  # jumps the session after the meeting on day 50
    bars = pd.DataFrame({"event_time": days, "close": close, "quality_flag": "PASS"})
    data = WebData(ctx, tmp_path, tmp_path, tmp_path)
    assert data.results_moves("NSE:INFY", bars) is None
    pd.DataFrame({"instrument_id": "NSE:INFY", "meeting_date": [days[50], days[99]], "is_results": True}).to_parquet(tmp_path / "board_meetings.parquet")
    out = data.results_moves("NSE:INFY", bars)  # the last-day meeting has no following close yet
    assert out["events"] == [{"date": str(days[50].date()), "move": 0.1}] and (out["typical_results"], out["typical_any"]) == (0.1, 0.0)
    blocks = {"5": {"forecasts": {i: {"prob_above_usual": p} for i, p in (("NSE:INFY", 0.9), ("NSE:TCS", 0.4), ("NSE:SBIN", 0.1))}}}
    assert data.peers("NSE:INFY", blocks) == {"5": {"sector_average": 0.4, "stocks": 1}}


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_restored_list_keeps_only_well_formed_entries():
    source = (WEB_DIR / "app.js").read_text()
    function = source[source.index("function readSaved"): source.index("function toggleWatch")]
    script = function + """console.log(JSON.stringify(readSaved(JSON.stringify({watch:["NSE:M&M","<img>",5],
      holdings:[["NSE:TCS",10],["NSE:TCS",-1],["bad",3],["NSE:INFY","9"],7]}))));"""
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == {"watch": ["NSE:M&M"], "holdings": [["NSE:TCS", 10]]}


def test_market_mood_and_unusualness_rank_only_the_past(site, tmp_path):
    ctx, _ = site  # "now" is 10 Jan 2026
    data = WebData(ctx, tmp_path, tmp_path, tmp_path)
    assert data.market() is None
    days = pd.date_range("2025-06-01", "2026-03-01", freq="B", tz="Asia/Kolkata")
    pd.DataFrame({"event_time": days, "available_time": days, "close": range(len(days))}).to_parquet(tmp_path / "macro_india_vix.parquet")
    m = data.market()
    assert m["date"] == "2026-01-09" and m["rank"] == 1.0 and m["history"][-1][0] == "2026-01-09"  # later rows are not visible
    t = pd.date_range("2024-01-01", periods=400, freq="B", tz="Asia/Kolkata")
    close = [100 + (i % 2) * (5 if i >= 370 else 1) for i in range(400)]
    u = data.unusual(pd.DataFrame({"event_time": t, "close": close, "quality_flag": "PASS"}))
    assert u["rank"] > 0.95 and u["windows"] == 380
    assert data.unusual(pd.DataFrame({"event_time": t[:100], "close": close[:100], "quality_flag": "PASS"})) is None


def test_portfolio_pairs_removal_and_range_grading(site, tmp_path, monkeypatch):
    import numpy as np

    from nifty_analyser.ops import web

    ctx, client = site
    f = {"prob_above_usual": 0.5, "forecast_vol_annualised": 0.2, "usual_vol_annualised": 0.2}
    (tmp_path / "volatility_latest.json").write_text(json.dumps({"5": {"data_date": "2026-01-09", "forecasts": {"NSE:INFY": f, "NSE:TCS": f}}}))
    t = pd.date_range("2025-01-01", periods=300, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300)))
    monkeypatch.setattr(web, "load_cached", lambda i: pd.DataFrame({"event_time": t, "close": close, "quality_flag": "PASS"}))
    body = client.get("/api/portfolio", params={"holdings": "NSE:INFY:1,NSE:TCS:1"}).json()  # identical price paths
    assert body["linked"] == [{"a": "INFY", "b": "TCS", "correlation": 1.0}] and [p["without"] for p in body["parts"]] == [pytest.approx(0.2)] * 2
    assert client.get("/api/portfolio", params={"holdings": "NSE:INFY:1"}).json()["parts"][0]["without"] is None
    ledger = AuditLog(tmp_path / "vol_ledger.jsonl")
    ledger.append("vol_forecast", {"date": "2026-01-01", "horizon_days": 1,
                                   "forecasts": [{"instrument_id": "NSE:INFY", "prob": 0.5, "med_rv": 1.0, "low_90": -0.05, "high_90": 0.05}]})
    days = pd.date_range("2026-01-01", periods=2, freq="B", tz="Asia/Kolkata")
    row = WebData(ctx, tmp_path, tmp_path, tmp_path).forward("NSE:INFY", pd.DataFrame({"event_time": days, "close": [100, 110], "quality_flag": "PASS"}))
    assert row[0]["inside_range"] is False


def test_sector_lists_its_stocks(site, monkeypatch):
    _, client = site
    body = client.get("/api/sector/IT").json()
    assert [r["symbol"] for r in body["rows"]] == ["INFY"] and body["correlation"] is None
    assert client.get("/api/sector/Nope").status_code == 404


def test_security_headers_and_no_inline_handlers(site):
    _, client = site
    for path in ("/", "/static/app.js", "/api/overview"):
        h = client.get(path).headers
        assert "script-src 'self'" in h["content-security-policy"] and h["x-content-type-options"] == "nosniff"
    assert " onclick=" not in client.get("/").text


def test_system_lists_recent_activity_without_payloads(site):
    ctx, client = site
    for i in range(12):
        ctx.audit.append(f"k{i}", {"secret": "x"})
    recent = client.get("/api/system").json()["recent"]
    assert [r["kind"] for r in recent][:2] == ["k11", "k10"] and len(recent) == 10 and set(recent[0]) == {"time", "kind"}


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_meetings_are_grouped_by_date_and_past_ones_dropped():
    source = (WEB_DIR / "app.js").read_text()
    function = source[source.index("function byDate"):source.index("async function meetings")]
    items = [{"date": "2026-01-03", "symbol": "B"}, {"date": "2026-01-01", "symbol": "OLD"}, {"date": "2026-01-02", "symbol": "Z"},
             {"date": "2026-01-03", "symbol": "A"}]
    script = f"{function}\nconsole.log(JSON.stringify(byDate({json.dumps(items)}, '2026-01-02').map(([d, l]) => [d, l.map((u) => u.symbol)])))"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [["2026-01-02", ["Z"]], ["2026-01-03", ["A", "B"]]]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_shared_watchlist_link_keeps_only_known_stocks_once():
    source = (WEB_DIR / "app.js").read_text()
    line = next(x for x in source.splitlines() if x.startswith("const sharedIds"))
    script = f"{line}\nconsole.log(JSON.stringify(sharedIds('TCS,<img>,TCS,M&M,ZZZ', ['NSE:TCS','NSE:M&M'])))"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == ["NSE:TCS", "NSE:M&M"]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_test_forecast_csv_rows_line_up_by_date():
    source = (WEB_DIR / "app.js").read_text()
    function = source[source.index("const recordRows"):source.index("async function stock")]
    script = f"{function}\nconsole.log(JSON.stringify(recordRows({{'5': {{dates: ['d1', 'd2'], forecast: [1, 2], actual: [3, 4]}}}})))"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out)[1:] == [[5, "d1", 1, 3], [5, "d2", 2, 4]]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_today_sort_orders():
    source = (WEB_DIR / "app.js").read_text()
    function = source[source.index("const sorters"):source.index("async function today")]
    rows = [{"symbol": s, "h5": {"prob_above_usual": p, "forecast_vol_annualised": v, "usual_vol_annualised": u}}
            for s, p, v, u in (("A", 0.4, 0.5, 0.5), ("B", 0.7, 0.3, 0.2), ("C", 0.5, 0.2, 0.2))]
    script = f"{function}\nconst r = {json.dumps(rows)};\nconsole.log(JSON.stringify(['prob','calm','jump','vol'].map((k) => [...r].sort(sorters('h5')[k]).map((x) => x.symbol).join(''))))"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == ["BCA", "ACB", "BAC", "ABC"]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_filing_rows_and_watchlist_fill():
    source = (WEB_DIR / "app.js").read_text()
    lines = "\n".join(x for x in source.splitlines() if x.startswith(("const filingRows", "const fromWatchlist")))
    script = (f"{lines}\nconsole.log(JSON.stringify([filingRows([{{symbol:'A',date:'d',category:'c',text:'t',url:null}}])[1], "
              "fromWatchlist(new Set(['a','b','z']), new Map([['a',5]]), new Map([['a'],['b']]), 100)]))")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [["A", "d", "c", "t", ""], [["b", 100]]]


def test_live_record_counts_only_matured_forecasts(site, tmp_path, monkeypatch):
    import numpy as np

    from nifty_analyser.ops import web

    _, client = site
    assert client.get("/api/record").json() == {"rows": [], "chain_ok": True}
    ledger = AuditLog(tmp_path / "vol_ledger.jsonl")
    for day in ("2026-01-01", "2026-01-08"):
        ledger.append("vol_forecast", {"date": day, "horizon_days": 2, "forecasts": [{"instrument_id": "NSE:INFY", "prob": 0.9, "med_rv": 0.01, "low_90": -0.5, "high_90": 0.5}]})
    days = pd.date_range("2026-01-01", periods=7, freq="B", tz="Asia/Kolkata")
    monkeypatch.setattr(web, "load_cached", lambda i: pd.DataFrame({"event_time": days, "close": 100 * np.array([1, 1.1] * 3 + [1]), "quality_flag": "PASS"}))
    rows = client.get("/api/record").json()["rows"]
    assert [(r["date"], r["graded"], r["right"], r["ranged"], r["inside"]) for r in rows] == [("2026-01-08", 0, 0, 0, 0), ("2026-01-01", 1, 1, 1, 1)]


@pytest.mark.parametrize("bad", ["NSE_INFY", "nse:infy", "NSE_NIFTY50_INDEX", "a" * 500])
def test_stock_rejects_malformed_ids_with_404(site, monkeypatch, bad):
    from nifty_analyser.ops import web

    _, client = site
    # reproduce a real cache hit for look-alike names (the path sanitiser maps them onto existing files)
    monkeypatch.setattr(web, "cache_path", lambda i: WEB_DIR / "index.html")
    assert client.get(f"/api/stock/{bad}").status_code == 404
