"""Writes a static, read-only snapshot of the web app to dist/nifty-tremor/ for static hosting (Vercel).
Deploy: python3 scripts/export_static.py && vercel deploy dist/nifty-tremor --prod --yes"""
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from serve import build_context  # noqa: E402

from nifty_analyser.data.ingest import load_cached  # noqa: E402
from nifty_analyser.data.instrument_master import get_instrument  # noqa: E402
from nifty_analyser.ops.web import WEB_DIR, WebData, attach_web  # noqa: E402

OUT = ROOT / "dist" / "nifty-tremor"
NAME = "Nifty Tremor"


def main():
    ctx = build_context()
    ctx = ctx[0] if isinstance(ctx, tuple) else ctx
    data = WebData(ctx)
    client = TestClient(attach_web(FastAPI(), ctx, data, hosts=["testserver"]))
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    shutil.copytree(WEB_DIR, OUT / "static", dirs_exist_ok=True)

    def save(name, value):
        (OUT / "data" / f"{name}.json").write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))

    def get(path):
        response = client.get(path)
        response.raise_for_status()
        return response.json()

    overview = get("/api/overview")
    if overview["status"] != "OK":
        sys.exit(f"forecast status is {overview['status']}; run scripts/daily_refresh.py first")
    save("overview", overview)
    save("model", get("/api/model"))
    save("system", get("/api/system"))
    save("refresh", {"running": False, "ok": None, "message": ""})
    save("filings_categories", get("/api/filings/categories"))
    ids = [r["id"] for r in overview["rows"]]
    for iid in ids:
        save(f"stock_{iid.replace(':', '_')}", get(f"/api/stock/{quote(iid)}"))
    for sector in {r["sector"] for r in overview["rows"] if r["sector"]}:
        save(f"sector_{sector.replace('/', '_').replace(' ', '-')}", get(f"/api/sector/{quote(sector)}"))
    # newest 100 filings per company, so company filters keep working
    filings = [f for iid in ids for f in data.filings(instrument=iid, limit=100)]
    save("filings", sorted(filings, key=lambda f: f["date"], reverse=True))

    returns = {}
    for iid in ids:
        bars = load_cached(iid)
        returns[iid] = np.log(bars[bars.quality_flag != "FAIL"].set_index("event_time").close).diff().tail(250)
    test = ROOT / "reports" / "portfolio_backtest.json"
    save("portfolio", {"blocks": data.forecasts()[1], "corr": pd.DataFrame(returns).corr().fillna(0.0).round(6).to_dict(),
                       "symbols": {i: get_instrument(i).symbol for i in ids}, "test": json.loads(test.read_text()) if test.exists() else {}})

    def edit(path, old, new):
        text = path.read_text()
        if old not in text:
            sys.exit(f"{path.name}: expected text not found: {old}")
        path.write_text(text.replace(old, new))

    page = OUT / "index.html"
    shutil.copy(WEB_DIR / "index.html", page)
    edit(page, '<script src="/static/app.js"', '<script src="/static/shim.js"></script>\n<script src="/static/app.js"')
    shutil.copy(ROOT / "scripts" / "static_shim.js", OUT / "static" / "shim.js")
    shutil.copy(WEB_DIR / "sw.js", OUT / "sw.js")
    app = OUT / "static" / "app.js"
    edit(app, "Question answering is off. Restart the server with: python3 scripts/serve.py --evidence --explain (needs Ollama).",
         "Question answering is not available on the hosted snapshot; it needs the local app.")
    edit(app, "Run scripts/daily_refresh.py.", "This hosted snapshot has not been updated.")
    edit(app, "Could not start the refresh.", "This is a hosted snapshot; refresh runs only in the local app.")
    edit(app, 'h("p", {}, h("a", { href: "/api/backup", download: "" }, "Download a backup of these records"))', '""')
    for path in (page, app, OUT / "static" / "manifest.webmanifest"):
        path.write_text(path.read_text().replace("Nifty Analyser", NAME))
    headers = {"Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                                          "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
               "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-cache"}
    (OUT / "vercel.json").write_text(json.dumps({"headers": [{"source": "/(.*)", "headers": [{"key": k, "value": v} for k, v in headers.items()]}]}))
    print(f"{OUT}: {len(ids)} stocks, data to {overview['data_date']}")


if __name__ == "__main__":
    main()
