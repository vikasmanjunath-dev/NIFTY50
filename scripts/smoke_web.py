"""End-to-end check of the web app against the real local data, without starting a network server.
Run: python3 scripts/smoke_web.py   (exit code 1 if any page or route is broken)."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import build_context

from nifty_analyser.ops.api import create_app
from nifty_analyser.ops.web import attach_web


def main():
    ctx, psi, metrics = build_context()
    client = TestClient(attach_web(create_app(ctx, psi, metrics), ctx), base_url="http://127.0.0.1")
    overview = client.get("/api/overview").json()
    ids = [r["id"] for r in overview["rows"]]
    paths = ["/", "/sw.js", "/static/app.js", "/static/app.css", "/api/model", "/api/system", "/api/backup", "/api/filings", "/api/filings/categories", "/v3/health"]
    paths += [f"/api/stock/{i}" for i in ids]
    if ids:
        paths.append("/api/portfolio?holdings=" + ",".join(f"{i}:1000" for i in ids[:5]))
    bad = [(p, r.status_code) for p in paths if (r := client.get(p)).status_code != 200]
    print(f"forecast status {overview['status']}, {len(ids)} stocks, {len(paths)} routes checked, {len(bad)} failed")
    for path, code in bad[:10]:
        print("  ", code, path)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
