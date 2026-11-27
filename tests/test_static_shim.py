"""The hosted snapshot recomputes portfolio risk and staleness in the browser; check that code with Node."""
import json
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

SHIM = Path(__file__).resolve().parents[1] / "scripts" / "static_shim.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def call(url: str, files: dict) -> dict:
    code = f"""
const files = {json.dumps(files)};
global.window = {{ fetch: async (u) => new Response(JSON.stringify(files[u] ?? {{}}), {{ status: files[u] ? 200 : 404 }}) }};
global.location = {{ href: "https://x.test/" }};
require({json.dumps(str(SHIM))});
window.fetch({json.dumps(url)}).then(async (r) => console.log(JSON.stringify({{ code: r.status, body: await r.json() }})));
"""
    return json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)


def snapshot(day: date) -> dict:
    def f(vol, usual):
        return {"forecast_vol_annualised": vol, "usual_vol_annualised": usual}
    return {"/data/portfolio.json": {"blocks": {"5": {"data_date": str(day), "forecasts": {"A": f(0.3, 0.25), "B": f(0.2, 0.2)}}},
                                     "corr": {"A": {"A": 1, "B": 0.5}, "B": {"A": 0.5, "B": 1}}, "symbols": {"A": "AA", "B": "BB"}, "test": {}}}


def test_portfolio_matches_the_matrix_formula():
    body = call("/api/portfolio?holdings=A:300,B:100&horizon=5", snapshot(date.today()))["body"]
    w, vol = np.array([0.75, 0.25]), np.array([0.3, 0.2])
    cov = np.outer(vol, vol) * np.array([[1, 0.5], [0.5, 1]])
    total_vol = float(np.sqrt(w @ cov @ w))
    assert body["volatility"] == pytest.approx(total_vol)
    assert body["large_move"] == pytest.approx(1.645 * total_vol * np.sqrt(5 / 252) * 400)
    assert body["parts"][0]["risk_share"] == pytest.approx(float(w[0] * (cov @ w)[0] / total_vol**2))


def test_old_forecasts_are_withheld():
    out = call("/api/portfolio?holdings=A:1&horizon=5", snapshot(date.today() - timedelta(days=9)))
    assert out["body"] == {"status": "STALE"}


def test_unknown_holding_is_rejected_and_posts_are_refused():
    assert call("/api/portfolio?holdings=ZZ:1", snapshot(date.today()))["code"] == 422
    assert call("/v3/explain", {})["code"] == 404
