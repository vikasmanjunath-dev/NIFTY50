"""Read-only JSON for the web app, plus the static files. All data comes from files the pipeline already writes.
Forecasts are withheld while the kill switch is tripped or when they are stale."""
import json
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_DIR
from nifty_analyser.data.ingest import cache_path, load_cached
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.vol_forecast import score

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
DISCLAIMER = "Forecasts the size of price moves, not their direction. Research output, not investment advice."


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


class WebData:
    def __init__(self, ctx, processed=DATA_PROCESSED_DIR, raw=DATA_RAW_DIR, reports=REPORTS_DIR):
        self.ctx, self.processed, self.raw, self.reports = ctx, Path(processed), Path(raw), Path(reports)
        self._cache: dict = {}

    def _parquet(self, name):
        path = self.raw / name
        if not path.exists():
            return None
        stamp = path.stat().st_mtime
        if self._cache.get(name, (None,))[0] != stamp:
            self._cache[name] = (stamp, pd.read_parquet(path))
        return self._cache[name][1]

    def forecasts(self):
        """(status, {horizon: block}); blocks are empty unless status is OK."""
        latest = _json(self.processed / "volatility_latest.json")
        if self.ctx.killswitch.manual().get("tripped"):
            return "HALTED", {}
        if not latest:
            return "NO_DATA", {}
        age = (self.ctx.now_fn().date() - date.fromisoformat(max(b["data_date"] for b in latest.values()))).days
        return ("STALE", {}) if age > self.ctx.stale_days else ("OK", latest)

    def overview(self):
        status, blocks = self.forecasts()
        rows = {}
        for horizon, block in blocks.items():
            for iid, f in block["forecasts"].items():
                inst = get_instrument(iid)
                rows.setdefault(iid, {"id": iid, "symbol": inst.symbol, "sector": getattr(inst, "sector", None)})[f"h{horizon}"] = f
        return {"status": status, "data_date": max((b["data_date"] for b in blocks.values()), default=None),
                "rows": list(rows.values()), "disclaimer": DISCLAIMER}

    def stock(self, iid):
        if not cache_path(iid).exists():
            raise HTTPException(404, "unknown instrument")
        bars = load_cached(iid)
        bars = bars[bars.quality_flag != "FAIL"].tail(260)
        status, blocks = self.forecasts()
        inst = get_instrument(iid)
        out = {"id": iid, "symbol": inst.symbol, "sector": getattr(inst, "sector", None), "status": status,
               "prices": [[str(t.date()), round(float(c), 2)] for t, c in zip(bars.event_time, bars.close)],
               "forecasts": {h: {**b["forecasts"][iid], "data_date": b["data_date"]} for h, b in blocks.items() if iid in b["forecasts"]},
               "meetings": [], "filings": self.filings(instrument=iid, limit=8), "disclaimer": DISCLAIMER}
        meetings = self._parquet("board_meetings.parquet")
        if meetings is not None:
            m = meetings[meetings.instrument_id == iid].sort_values("meeting_date").drop_duplicates("meeting_date").tail(5)
            out["meetings"] = [{"date": str(r.meeting_date.date()), "results": bool(r.is_results)} for r in m.itertuples()][::-1]
        return out

    def filings(self, q: str = "", instrument: str | None = None, limit: int = 30):
        docs = self._parquet("announcements_nifty50.parquet")
        if docs is None:
            return []
        if instrument:
            docs = docs[docs.instrument_id == instrument]
        for word in q.split()[:6]:
            hit = docs.text.str.contains(word, case=False, regex=False, na=False) | docs.category.str.contains(word, case=False, regex=False, na=False)
            docs = docs[hit | docs.instrument_id.str.contains(word, case=False, regex=False)]
        docs = docs.sort_values("available_time", ascending=False).head(max(1, min(limit, 100)))
        return [{"symbol": r.instrument_id.split(":")[-1], "date": str(r.available_time)[:10], "category": r.category,
                 "text": str(r.text)[:400], "url": r.attachment_url if str(r.attachment_url).startswith("https://") else None}
                for r in docs.itertuples()]

    def model(self):
        rounds = [{"label": "Persistence rule", "h5": 0.598, "h20": 0.669}, {"label": "First blend", "h5": 0.671, "h20": 0.702},
                  {"label": "+ 2008–2020 history", "h5": 0.684, "h20": 0.739}, {"label": "+ extra features (20-day)", "h5": 0.684, "h20": 0.750},
                  {"label": "+ meeting dates (5-day)", "h5": 0.691, "h20": 0.750}]
        ranges = {h: (_json(self.reports / f"vol_ranges_h{h}.json") or {}).get("ranges", {}).get("model") for h in ("5", "20")}
        return {"rounds": rounds, "ranges": ranges, "direction_auc": 0.50,
                "rejected": ["Delivery %, announcement counts, FII flows", "Chronos foundation model", "Overnight US market",
                             "Intraday realised volatility", "Option-implied volatility (retest later)", "Deeper trees, ensembles, recalibration"]}

    def scorecard(self):
        """Forward AUC from the hash-chained forecast ledger; empty until forecasts mature."""
        path = self.processed / "vol_ledger.jsonl"
        if not path.exists():
            return {}
        stamp = path.stat().st_mtime
        if self._cache.get("score", (None,))[0] != stamp:
            self._cache["score"] = (stamp, score(AuditLog(path), load_universe_bars()))
        return self._cache["score"][1]

    def system(self):
        health = self.ctx.killswitch.effective(self.ctx.health_fn()).to_dict()
        ok, _ = self.ctx.audit.verify()
        ledgers = {}
        for name in ("vol_ledger", "paper_ledger", "paper_ledger_blend"):
            path = self.processed / f"{name}.jsonl"
            ledgers[name] = sum(1 for line in path.read_text().splitlines() if line.strip()) if path.exists() else 0
        return {"health": health, "model_version": self.ctx.version, "audit_ok": ok, "audit_entries": len(self.ctx.audit.read()),
                "ledger_entries": ledgers, "forecast_status": self.forecasts()[0], "scorecard": self.scorecard()}


def attach_web(app: FastAPI, ctx, data: WebData | None = None) -> FastAPI:
    data = data or WebData(ctx)
    app.get("/api/overview")(data.overview)
    app.get("/api/stock/{iid}")(data.stock)
    app.get("/api/filings")(data.filings)
    app.get("/api/model")(data.model)
    app.get("/api/system")(data.system)
    app.get("/", include_in_schema=False)(lambda: FileResponse(WEB_DIR / "index.html"))
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
