"""Read-only JSON for the web app, plus the static files. All data comes from files the pipeline already writes.
Forecasts are withheld while the kill switch is tripped or when they are stale."""
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_DIR
from nifty_analyser.data.ingest import cache_path, load_cached
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.refresh import Refresher
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

    def _bars(self, iid: str) -> pd.DataFrame:
        stamp = cache_path(iid).stat().st_mtime if cache_path(iid).exists() else None
        if self._cache.get(("bars", iid), (None,))[0] != stamp or stamp is None:
            self._cache[("bars", iid)] = (stamp, load_cached(iid))
        return self._cache[("bars", iid)][1]

    def _ledger(self) -> list[dict]:
        path = self.processed / "vol_ledger.jsonl"
        if not path.exists():
            return []
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        if self._cache.get("ledger", (None,))[0] != stamp:
            self._cache["ledger"] = (stamp, AuditLog(path).read())
        return self._cache["ledger"][1]

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
                "rows": list(rows.values()), "upcoming": self.upcoming(set(rows)), "market": self.market(), "changes": self.changes(blocks), "disclaimer": DISCLAIMER}

    def changes(self, blocks: dict, top: int = 6) -> dict:
        """Largest moves in each chance since the previous recorded forecast date (from the forward ledger)."""
        entries = [e["payload"] for e in self._ledger() if e["kind"] == "vol_forecast"]
        out = {}
        for horizon, block in blocks.items():
            older = [p for p in entries if str(p["horizon_days"]) == horizon and p["date"] < block["data_date"]]
            if not older:
                continue
            previous = max(older, key=lambda p: p["date"])
            before = {f["instrument_id"]: f["prob"] for f in previous["forecasts"]}
            moves = [{"id": i, "symbol": get_instrument(i).symbol, "now": f["prob_above_usual"], "before": before[i]}
                     for i, f in block["forecasts"].items() if i in before and abs(f["prob_above_usual"] - before[i]) >= 0.02]
            out[horizon] = {"since": previous["date"], "rows": sorted(moves, key=lambda m: -abs(m["now"] - m["before"]))[:top]}
        return out

    def upcoming(self, ids: set, days: int = 30) -> list[dict]:
        """Announced meetings from today onward, for the stocks shown; only intimations already published."""
        m = self._parquet("board_meetings.parquet")
        if m is None or not ids:
            return []
        now = self.ctx.now_fn()
        today = now.normalize()
        m = m[m.instrument_id.isin(ids) & (m.available_time <= now) & (m.meeting_date >= today) & (m.meeting_date <= today + pd.Timedelta(days=days))]
        m = m.sort_values(["meeting_date", "is_results"], ascending=[True, False]).drop_duplicates(["instrument_id", "meeting_date"])
        return [{"id": r.instrument_id, "symbol": get_instrument(r.instrument_id).symbol, "date": str(r.meeting_date.date()),
                 "results": bool(r.is_results)} for r in m.itertuples()]

    def _backtest(self, horizon: str):
        path = self.processed / f"vol_oos_h{horizon}.parquet"
        return pd.read_parquet(path) if path.exists() else None

    def track_record(self, iid: str) -> dict:
        """Past walk-forward forecasts of volatility beside what then happened (both yearly), per horizon."""
        out = {}
        for h in ("5", "20"):
            d = self._backtest(h)
            if d is None:
                continue
            d = d[d.instrument_id == iid].sort_values("event_time").tail(250)
            if len(d) < 30:
                continue
            yearly = np.sqrt(252 / int(h))
            out[h] = {"dates": [str(t)[:10] for t in d.event_time], "forecast": [round(float(v), 4) for v in np.exp(d.r) * yearly],
                      "actual": [round(float(v), 4) for v in np.exp(d.lv) * yearly]}
        return out

    def reliability(self) -> dict:
        """When the model said x%, how often did volatility actually come out above usual?"""
        out = {}
        for h in ("5", "20"):
            d = self._backtest(h)
            if d is None:
                continue
            bins = np.clip((d.p * 5).astype(int), 0, 4)
            out[h] = [{"said": round(float(g.p.mean()), 3), "happened": round(float(g.y.mean()), 3), "n": int(len(g))}
                      for _, g in d.groupby(bins) if len(g) >= 100]
        return out

    def forward(self, iid: str, bars: pd.DataFrame) -> list[dict]:
        """Every forecast saved for this stock before its outcome, graded once the horizon has fully passed."""
        entries = self._ledger()
        if not entries:
            return []
        close = bars[bars.quality_flag != "FAIL"].sort_values("event_time")
        days, prices = close.event_time.dt.strftime("%Y-%m-%d").to_numpy(), close.close.to_numpy(float)
        out = []
        for e in entries:
            if e["kind"] != "vol_forecast":
                continue
            p = e["payload"]
            f = next((f for f in p["forecasts"] if f["instrument_id"] == iid), None)
            if f is None:
                continue
            h = p["horizon_days"]
            i = int(np.searchsorted(days, p["date"]))
            above = inside = None
            if i + h < len(prices):
                if "low_90" in f:
                    inside = bool(f["low_90"] <= prices[i + h] / prices[i] - 1 <= f["high_90"])
                above = bool(np.sqrt((np.diff(np.log(prices[i: i + h + 1])) ** 2).sum()) > f["med_rv"])
            out.append({"date": p["date"], "horizon": h, "prob": f["prob"], "above_usual": above, "inside_range": inside, "days_left": max(0, i + h - len(prices) + 1)})
        return out[::-1][:20]

    def peers(self, iid: str, blocks: dict) -> dict:
        """Average chance across the other stocks of the same sector, per horizon."""
        sector = get_instrument(iid).sector
        out = {}
        for horizon, block in blocks.items():
            others = [f["prob_above_usual"] for i, f in block["forecasts"].items() if i != iid and get_instrument(i).sector == sector]
            if others:
                out[horizon] = {"sector_average": round(float(np.mean(others)), 4), "stocks": len(others)}
        return out

    def results_moves(self, iid: str, bars: pd.DataFrame, last: int = 8) -> dict | None:
        """Absolute price change from the close before each past results meeting to the close after it (two sessions, because
        results often arrive after the close), beside the typical two-session change."""
        m = self._parquet("board_meetings.parquet")
        if m is None:
            return None
        close = bars[bars.quality_flag != "FAIL"].sort_values("event_time")
        days, prices = close.event_time.dt.strftime("%Y-%m-%d").to_numpy(), close.close.to_numpy(float)
        if len(prices) < 60:
            return None
        events = []
        for day in sorted(m[(m.instrument_id == iid) & m.is_results].meeting_date.dt.strftime("%Y-%m-%d").unique(), reverse=True):
            i = int(np.searchsorted(days, day))  # first session on or after the meeting
            if 0 < i < len(prices) - 1:
                events.append({"date": day, "move": round(float(prices[i + 1] / prices[i - 1] - 1), 4)})
            if len(events) == last:
                break
        if not events:
            return None
        usual = np.abs(prices[2:] / prices[:-2] - 1)[-250:]
        return {"events": events, "typical_results": round(float(np.median([abs(e["move"]) for e in events])), 4),
                "typical_any": round(float(np.median(usual)), 4)}

    def market(self) -> dict | None:
        """India VIX now, its rank within the past year, and the year's path. Measured history, not a forecast."""
        vix = self._parquet("macro_india_vix.parquet")
        if vix is None:
            return None
        v = vix[vix.available_time <= self.ctx.now_fn()].dropna(subset=["close"]).sort_values("event_time").tail(250)
        if len(v) < 60:
            return None
        now = float(v.close.iloc[-1])
        return {"vix": round(now, 2), "date": str(v.event_time.iloc[-1])[:10], "rank": round(float((v.close <= now).mean()), 3),
                "history": [[str(t)[:10], round(float(c), 2)] for t, c in zip(v.event_time, v.close)]}

    def unusual(self, bars: pd.DataFrame) -> dict | None:
        """Yearly volatility over the last 20 days, ranked against every 20-day window of the past five years."""
        close = bars[bars.quality_flag != "FAIL"].sort_values("event_time")
        series = (np.log(close.set_index("event_time").close).diff().rolling(20).std() * np.sqrt(252)).dropna().tail(1250)
        if len(series) < 250:
            return None
        recent = series.tail(250)
        return {"now": round(float(series.iloc[-1]), 4), "rank": round(float((series <= series.iloc[-1]).mean()), 3), "windows": int(len(series)),
                "history": [[str(t)[:10], round(float(x), 4)] for t, x in recent.items()]}

    def sector(self, name: str):
        """Stocks of one sector with their forecasts and the average past-year correlation between them."""
        status, blocks = self.forecasts()
        view = self.overview()
        rows = [r for r in view["rows"] if r["sector"] == name]
        if status == "OK" and not rows:
            raise HTTPException(404, "unknown sector")
        togetherness = None
        if len(rows) > 1:
            returns = {}
            for r in rows:
                bars = self._bars(r["id"])
                returns[r["id"]] = np.log(bars[bars.quality_flag != "FAIL"].set_index("event_time").close).diff().tail(250)
            corr = pd.DataFrame(returns).corr().to_numpy()
            togetherness = round(float(np.nanmean(corr[~np.eye(len(corr), dtype=bool)])), 2)
        return {"status": status, "sector": name, "rows": rows, "correlation": togetherness, "disclaimer": view["disclaimer"]}

    def backup(self):
        """Zip of the forward ledgers (they cannot be recreated) with each hash chain's status."""
        buffer, status = io.BytesIO(), {}
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(self.processed.glob("*ledger*.jsonl")):
                z.writestr(path.name, path.read_bytes())
                status[path.name] = AuditLog(path).verify()[0]
            z.writestr("chain_status.json", json.dumps(status))
        name = f"nifty-forward-records-{self.ctx.now_fn():%Y-%m-%d}.zip"
        return Response(buffer.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})

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
               "meetings": [], "record": self.track_record(iid), "forward": self.forward(iid, self._bars(iid)), "results_moves": self.results_moves(iid, self._bars(iid)), "unusual": self.unusual(self._bars(iid)),
               "peers": self.peers(iid, blocks), "filings": self.filings(instrument=iid, limit=8), "disclaimer": DISCLAIMER}
        meetings = self._parquet("board_meetings.parquet")
        if meetings is not None:
            m = meetings[meetings.instrument_id == iid].sort_values("meeting_date").drop_duplicates("meeting_date").tail(5)
            out["meetings"] = [{"date": str(r.meeting_date.date()), "results": bool(r.is_results)} for r in m.itertuples()][::-1]
        return out

    def filings(self, q: str = "", instrument: str | None = None, limit: int = 30, category: str = ""):
        docs = self._parquet("announcements_nifty50.parquet")
        if docs is None:
            return []
        if category:
            docs = docs[docs.category == category]
        if instrument:  # one id, or several separated by commas
            docs = docs[docs.instrument_id.isin(instrument.split(",")[:60])]
        for word in q.split()[:6]:
            hit = docs.text.str.contains(word, case=False, regex=False, na=False) | docs.category.str.contains(word, case=False, regex=False, na=False)
            docs = docs[hit | docs.instrument_id.str.contains(word, case=False, regex=False)]
        docs = docs.sort_values("available_time", ascending=False).head(max(1, min(limit, 100)))
        return [{"symbol": r.instrument_id.split(":")[-1], "date": str(r.available_time)[:10], "category": r.category,
                 "text": str(r.text)[:400], "url": r.attachment_url if str(r.attachment_url).startswith("https://") else None}
                for r in docs.itertuples()]

    def categories(self, top: int = 25) -> list[str]:
        docs = self._parquet("announcements_nifty50.parquet")
        return [] if docs is None else docs.category.value_counts().head(top).index.tolist()

    def model(self):
        rounds = [{"label": "Persistence rule", "h5": 0.598, "h20": 0.669}, {"label": "First blend", "h5": 0.671, "h20": 0.702},
                  {"label": "+ 2008–2020 history", "h5": 0.684, "h20": 0.739}, {"label": "+ extra features (20-day)", "h5": 0.684, "h20": 0.750},
                  {"label": "+ meeting dates (5-day)", "h5": 0.691, "h20": 0.750}]
        ranges = {h: (_json(self.reports / f"vol_ranges_h{h}.json") or {}).get("ranges", {}).get("model") for h in ("5", "20")}
        return {"reliability": self.reliability(), "rounds": rounds, "ranges": ranges, "direction_auc": 0.50,
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

    def portfolio(self, holdings: str, horizon: int = 5):
        """Risk of a long-only mix: forecast volatility per stock combined with the past year's correlations.
        `holdings` is "NSE:INFY:50000,NSE:TCS:25000" (amounts in rupees)."""
        amounts: dict[str, float] = {}
        try:
            for part in holdings.split(",")[:60]:
                iid, amount = part.rsplit(":", 1)
                value = float(amount)
                if not (value > 0 and value < 1e12):
                    raise ValueError
                amounts[iid] = amounts.get(iid, 0.0) + value
        except ValueError:
            raise HTTPException(422, "holdings must look like NSE:INFY:50000,NSE:TCS:25000 with positive amounts") from None
        status, blocks = self.forecasts()
        block = blocks.get(str(horizon))
        if block is None:
            return {"status": status if status != "OK" else "NO_DATA"}
        unknown = sorted(set(amounts) - set(block["forecasts"]))
        if unknown:
            raise HTTPException(422, f"no forecast for: {', '.join(unknown)}")
        returns = {}
        for iid in amounts:
            bars = load_cached(iid)
            bars = bars[bars.quality_flag != "FAIL"]
            returns[iid] = np.log(bars.set_index("event_time").close).diff().tail(250)
        corr = pd.DataFrame(returns).corr().fillna(0.0).to_numpy(copy=True)
        np.fill_diagonal(corr, 1.0)
        ids = list(amounts)
        total = sum(amounts.values())
        weight = np.array([amounts[i] for i in ids]) / total
        vol = np.array([block["forecasts"][i]["forecast_vol_annualised"] for i in ids])
        usual = np.array([block["forecasts"][i]["usual_vol_annualised"] for i in ids])
        cov = np.outer(vol, vol) * corr
        variance = float(weight @ cov @ weight)
        share = weight * (cov @ weight) / variance
        period = np.sqrt(variance * horizon / 252)
        pairs = sorted(((float(corr[a, b]), ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))), reverse=True)[:3]

        def without(k):  # same money split among the others in their present proportions
            keep = [j for j in range(len(ids)) if j != k]
            if not keep:
                return None
            v = weight[keep] / weight[keep].sum()
            return float(np.sqrt(v @ cov[np.ix_(keep, keep)] @ v))
        test = (_json(self.reports / "portfolio_backtest.json") or {}).get(str(horizon), {})
        multiplier = test.get("multiplier_for_90pct", 1.645)   # measured on past data; the bell curve's 1.645 is too small
        return {"status": "OK", "horizon": horizon, "data_date": block["data_date"], "total": total,
                "volatility": float(np.sqrt(variance)), "usual_volatility": float(np.sqrt(weight @ (np.outer(usual, usual) * corr) @ weight)),
                "undiversified": float(weight @ vol), "typical_move": float(period * total), "large_move": float(multiplier * period * total), "large_move_held": test.get("second_half_within_calibrated"),
                "linked": [{"a": get_instrument(a).symbol, "b": get_instrument(b).symbol, "correlation": round(c, 2)} for c, a, b in pairs],
                "parts": sorted(({"without": without(k), "id": i, "symbol": get_instrument(i).symbol, "weight": float(weight[k]), "risk_share": float(share[k]),
                                  "volatility": float(vol[k])} for k, i in enumerate(ids)), key=lambda p: -p["risk_share"])}

    # label, file under data/raw, date column, days of lag before it is flagged, how it is refreshed
    SOURCES = (("Prices", "NSE_NIFTY50_INDEX.parquet", "event_time", 4, "daily refresh"),
               ("India VIX", "macro_india_vix.parquet", "event_time", 5, "daily refresh"),
               ("Board meetings", "board_meetings.parquet", "available_time", 45, "daily refresh"),
               ("Company filings", "announcements_nifty50.parquet", "available_time", 7, "daily refresh"))

    def freshness(self) -> list[dict]:
        today = self.ctx.now_fn().date()
        out = []
        for label, name, column, limit, how in self.SOURCES:
            frame = self._parquet(name)
            if frame is None or frame.empty:
                out.append({"source": label, "newest": None, "age_days": None, "late": True, "refresh_with": how})
                continue
            newest = pd.to_datetime(frame[column], utc=True).max().tz_convert("Asia/Kolkata").date()
            out.append({"source": label, "newest": str(newest), "age_days": (today - newest).days,
                        "late": (today - newest).days > limit, "refresh_with": how})
        return out

    def system(self):
        health = self.ctx.killswitch.effective(self.ctx.health_fn()).to_dict()
        ok, _ = self.ctx.audit.verify()
        ledgers = {}
        for name in ("vol_ledger", "paper_ledger", "paper_ledger_blend"):
            path = self.processed / f"{name}.jsonl"
            ledgers[name] = sum(1 for line in path.read_text().splitlines() if line.strip()) if path.exists() else 0
        return {"health": health, "model_version": self.ctx.version, "audit_ok": ok, "audit_entries": len(self.ctx.audit.read()),
                "ledger_entries": ledgers, "forecast_status": self.forecasts()[0], "scorecard": self.scorecard(), "sources": self.freshness()}


LOCAL_HOSTS = ["127.0.0.1", "localhost", "[::1]"]


def attach_web(app: FastAPI, ctx, data: WebData | None = None, refresher: Refresher | None = None, hosts=LOCAL_HOSTS) -> FastAPI:
    data, refresher = data or WebData(ctx), refresher or Refresher()
    # Rejects requests that reach the server under another site's name (DNS rebinding).
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(hosts))

    def refresh(x_requested_with: str | None = Header(None), origin: str | None = Header(None), host: str = Header("")):
        # A custom header cannot be sent cross-site without a CORS preflight, which this server never grants.
        if x_requested_with != "nifty-web" or (origin is not None and origin.split("//")[-1] != host):
            raise HTTPException(403, "refresh can only be started from the app itself")
        if not refresher.start():
            raise HTTPException(409, "a refresh is already running")
        ctx.audit.append("refresh", {"source": "web"})
        return refresher.state

    app.post("/api/refresh")(refresh)
    app.get("/api/refresh")(lambda: refresher.state)
    app.get("/api/overview")(data.overview)
    app.get("/api/stock/{iid}")(data.stock)
    app.get("/api/filings")(data.filings)
    app.get("/api/filings/categories")(data.categories)
    app.get("/api/portfolio")(data.portfolio)
    app.get("/api/sector/{name}")(data.sector)
    app.get("/api/backup")(data.backup)
    app.get("/api/model")(data.model)
    app.get("/api/system")(data.system)
    app.get("/", include_in_schema=False)(lambda: FileResponse(WEB_DIR / "index.html"))
    app.get("/sw.js", include_in_schema=False)(lambda: FileResponse(WEB_DIR / "sw.js", media_type="text/javascript"))
    @app.middleware("http")
    async def revalidate(request, call_next):
        # Without this, browsers reuse old scripts after an update; ETags keep the re-check cheap.
        response = await call_next(request)
        if not request.url.path.startswith("/v3"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
