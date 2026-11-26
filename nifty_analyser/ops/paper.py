"""Forward paper-trading ledger: the only evidence that cannot have been tuned on is data that did not exist yet.
The strategy spec (quantile, hold period, cost model) is PRE-REGISTERED -- hashed into every ledger entry, so changing it
afterwards is visible -- and each rebalance book is appended to the hash-chained audit log using only features with
available_time <= as_of. ``realized`` later scores the books against prices that arrived after they were written."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_analyser.eval.costs import ImpactCostModel
from nifty_analyser.ops.audit import AuditLog, digest

SPEC = {"strategy": "xs_long_short", "q": 0.2, "hold_days": 20, "legs": "50/50 dollar-neutral",
        "cost": {"fixed_bps": 10.0, "half_spread_bps": 5.0, "impact_coef": 0.5, "capital": 1e8}, "registered": "2026-11-24"}
SPEC_HASH = digest(SPEC)
# Second pre-registered arm (registered 2026-11-24, before any outcome of either arm exists): same portfolio rules, but scored by a
# logit-average of LightGBM + CatBoost (seed 42) refit on every labelled row whose outcome was already public at the decision time.
# Motivation: the production stacked ensemble showed no cross-sectional signal in walk-forward tests (reports/longshort_models.json).
SPEC_BLEND = {**SPEC, "model": "tree_blend: logit-mean(lightgbm, catboost), seed 42, refit on rows with outcome_available_time <= as_of"}
SPEC_BLEND_HASH = digest(SPEC_BLEND)
MIN_NAMES = 20


def target_book(model, feats: pd.DataFrame, cols: list[str], as_of: pd.Timestamp, q: float = SPEC["q"]) -> dict:
    """Scores the newest feature row per instrument published by ``as_of`` (nothing later can leak in)."""
    f = feats[feats["available_time"] <= as_of]
    if f.empty:
        raise ValueError("no features available at as_of")
    day = f["event_time"].max()
    f = f[f["event_time"] == day].dropna(subset=cols, how="all")
    if len(f) < MIN_NAMES:
        raise ValueError(f"only {len(f)} names scored on {day.date()}; need >= {MIN_NAMES}")
    s = pd.Series(model.predict_proba_up(f[cols].to_numpy(dtype=float)), index=f["instrument_id"].to_numpy()).sort_values(kind="stable")
    k = max(int(round(q * len(s))), 1)
    return {"decision_date": str(day.date()), "n": int(len(s)), "k": k, "short": s.index[:k].tolist(), "long": s.index[-k:].tolist()}


def books(ledger: AuditLog) -> list[dict]:
    return [e["payload"] for e in ledger.read() if e["kind"] == "paper_book"]


def is_due(ledger: AuditLog, decision_date: str, trading_dates: list[str], hold_days: int = SPEC["hold_days"]) -> bool:
    b = books(ledger)
    if not b:
        return True
    last = b[-1]["decision_date"]
    return sum(1 for d in trading_dates if last < d <= decision_date) >= hold_days


def record_book(ledger: AuditLog, book: dict, model_version: str, spec_hash: str = SPEC_HASH) -> dict:
    if any(b["decision_date"] == book["decision_date"] for b in books(ledger)):
        raise ValueError(f"book for {book['decision_date']} already recorded (ledger is append-only)")
    return ledger.append("paper_book", {**book, "model_version": model_version, "spec_hash": spec_hash})


def realized(ledger: AuditLog, bars: pd.DataFrame, liq: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily gross/net returns of the recorded books on prices AFTER each decision date (close of the decision day -> later closes).
    Each book is held until the next book is recorded; empty frame if no book has any later price yet."""
    bk = books(ledger)
    if not bk:
        return pd.DataFrame(columns=["gross", "net", "turnover"])
    c = bars.pivot_table(index="event_time", columns="instrument_id", values="close").sort_index()
    c.index = pd.DatetimeIndex(c.index).tz_localize(None) if c.index.tz is None else pd.DatetimeIndex(c.index).tz_convert("Asia/Kolkata").tz_localize(None)
    ret = np.log(c / c.shift(1))
    m, prev, out = ImpactCostModel(**SPEC["cost"]), pd.Series(dtype=float), []
    for i, b in enumerate(bk):
        d0 = pd.Timestamp(b["decision_date"])
        d1 = pd.Timestamp(bk[i + 1]["decision_date"]) if i + 1 < len(bk) else ret.index.max()
        w = pd.Series(0.0, index=c.columns)
        w[b["long"]], w[b["short"]] = 0.5 / b["k"], -0.5 / b["k"]
        dw = (w - prev.reindex(w.index).fillna(0.0)).abs()
        cost_bps = np.full(len(w), m.fixed_bps + m.half_spread_bps)
        if liq is not None:
            l = liq[liq["event_time"].dt.tz_localize(None) == d0].set_index("instrument_id").reindex(w.index) if liq["event_time"].dt.tz is not None else liq[liq["event_time"] == d0].set_index("instrument_id").reindex(w.index)
            part = (dw * m.capital / l["adv_value"].where(l["adv_value"] > 0)).clip(upper=1.0).fillna(1.0)
            cost_bps = cost_bps + m.impact_coef * l["sigma"].fillna(l["sigma"].median()).fillna(0.02) * np.sqrt(part) * 1e4
        days = ret.loc[(ret.index > d0) & (ret.index <= d1)]
        if days.empty:
            continue
        g = (days.fillna(0.0) * w).sum(axis=1)
        cost = float((dw * cost_bps / 1e4).sum())
        n = g.copy(); n.iloc[0] -= cost
        out.append(pd.DataFrame({"gross": g, "net": n, "turnover": [float(dw.sum())] + [0.0] * (len(g) - 1)}))
        prev = w
    return pd.concat(out) if out else pd.DataFrame(columns=["gross", "net", "turnover"])
