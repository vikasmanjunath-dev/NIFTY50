"""Self-contained HTML monitoring dashboard (Blueprint §5.5): health, drift, model, audit."""
from __future__ import annotations

from collections import Counter
from html import escape

import pandas as pd

_COLOR = {"GREEN": "#1a7f37", "AMBER": "#b7791f", "RED": "#c0392b"}


def _table(headers, rows) -> str:
    th = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def render_dashboard(health: dict, psi_table: pd.DataFrame | None, meta: dict, audit_entries: list[dict], audit_ok: bool, metrics: dict | None = None) -> str:
    color = _COLOR.get(health["state"], "#555")
    calls = [e for e in audit_entries if e["kind"] == "analyse"]
    status = Counter(e["payload"]["status"] for e in calls)
    drift = "<p>not assessed</p>" if psi_table is None else _table(
        ["feature", "tier", "PSI", "status"], [(r.feature, r.tier, "" if pd.isna(r.psi) else f"{r.psi:.3f}", r.status) for r in psi_table.head(15).itertuples()])
    if psi_table is not None and len(psi_table):
        t = psi_table["tier"].value_counts().to_dict()
        drift += f"<p>Monitored on rows: {t.get('instrument', 0)} &middot; market-wide (per-day): {t.get('market_wide', 0)} &middot; scale-dependent, not monitored: {t.get('scale_dependent', 0)}</p>"
    recent = _table(["seq", "time (UTC)", "instrument", "status", "prob_up"],
                    [(e["seq"], e["ts"][:19], e["payload"]["instrument_id"], e["payload"]["status"],
                      (e["payload"]["signal"] or {}).get("prob_up", "")) for e in calls[-10:][::-1]])
    m = "" if not metrics else "<h2>Locked validation metrics</h2>" + _table(["metric", "value"], [(k, v) for k, v in metrics.items()])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>STM monitoring</title><style>
body{{font:14px system-ui;margin:24px;max-width:960px}}table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:left}}
.banner{{padding:12px 16px;color:#fff;border-radius:6px;background:{color};font-weight:600}}</style></head><body>
<h1>Nifty 50 analyser — monitoring</h1>
<div class="banner">Serving state: {escape(health['state'])}</div>
<ul>{"".join(f"<li>{escape(r)}</li>" for r in health.get('reasons', [])) or "<li>no active alerts</li>"}</ul>
<p>Data age: {health.get('data_age_days') if health.get('data_age_days') is None else round(health['data_age_days'], 2)} days &middot;
significant-drift share: {health.get('psi_significant_frac') if health.get('psi_significant_frac') is None else f"{health['psi_significant_frac']:.0%}"}</p>
<h2>Model</h2>{_table(["version", "trained through", "features"], [(meta.get("version"), meta.get("trained_through"), len(meta.get("feature_cols", [])))])}
<p><em>{escape(meta.get("disclaimer", ""))}</em></p>{m}
<h2>Scale-free feature drift (top 15 by PSI)</h2>{drift}
<h2>Audit log</h2><p>Chain integrity: <b>{"OK" if audit_ok else "BROKEN"}</b> &middot; {len(audit_entries)} entries &middot; calls: {escape(str(dict(status)))}</p>{recent}
</body></html>"""
