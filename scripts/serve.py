"""Run the /v3 service (localhost only by default). Admin endpoints need STM_ADMIN_TOKEN.
Run: STM_ADMIN_TOKEN=... python3 scripts/serve.py [--port 8310]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.config import DATA_PROCESSED_DIR, REPORTS_DIR
from nifty_analyser.data.timestamps import to_ist
from nifty_analyser.ops.api import create_app
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.kill_switch import KillSwitch, Thresholds, compute_live_health
from nifty_analyser.ops.registry import load_model
from nifty_analyser.ops.service import ServingContext, TTLCached

OPS_DIR = DATA_PROCESSED_DIR / "ops"


def build_context():
    model, meta = load_model(DATA_PROCESSED_DIR / "models")
    feats = pd.read_parquet(DATA_PROCESSED_DIR / "serving_features.parquet")
    for c in ("event_time", "available_time"):
        feats[c] = pd.to_datetime(feats[c], utc=True).dt.tz_convert("Asia/Kolkata")
    trained_through = to_ist(pd.Timestamp(meta["trained_through"]))
    cache = {}
    cal_path = DATA_PROCESSED_DIR / "models" / "drift_calibration.json"
    # Calibrated thresholds if scripts/calibrate_drift.py has been run; otherwise textbook defaults, which are known
    # to over-alarm on these autocorrelated features (see monitoring.drift.calibrate_drift_thresholds).
    th = Thresholds.calibrated(json.loads(cal_path.read_text())) if cal_path.exists() else Thresholds()

    def live():
        h, table = compute_live_health(feats, meta["feature_cols"], trained_through, to_ist(pd.Timestamp.now(tz="Asia/Kolkata")), th)
        cache["psi"] = table
        return h

    ctx = ServingContext(features=feats, cols=meta["feature_cols"], model=model, meta=meta, killswitch=KillSwitch(OPS_DIR / "kill_switch.json"),
                         audit=AuditLog(OPS_DIR / "audit.jsonl"), health_fn=TTLCached(live, 300), horizon_days=meta["horizon_days"])
    metrics = None
    try:
        a = json.loads((REPORTS_DIR / "phase2_enriched_metrics.json").read_text())["results"]["pit"]["aggregate"]
        metrics = {k: f"{a[k]['mean']:.4f} ± {a[k]['std']:.4f}" for k in ("balanced_accuracy", "roc_auc", "information_coefficient", "log_loss")}
    except Exception:  # noqa: BLE001
        pass
    return ctx, (lambda: cache.get("psi")), metrics


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8310)
    ap.add_argument("--evidence", action="store_true", help="also serve /v3/evidence (loads announcements + vector index)")
    ap.add_argument("--rerank", action="store_true", help="enable the LLM reranker for /v3/evidence (needs Ollama)")
    ap.add_argument("--explain", action="store_true", help="also serve /v3/explain (guarded LLM answers; needs --evidence and Ollama)")
    ap.add_argument("--classify", action="store_true", help="also serve /v3/classify (needs data/processed/models/event_classifier*.joblib from scripts/train_event_classifier.py)")
    ap.add_argument("--nli", action="store_true", help="load the pretrained NLI faithfulness model (advisory scores; veto when a request sets strict_faithfulness)")
    args = ap.parse_args()
    ctx, psi, metrics = build_context()
    docs = None
    if args.evidence:
        from nifty_analyser.config import DATA_RAW_DIR
        from nifty_analyser.documents.index import VectorIndex, make_ollama_embedder
        from nifty_analyser.documents.rerank import ollama_scorer
        from nifty_analyser.ops.evidence_service import DocsContext
        d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
        d["available_time"] = pd.to_datetime(d["available_time"], utc=True).dt.tz_convert("Asia/Kolkata")
        d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
        from nifty_analyser.documents.index import CategoryRouter
        emb = make_ollama_embedder()
        docs = DocsContext(d, VectorIndex(DATA_PROCESSED_DIR / "vector_index", emb), ctx.audit, nli_fn=(__import__('nifty_analyser.documents.nli', fromlist=['nli_check']).nli_check if args.nli else None), router=CategoryRouter(d.category.unique().tolist(), emb), scorer= ollama_scorer() if args.rerank else None,
                       generate=__import__('nifty_analyser.documents.llm', fromlist=['generate_explanation']).generate_explanation if args.explain else None)
    clf = None
    if args.classify:
        from nifty_analyser.documents.event_classifier import EventClassifier
        mp = DATA_PROCESSED_DIR / 'models'; clf = EventClassifier.load(mp / ('event_classifier_final.joblib' if (mp / 'event_classifier_final.joblib').exists() else 'event_classifier.joblib'))
    vp = DATA_PROCESSED_DIR / "volatility_latest.json"      # written by scripts/forecast_volatility.py
    vol = __import__("json").loads(vp.read_text()) if vp.exists() else None
    uvicorn.run(create_app(ctx, psi, metrics, docs, clf, vol), host=args.host, port=args.port)
