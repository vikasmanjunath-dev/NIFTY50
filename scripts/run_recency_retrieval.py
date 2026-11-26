"""Retrieval hit-rate (gold filing in top-5) with and without a recency boost, no LLM. Questions as in run_filing_qa.py (dividend record date / results period),
400-day window, instrument-filtered. Weight chosen on TUNE seed, confirmed on HOLDOUT seed."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.instrument_master import NIFTY50
from nifty_analyser.documents.index import VectorIndex, make_ollama_embedder, retrieve
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
d["available_time"] = pd.to_datetime(d["available_time"], utc=True).dt.tz_convert("Asia/Kolkata"); d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
syms = {i.instrument_id: i.symbol for i in NIFTY50.values()}
rec = d[d.text.str.contains("record date", case=False) & d.text.str.contains("dividend", case=False) & d.text.str.contains(r"is\s*\d{1,2}[\s\-][A-Za-z]{3,9}[\s\-]\d{4}", case=False, regex=True)]
res_ = d[d.text.str.contains(r"financial results for the period ended", case=False)]
ix = VectorIndex(DATA_PROCESSED_DIR / "vector_index", make_ollama_embedder())
def questions(seed, n):
    out = []
    for kind, pool, tmpl in (("record_date", rec, "What is the record date for the dividend announced by {s}?"), ("results_period", res_, "For which period did {s} submit its latest financial results?")):
        p = pool[(pool.available_time >= "2023-01-01") & (pool.available_time < "2026-10-01")].sample(n, random_state=seed)
        out += [(kind, r.instrument_id, r.doc_id, r.available_time + pd.Timedelta(days=1), tmpl.format(s=syms.get(r.instrument_id, r.instrument_id))) for r in p.itertuples()]
    return out
def hit(qs, w, k=5):
    h = {"record_date": [], "results_period": []}
    for kind, inst, doc, as_of, q in qs:
        docs = d[(d.available_time > as_of - pd.Timedelta(days=400)) & (d.available_time <= as_of)]
        h[kind].append(doc in retrieve(q, docs, as_of, ix, top_k=k, instrument_id=inst, recency_weight=w).doc_id.tolist())
    return {kind: round(float(np.mean(v)), 3) for kind, v in h.items()} | {"all": round(float(np.mean(sum(h.values(), []))), 3)}
W = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0); tune, hold = questions(101, 100), questions(202, 100); out = {"tune": {str(w): hit(tune, w) for w in W}}
best = max(W, key=lambda w: (out["tune"][str(w)]["all"], -w)); out["chosen_on_tune"] = best; out["holdout"] = {str(w): hit(hold, w) for w in (0.0, best)}
out["holdout_top3_baseline_vs_chosen"] = {str(w): hit(hold, w, k=3)["all"] for w in (0.0, best)}
Path("reports/recency_retrieval.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
