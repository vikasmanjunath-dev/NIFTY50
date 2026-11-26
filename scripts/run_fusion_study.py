"""Weighted-RRF study with a proper tune/holdout split (no selection on the evaluation queries): 200 instrument+category
queries, weights chosen on the first 100, reported on the held-out 100. Both unfiltered and instrument-filtered retrieval."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.documents.eval import evaluate_ranking, ndcg_at_k
from nifty_analyser.documents.index import VectorIndex, hybrid_retrieve_indexed

VAGUE, W = {"Updates", "General Updates"}, (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
for c in ("event_time", "available_time"):
    d[c] = pd.to_datetime(d[c], utc=True).dt.tz_convert("Asia/Kolkata")
d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
dd = d[(d.text.str.len() > 20) & ~d.category.isin(VAGUE)]; g = dd.groupby(["instrument_id", "category"]).size()
pairs = g[(g >= 4) & (g <= 12)].reset_index()[["instrument_id", "category"]].sample(200, random_state=7)       # different seed from the earlier 40-query sets
ix = VectorIndex(DATA_PROCESSED_DIR / "vector_index"); rows = []
for i, r in enumerate(pairs.itertuples()):
    rel = set(d[(d.instrument_id == r.instrument_id) & (d.category == r.category)].doc_id)
    q = f"{get_instrument(r.instrument_id).symbol} {r.category}"; as_of = d[d.doc_id.isin(rel)].available_time.max() + pd.Timedelta(minutes=1); qv = ix.embed_batch([q])[0]
    for filt in (False, True):
        for w in W:
            ids = hybrid_retrieve_indexed(q, d, as_of, ix, top_k=10, instrument_id=r.instrument_id if filt else None, query_vec=qv, bm25_weight=w).doc_id.tolist()
            rows.append({"q": i, "split": "tune" if i < 100 else "holdout", "filter": filt, "w": w, "ndcg5": ndcg_at_k(ids, rel, 5), "mrr": evaluate_ranking(ids, rel, 10).reciprocal_rank})
    if i % 25 == 0: print("q", i, flush=True)
r = pd.DataFrame(rows); out = {}
for filt in (False, True):
    sub = r[r["filter"] == filt]; tune = sub[sub.split == "tune"].groupby("w").ndcg5.mean(); best = float(tune.idxmax())
    ho = sub[sub.split == "holdout"].pivot(index="q", columns="w", values="ndcg5")
    dl = ho[best] - ho[1.0]
    out[f"filter={filt}"] = {"tune_ndcg5_by_w": tune.round(4).to_dict(), "chosen_w": best, "holdout_ndcg5_by_w": ho.mean().round(4).to_dict(),
                             "holdout_chosen_minus_classic_rrf(w=1)": {"mean": round(float(dl.mean()), 4), "se": round(float(dl.std(ddof=1) / np.sqrt(len(dl))), 4), "better": int((dl > 0).sum()), "worse": int((dl < 0).sum())}}
print(json.dumps(out)); Path("reports/fusion_study.json").write_text(json.dumps(out, indent=1)); print("done")
