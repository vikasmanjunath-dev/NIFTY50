"""Full-corpus retrieval ablation (no LLM): component and depth/instrument-filter levers. Same 40 queries as the full-scale eval."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.documents.eval import evaluate_ranking, ndcg_at_k
from nifty_analyser.documents.index import VectorIndex, hybrid_retrieve_indexed

SEED, N_Q, VAGUE = 42, 40, {"Updates", "General Updates"}
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
for c in ("event_time", "available_time"):
    d[c] = pd.to_datetime(d[c], utc=True).dt.tz_convert("Asia/Kolkata")
d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
dd = d[(d.text.str.len() > 20) & ~d.category.isin(VAGUE)]; g = dd.groupby(["instrument_id", "category"]).size()
pairs = g[(g >= 4) & (g <= 12)].reset_index()[["instrument_id", "category"]].sample(N_Q, random_state=SEED)
ix = VectorIndex(DATA_PROCESSED_DIR / "vector_index")
variants = {"rrf (k=50)": dict(), "bm25 only": dict(mode="bm25"), "dense only": dict(mode="dense"), "rrf candidate_k=200": dict(candidate_k=200),
            "rrf + instrument filter": dict(instrument=True), "bm25 + instrument filter": dict(instrument=True, mode="bm25"), "dense + instrument filter": dict(instrument=True, mode="dense")}
rows = []
for i, r in enumerate(pairs.itertuples()):
    rel = set(d[(d.instrument_id == r.instrument_id) & (d.category == r.category)].doc_id)
    q = f"{get_instrument(r.instrument_id).symbol} {r.category}"; as_of = d[d.doc_id.isin(rel)].available_time.max() + pd.Timedelta(minutes=1)
    qv = ix.embed_batch([q])[0]
    for name, kw in variants.items():
        kw = dict(kw); inst = r.instrument_id if kw.pop("instrument", False) else None
        ids = hybrid_retrieve_indexed(q, d, as_of, ix, top_k=10, instrument_id=inst, query_vec=qv, **kw).doc_id.tolist()
        m = evaluate_ranking(ids, rel, k=10)
        rows.append({"q": i, "variant": name, "mrr": m.reciprocal_rank, "recall10": m.recall_at_k, "p5": evaluate_ranking(ids, rel, k=5).precision_at_k, "ndcg5": ndcg_at_k(ids, rel, 5)})
r = pd.DataFrame(rows); out = {"mean": r.groupby("variant")[["mrr", "recall10", "p5", "ndcg5"]].mean().round(4).to_dict("index")}
p = r.pivot(index="q", columns="variant", values="ndcg5")
for v in variants:
    if v != "rrf (k=50)":
        dl = p[v] - p["rrf (k=50)"]; out.setdefault("ndcg5_delta_vs_rrf", {})[v] = {"mean": round(float(dl.mean()), 4), "se": round(float(dl.std(ddof=1) / np.sqrt(len(dl))), 4), "better": int((dl > 0).sum()), "worse": int((dl < 0).sum())}
print(json.dumps(out)); Path("reports/retrieval_ablation.json").write_text(json.dumps(out, indent=1)); print("done")
