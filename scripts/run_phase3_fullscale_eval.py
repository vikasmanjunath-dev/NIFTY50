"""Full-corpus (58k docs) retrieval evaluation via the persistent index: RRF vs RRF+LLM rerank; relevant = same instrument AND category."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.documents.eval import evaluate_ranking, ndcg_at_k
from nifty_analyser.documents.index import VectorIndex, hybrid_retrieve_indexed
from nifty_analyser.documents.rerank import ollama_scorer, rerank

FILTER = "--instrument-filter" in sys.argv
SEED, N_Q, VAGUE = 42, 40, {"Updates", "General Updates"}
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
for c in ("event_time", "available_time"):
    d[c] = pd.to_datetime(d[c], utc=True).dt.tz_convert("Asia/Kolkata")
d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
dd = d[(d.text.str.len() > 20) & ~d.category.isin(VAGUE)]
g = dd.groupby(["instrument_id", "category"]).size()
pairs = g[(g >= 4) & (g <= 12)].reset_index()[["instrument_id", "category"]].sample(N_Q, random_state=SEED)   # same pairs as the 700-doc benchmark
ix = VectorIndex(DATA_PROCESSED_DIR / "vector_index"); scorer = ollama_scorer(); rows = []
for i, r in enumerate(pairs.itertuples()):
    rel = set(d[(d.instrument_id == r.instrument_id) & (d.category == r.category)].doc_id)     # full-corpus relevant set (all categories incl. text<=20 chars)
    q = f"{get_instrument(r.instrument_id).symbol} {r.category}"
    as_of = d[d.doc_id.isin(rel)].available_time.max() + pd.Timedelta(minutes=1)
    cand = hybrid_retrieve_indexed(q, d, as_of, ix, top_k=20, instrument_id=r.instrument_id if FILTER else None)
    rel_pit = rel & set(d[d.available_time <= as_of].doc_id)
    base, rr = cand.doc_id.tolist(), rerank(q, cand, scorer).doc_id.tolist()
    for name, ids in (("rrf", base), ("rrf+llm_rerank", rr)):
        m = evaluate_ranking(ids, rel_pit, k=10)
        rows.append({"q": i, "variant": name, "mrr": m.reciprocal_rank, "recall10": m.recall_at_k, "p5": evaluate_ranking(ids, rel_pit, k=5).precision_at_k, "ndcg5": ndcg_at_k(ids, rel_pit, 5)})
r = pd.DataFrame(rows); p = r.pivot(index="q", columns="variant", values=["mrr", "recall10", "p5", "ndcg5"])
out = {"corpus_docs": len(d), "n_queries": N_Q, "mean": r.groupby("variant")[["mrr", "recall10", "p5", "ndcg5"]].mean().round(4).to_dict("index")}
for m in ("mrr", "recall10", "p5", "ndcg5"):
    dl = p[m]["rrf+llm_rerank"] - p[m]["rrf"]
    out[f"delta_{m}"] = {"mean": round(float(dl.mean()), 4), "se": round(float(dl.std(ddof=1) / np.sqrt(len(dl))), 4), "better": int((dl > 0).sum()), "worse": int((dl < 0).sum())}
print(json.dumps(out, indent=1)); Path("reports/fullscale_retrieval_instrument_filter.json" if FILTER else "reports/fullscale_retrieval.json").write_text(json.dumps(out, indent=1)); print("done")
