"""Does LLM reranking help? Stricter benchmark than the same-instrument proxy: a document is relevant only if it has the
query's instrument AND category (structured labels the retriever never sees; it sees text only). Paired by query."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.documents.embeddings import EmbeddingCache, embed_many
from nifty_analyser.documents.eval import evaluate_ranking, ndcg_at_k
from nifty_analyser.documents.rerank import RERANK_MODEL, ollama_scorer, rerank
from nifty_analyser.documents.retrieval import hybrid_retrieve

SEED, N_Q, POOL, K = 42, 40, 700, 5
VAGUE = {"Updates", "General Updates"}
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet")
d = d[d.quality_flag == "PASS"].copy()
for c in ("event_time", "available_time"):
    d[c] = pd.to_datetime(d[c], utc=True).dt.tz_convert("Asia/Kolkata")
d = d[(d.text.str.len() > 20) & ~d.category.isin(VAGUE)].reset_index(drop=True)
g = d.groupby(["instrument_id", "category"]).size()
pairs = g[(g >= 4) & (g <= 12)].reset_index()[["instrument_id", "category"]].sample(N_Q, random_state=SEED)
rel_docs = pd.concat([d[(d.instrument_id == r.instrument_id) & (d.category == r.category)] for r in pairs.itertuples()])
pool = pd.concat([rel_docs, d.drop(rel_docs.index, errors="ignore").sample(POOL - len(rel_docs), random_state=SEED)]).drop_duplicates("doc_id").reset_index(drop=True)
cache = EmbeddingCache(); print(f"pool {len(pool)}, embedding...", flush=True); embed_many(pool.text.tolist(), cache=cache); cache.save()
scorer = ollama_scorer(RERANK_MODEL)
rows = []
for i, r in enumerate(pairs.itertuples()):
    rel = set(pool[(pool.instrument_id == r.instrument_id) & (pool.category == r.category)].doc_id)
    q = f"{get_instrument(r.instrument_id).symbol} {r.category}"
    as_of = pool[pool.doc_id.isin(rel)].available_time.max() + pd.Timedelta(minutes=1)
    cand = hybrid_retrieve(q, pool, as_of, top_k=20, embedding_cache=cache)
    base = cand.doc_id.tolist()
    t = time.time(); rr = rerank(q, cand, scorer).doc_id.tolist()
    for name, ids in (("rrf", base), ("rrf+llm_rerank", rr)):
        m = evaluate_ranking(ids, rel, k=10)
        rows.append({"q": i, "variant": name, "mrr": m.reciprocal_rank, "p5": evaluate_ranking(ids, rel, k=K).precision_at_k, "ndcg5": ndcg_at_k(ids, rel, K), "n_rel": len(rel), "cand_has_rel": int(any(x in rel for x in base))})
    print(f"q{i} {q[:50]!r} n_rel={len(rel)} {time.time()-t:.1f}s", flush=True)
r = pd.DataFrame(rows)
out = {"n_queries": N_Q, "pool": len(pool), "model": RERANK_MODEL, "mean": r.groupby("variant")[["mrr", "p5", "ndcg5"]].mean().round(4).to_dict("index")}
p = r.pivot(index="q", columns="variant", values=["mrr", "p5", "ndcg5"])
for m in ("mrr", "p5", "ndcg5"):
    dlt = p[m]["rrf+llm_rerank"] - p[m]["rrf"]
    out[f"delta_{m}"] = {"mean": round(float(dlt.mean()), 4), "se": round(float(dlt.std(ddof=1) / np.sqrt(len(dlt))), 4), "better": int((dlt > 0).sum()), "worse": int((dlt < 0).sum())}
print(json.dumps(out, indent=1)); Path("reports/rerank_eval.json").write_text(json.dumps(out, indent=1)); print("done")
