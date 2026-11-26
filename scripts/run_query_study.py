"""BM25-weight grid beyond 4 and deterministic query rewrites, tune/holdout (100/100, fresh seed). Embedding index selectable via argv[1] (default mxbai)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.documents.eval import ndcg_at_k
from nifty_analyser.documents.index import VectorIndex, hybrid_retrieve_indexed, make_ollama_embedder

model = sys.argv[1] if len(sys.argv) > 1 else "mxbai-embed-large"
dirn = "vector_index" if model == "mxbai-embed-large" else f"vector_index_{model.split(':')[0]}"
VAGUE = {"Updates", "General Updates"}
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
d["available_time"] = pd.to_datetime(d["available_time"], utc=True).dt.tz_convert("Asia/Kolkata")
d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
dd = d[(d.text.str.len() > 20) & ~d.category.isin(VAGUE)]; g = dd.groupby(["instrument_id", "category"]).size()
pairs = g[(g >= 4) & (g <= 12)].reset_index()[["instrument_id", "category"]].sample(200, random_state=11)
ix = VectorIndex(DATA_PROCESSED_DIR / dirn, make_ollama_embedder(model), model=model)
forms = {"symbol+category": lambda s, c: f"{s} {c}", "category only": lambda s, c: c, "symbol: category announcement": lambda s, c: f"{s}: {c} announcement"}
cfgs = [("filtered", w, f) for w in (4, 16, 64) for f in ("symbol+category", "category only")] + [("unfiltered dense", 0.0, f) for f in forms]
rows = []
for i, r in enumerate(pairs.itertuples()):
    rel = set(d[(d.instrument_id == r.instrument_id) & (d.category == r.category)].doc_id)
    as_of = d[d.doc_id.isin(rel)].available_time.max() + pd.Timedelta(minutes=1); sym = get_instrument(r.instrument_id).symbol
    qvs = {f: ix.embed_batch([forms[f](sym, r.category)])[0] for f in forms}
    for kind, w, f in cfgs:
        ids = hybrid_retrieve_indexed(forms[f](sym, r.category), d, as_of, ix, top_k=10, instrument_id=r.instrument_id if kind == "filtered" else None, query_vec=qvs[f], bm25_weight=w).doc_id.tolist()
        rows.append({"q": i, "split": "tune" if i < 100 else "holdout", "cfg": f"{kind} w={w:g} | {f}", "ndcg5": ndcg_at_k(ids, rel, 5)})
r = pd.DataFrame(rows); out = {"model": model}
for pref in ("filtered", "unfiltered"):
    sub = r[r.cfg.str.startswith(pref)]; tune = sub[sub.split == "tune"].groupby("cfg").ndcg5.mean(); best = tune.idxmax()
    ho = sub[sub.split == "holdout"].pivot(index="q", columns="cfg", values="ndcg5"); base = [c for c in ho.columns if "symbol+category" in c and (c.startswith("filtered w=4") or pref == "unfiltered")][0]
    dl = ho[best] - ho[base]
    out[pref] = {"tune": tune.round(4).to_dict(), "chosen": best, "holdout": ho.mean().round(4).to_dict(), "baseline": base, "holdout_chosen_minus_baseline": {"mean": round(float(dl.mean()), 4), "se": round(float(dl.std(ddof=1) / np.sqrt(len(dl))), 4), "better": int((dl > 0).sum()), "worse": int((dl < 0).sum())}}
print(json.dumps(out)); Path(f"reports/query_study_{model.split(':')[0]}.json").write_text(json.dumps(out, indent=1)); print("done")
