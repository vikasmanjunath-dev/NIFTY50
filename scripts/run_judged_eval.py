"""Content-judged retrieval evaluation: 60 natural-language queries (30 company-specific, 30 market-wide), pooled candidates
from every system, graded 0-3 relevance from two independent local LLM judges (agreement reported), systems compared on
graded nDCG@5 with paired bootstrap-free SEs. Resumable (cache in reports/judged_cache.json).
Caveats are written into the report: LLM judges are not humans; a reranker from the same model family as a judge is favoured."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.documents.index import VectorIndex, hybrid_retrieve_indexed, make_ollama_embedder, retrieve
from nifty_analyser.documents.judge import consensus, export_label_sheet, graded_ndcg_at_k, ollama_judge, weighted_kappa
from nifty_analyser.documents.rerank import ollama_scorer, rerank

JUDGES = ["qwen3:14b", "gemma3:12b-it-qat"]
RERANKERS = {"rerank_llama3.2:3b": ollama_scorer("llama3.2:3b"), "rerank_qwen3:8b": ollama_scorer("qwen3:8b", think=False)}
if "--cross-encoders" in sys.argv:          # trained cross-encoders (downloads from Hugging Face on first use)
    from nifty_analyser.documents.crossencoder import cross_encoder_scorer
    RERANKERS.update({"rerank_ce_minilm": cross_encoder_scorer("minilm"), "rerank_ce_bge": cross_encoder_scorer("bge")})
CACHE = Path("reports/judged_cache.json"); cache = json.loads(CACHE.read_text()) if CACHE.exists() else {"judge": {}, "rerank": {}}
save = lambda: CACHE.write_text(json.dumps(cache))
TOPICS = ["dividend declaration or record date for dividend", "acquisition of a company or stake", "credit rating upgrade or downgrade",
          "appointment or resignation of a director or senior executive", "quarterly financial results", "raising funds through debentures or bonds",
          "allotment of shares under employee stock options", "regulatory penalty, fine or show-cause notice", "new plant, capacity expansion or capital expenditure",
          "merger, demerger or scheme of arrangement", "share buyback", "fire, accident or operational disruption at a facility"]
rng = np.random.default_rng(2026)

d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
d["available_time"] = pd.to_datetime(d["available_time"], utc=True).dt.tz_convert("Asia/Kolkata")
d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
ix = VectorIndex(DATA_PROCESSED_DIR / "vector_index", make_ollama_embedder("mxbai-embed-large"), model="mxbai-embed-large")
ids = sorted(d.instrument_id.unique()); dates = pd.date_range("2023-01-01", "2026-10-31", freq="D", tz="Asia/Kolkata")
queries = []
for i in range(60):
    topic = TOPICS[i % len(TOPICS)]; inst = ids[rng.integers(len(ids))] if i < 30 else None
    as_of = dates[rng.integers(len(dates))]; sym = get_instrument(inst).symbol if inst else None
    queries.append({"qid": str(i), "topic": topic, "inst": inst, "as_of": as_of, "judge_q": f"{topic}" + (f" ({sym})" if sym else ""),
                    "window": pd.Timedelta(days=365 if inst else 120)})

from nifty_analyser.documents.index import CategoryRouter
router = CategoryRouter(d.category.unique().tolist(), make_ollama_embedder("mxbai-embed-large"))
ROUTED = {f"routed_n{n}_w{w:g}": (n, w) for n in (1, 2) for w in (1.0, 3.0)} | {"routed_n2_w10": (2, 10.0), "routed_n3_w3": (3, 3.0)}
pools, runs = {}, {}
t0 = time.time()
for q in queries:
    docs = d[(d.available_time > q["as_of"] - q["window"]) & (d.available_time <= q["as_of"])]
    if q["inst"] is not None:                                 # production path strips the ticker itself; here the topic text has none
        pass
    kw = dict(documents=docs, as_of=q["as_of"], index=ix, instrument_id=q["inst"])
    runs[q["qid"]] = {"bm25": hybrid_retrieve_indexed(q["topic"], mode="bm25", top_k=10, **kw).doc_id.tolist(),
                      "dense": hybrid_retrieve_indexed(q["topic"], mode="dense", top_k=10, **kw).doc_id.tolist(),
                      "rrf_w1": hybrid_retrieve_indexed(q["topic"], mode="rrf", bm25_weight=1.0, top_k=10, **kw).doc_id.tolist()}
    cand = retrieve(q["topic"], top_k=20, **kw)               # auto-tuned defaults, depth 20: the candidate set rerankers reorder
    runs[q["qid"]]["auto"] = cand.doc_id.tolist()[:10]
    runs[q["qid"]]["auto_recent"] = retrieve(q["topic"], top_k=10, recency_weight=0.5, **kw).doc_id.tolist()      # new service default for queries with no time reference
    for rn, (n_, w_) in ROUTED.items():
        runs[q["qid"]][rn] = retrieve(q["topic"], top_k=10, router=router, route_top_n=n_, route_weight=w_, **kw).doc_id.tolist()
    candR = retrieve(q["topic"], top_k=20, router=router, route_top_n=2, route_weight=3.0, **kw)          # routed candidates -> LLM rerank
    key = f"rerank_qwen3:8b+routed|{q['qid']}"
    if key not in cache["rerank"]:
        cache["rerank"][key] = rerank(q["topic"], candR, RERANKERS["rerank_qwen3:8b"], top_k=10).doc_id.tolist() if len(candR) else []
    runs[q["qid"]]["rerank_qwen3:8b+routed"] = cache["rerank"][key]; extra_pool = set(candR.doc_id)
    for name, sc in RERANKERS.items():
        key = f"{name}|{q['qid']}"
        if key not in cache["rerank"]:
            cache["rerank"][key] = rerank(q["topic"], cand, sc, top_k=10).doc_id.tolist() if len(cand) else []
        runs[q["qid"]][name] = cache["rerank"][key]
    pools[q["qid"]] = sorted(set(cand.doc_id) | extra_pool | {x for k, v in runs[q["qid"]].items() for x in v})
    save()
print(f"retrieval+rerank done {time.time()-t0:.0f}s; pooled docs/query mean {np.mean([len(v) for v in pools.values()]):.1f}", flush=True)

meta = d.set_index("doc_id")
for jn in JUDGES:
    judge = ollama_judge(jn)
    for q in queries:
        todo = [x for x in pools[q["qid"]] if f"{jn}|{q['qid']}|{x}" not in cache["judge"]]
        if todo:
            docs = [{"text": meta.at[x, "text"], "when": str(meta.at[x, "available_time"].date()), "company": get_instrument(meta.at[x, "instrument_id"]).symbol} for x in todo]
            for x, g in zip(todo, judge(q["judge_q"], docs)):
                cache["judge"][f"{jn}|{q['qid']}|{x}"] = g
        save()
    print(f"judged with {jn} {time.time()-t0:.0f}s", flush=True)

grades = {q["qid"]: {jn: {x: cache["judge"][f"{jn}|{q['qid']}|{x}"] for x in pools[q["qid"]]} for jn in JUDGES} for q in queries}
a = [g for qid in grades for g in grades[qid][JUDGES[0]].values()]; b = [g for qid in grades for g in grades[qid][JUDGES[1]].values()]
agree = {"weighted_kappa": round(weighted_kappa(a, b), 3), "exact": round(float(np.mean(np.array(a) == np.array(b))), 3), "within1": round(float(np.mean(np.abs(np.array(a) - np.array(b)) <= 1)), 3),
         "mean_grade": {JUDGES[0]: round(float(np.mean(a)), 3), JUDGES[1]: round(float(np.mean(b)), 3)}, "n_judgments": len(a)}
systems = list(next(iter(runs.values())).keys()); rows = []
for q in queries:
    gains = consensus(grades[q["qid"]]); gains_by_judge = {jn: grades[q["qid"]][jn] for jn in JUDGES}
    for s in systems:
        r = {"q": q["qid"], "kind": "company" if q["inst"] else "market", "system": s, "ndcg5": graded_ndcg_at_k(runs[q["qid"]][s], gains, 5)}
        for jn in JUDGES:
            r[f"ndcg5_{jn}"] = graded_ndcg_at_k(runs[q["qid"]][s], gains_by_judge[jn], 5)
        rows.append(r)
R = pd.DataFrame(rows).dropna(subset=["ndcg5"])
piv = R.pivot(index="q", columns="system", values="ndcg5").dropna()
out = {"agreement": agree, "n_queries_scored": int(len(piv)), "mean_ndcg5": piv.mean().round(4).to_dict(),
       "by_kind": {k: R[R.kind == k].pivot(index="q", columns="system", values="ndcg5").mean().round(4).to_dict() for k in ("company", "market")},
       "per_judge": {jn: R.pivot(index="q", columns="system", values=f"ndcg5_{jn}").mean().round(4).to_dict() for jn in JUDGES}, "paired_vs_auto": {}}
for s in systems:
    if s == "auto": continue
    dl = piv[s] - piv["auto"]
    out["paired_vs_auto"][s] = {"mean": round(float(dl.mean()), 4), "se": round(float(dl.std(ddof=1) / np.sqrt(len(dl))), 4), "better": int((dl > 0).sum()), "worse": int((dl < 0).sum())}
out["paired_vs_auto_by_half"] = {}
for half, sel in (("tune_even_qids", lambda i: int(i) % 2 == 0), ("holdout_odd_qids", lambda i: int(i) % 2 == 1)):
    ph = piv[[sel(i) for i in piv.index]]
    out["paired_vs_auto_by_half"][half] = {s_: {"mean": round(float((ph[s_] - ph["auto"]).mean()), 4), "se": round(float((ph[s_] - ph["auto"]).std(ddof=1) / np.sqrt(len(ph))), 4)} for s_ in systems if s_.startswith("routed")} | {"n": len(ph)}
out["paired_vs_auto_company_only"] = {}
pc = R[R.kind == "company"].pivot(index="q", columns="system", values="ndcg5").dropna()
for s_ in systems:
    if s_ != "auto":
        dl = pc[s_] - pc["auto"]; out["paired_vs_auto_company_only"][s_] = {"mean": round(float(dl.mean()), 4), "se": round(float(dl.std(ddof=1) / np.sqrt(len(dl))), 4), "n": len(dl)}
Path("reports/judged_eval.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
rows = [{"query_id": q["qid"], "query": q["judge_q"], "as_of": str(q["as_of"].date()), "doc_id": x, "company": get_instrument(meta.at[x, "instrument_id"]).symbol,
         "text": meta.at[x, "text"], "llm_grade": float(np.mean([grades[q["qid"]][jn][x] for jn in JUDGES]))} for q in queries for x in pools[q["qid"]]]
export_label_sheet(pd.DataFrame(rows), Path("reports/label_sheet.csv"), sample_per_query=12); print("label sheet exported")
