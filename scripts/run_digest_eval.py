"""Digest-fallback gate: rebuild the real evidence packs (routing + qwen3:8b rerank) for the explain-study questions, read the cached model answers and the judge answerability, sweep the reranker-score gate; tune on even ids, confirm on odd."""
"""Replays the CACHED real model answers of the explain study through the guardrails at several overlap thresholds (packs rebuilt with routing+rerank, no new generation). Reports, per threshold: real answers released/withheld (all were judged faithful), and the planted-fabrication block rate from the probe."""
"""Faithfulness study of /v3/explain: 50 questions (30 with a 365-day window, 20 with a 14-day window where relevant evidence often does not exist),
real llama3.2:3b answers through retrieve -> generate -> guardrails (the answer text is captured even when the guardrails withhold it). An independent judge
(qwen3:14b) grades every generated answer against its own evidence (2 fully supported / 1 minor unsupported claim / 0 unsupported or contradicted) and every
evidence pack for answerability (some item graded >= 2 for the question). Reports guardrail outcomes, faithfulness of released vs withheld answers,
leakage (unfaithful answers that were released) and abstention behaviour when the evidence cannot answer. Resumable cache: reports/explain_eval_cache.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.instrument_master import NIFTY50
from nifty_analyser.documents.index import VectorIndex, make_ollama_embedder


from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.evidence_service import DocsContext, EvidenceRequest, explain
MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"; SUF = ("" if MODEL == "llama3.2:3b" else "_" + MODEL.replace(":", "-")) + ("_routed" if "--route" in sys.argv else "") + ("_rerank" if "--rerank" in sys.argv else "")
CACHE = Path(f"reports/explain_eval_cache{SUF}.json"); cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
TOPICS = ["dividend declaration or record date", "acquisition of a company or stake", "credit rating changes", "appointment or resignation of directors or executives",
          "quarterly financial results", "raising funds through debentures or bonds", "regulatory penalties or fines", "new plant or capacity expansion",
          "merger, demerger or scheme of arrangement", "share buyback"]
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
d["available_time"] = pd.to_datetime(d["available_time"], utc=True).dt.tz_convert("Asia/Kolkata")
d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
last = {}

from nifty_analyser.documents.index import CategoryRouter
from nifty_analyser.documents.rerank import ollama_scorer
_emb = make_ollama_embedder()
ctx = DocsContext(d, VectorIndex(DATA_PROCESSED_DIR / "vector_index", _emb), AuditLog(Path("/tmp/explain_eval_audit.jsonl")), generate=None,
                  router=CategoryRouter(d.category.unique().tolist(), _emb) if "--route" in sys.argv else None, scorer=ollama_scorer() if "--rerank" in sys.argv else None)
rng = np.random.default_rng(77); syms = {i.instrument_id: i.symbol for i in NIFTY50.values()}; ids = sorted(d.instrument_id.unique()); dates = pd.date_range("2023-06-01", "2026-10-31", tz="Asia/Kolkata")
Qs = [{"k": k, "inst": (inst := ids[rng.integers(len(ids))]), "topic": (t := TOPICS[k % len(TOPICS)]), "as_of": dates[rng.integers(len(dates))], "win": 365 if k < 30 else 14,
       "q": f"What did {syms.get(inst, inst)} announce about {t}?"} for k in range(50)]


from nifty_analyser.documents.digest import best_rerank
from nifty_analyser.documents.evidence import EvidenceItem, EvidencePack
cache = json.loads(Path("reports/explain_eval_cache_qwen3-8b_routed_rerank.json").read_text()); cur = {}; ctx.generate = lambda q, pack: cur["t"]; rows = []
for q in Qs:
    c = cache.get(str(q["k"]))
    if not c or not c.get("text"): continue
    cur["t"] = c["text"]; r = explain(EvidenceRequest(query=q["q"], instrument_id=q["inst"], as_of=q["as_of"].to_pydatetime(), top_k=4, window_days=q["win"], rerank=True), ctx)
    pk = EvidencePack(**{**r.evidence.pack, "evidence_items": [EvidenceItem(**i) for i in r.evidence.pack["evidence_items"]]}) if r.evidence.pack else None
    rows.append({"k": q["k"], "answerable": bool(c["answerable"]), "llm_answered": c["status"] == "OK", "best": best_rerank(pk) if pk else None})
R = pd.DataFrame(rows).fillna({"best": 0.0}); out = {"n": len(R), "baseline_answer_rate_when_answerable": round(float(R[R.answerable].llm_answered.mean()), 3), "baseline_answered_when_NOT_answerable": round(float(R[~R.answerable].llm_answered.mean()), 3)}
def at(df, g):
    ans = df.llm_answered | (df.best >= g); a, n = df[df.answerable], df[~df.answerable]
    return {"ans_when_answerable": round(float(ans[a.index].mean()), 3), "ans_when_NOT": round(float(ans[n.index].mean()), 3), "n_a": len(a), "n_not": len(n)}
for g in (5, 6, 7, 8, 9, 10):
    out[f"gate>={g}"] = {"tune_even": at(R[R.k % 2 == 0], g), "holdout_odd": at(R[R.k % 2 == 1], g), "all": at(R, g)}
Path("reports/digest_eval.json").write_text(json.dumps(out, indent=1)); R.to_csv("reports/digest_eval_rows.csv", index=False); print(json.dumps(out))
