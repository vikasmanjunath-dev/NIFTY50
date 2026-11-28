"""Replay the cached REAL model answers (all judged faithful) through lexical + NLI-veto guardrails; shows what the NLI veto would withhold."""
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


import nifty_analyser.documents.guardrails as G
from nifty_analyser.documents.evidence import EvidenceItem, EvidencePack
from nifty_analyser.documents.nli import nli_check
cache = json.loads(Path("reports/explain_eval_cache_qwen3-8b_routed_rerank.json").read_text()); cur = {}; ctx.generate = lambda q, pack: cur["t"]
rows = []
for q in [x for x in Qs if cache.get(str(x["k"]), {}).get("text")]:
    cur["t"] = cache[str(q["k"])]["text"]
    r = explain(EvidenceRequest(query=q["q"], instrument_id=q["inst"], as_of=q["as_of"].to_pydatetime(), top_k=4, window_days=q["win"], rerank=True), ctx)
    if not r.evidence.pack: continue
    pack = EvidencePack(**{**r.evidence.pack, "evidence_items": [EvidenceItem(**i) for i in r.evidence.pack["evidence_items"]]}); v = G.verify_response(cur["t"], pack); n = nli_check(cur["t"], pack, 0.0, 1.0)
    rows.append({"k": q["k"], "lex": v.passed, "abstained": v.abstained, "n_claims": n["n_claims"], "min_entail": n["min_entail"], "max_contra": n["max_contra"], "worst": (n["worst_claim"] or "")[:110]})
R = pd.DataFrame(rows); sub = R[R.n_claims > 0]
veto = lambda c, e: (sub.max_contra <= c) & (sub.min_entail >= e)
out = {"n_real_answers": len(R), "with_factual_claims": len(sub), "lexical_pass": int(R.lex.sum())}
for c, e in ((0.9, 0.003), (0.7, 0.003), (0.9, 0.001), (0.9, 0.0)):
    out[f"nli_veto_withholds@contra<={c},entail>={e}"] = int((~veto(c, e)).sum())
Path("reports/nli_replay.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))
print(sub[~veto(0.9, 0.003)][["k", "min_entail", "max_contra", "worst"]].to_string())
