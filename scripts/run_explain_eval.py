"""Faithfulness study of /v3/explain: 50 questions (30 with a 365-day window, 20 with a 14-day window where relevant evidence often does not exist),
real llama3.2:3b answers through retrieve -> generate -> guardrails (the answer text is captured even when the guardrails withhold it). An independent judge
(qwen3:14b) grades every generated answer against its own evidence (2 fully supported / 1 minor unsupported claim / 0 unsupported or contradicted) and every
evidence pack for answerability (some item graded >= 2 for the question). Reports guardrail outcomes, faithfulness of released vs withheld answers,
leakage (unfaithful answers that were released) and abstention behaviour when the evidence cannot answer. Resumable cache: reports/explain_eval_cache.json."""
import json, re, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.instrument_master import NIFTY50
from nifty_analyser.documents.index import VectorIndex, make_ollama_embedder
from nifty_analyser.documents.judge import ollama_judge
from nifty_analyser.documents.llm import OLLAMA_GENERATE_URL, generate_explanation
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
def gen(q, pack): last["text"] = generate_explanation(q, pack, model=MODEL); return last["text"]
from nifty_analyser.documents.index import CategoryRouter
from nifty_analyser.documents.rerank import ollama_scorer
_emb = make_ollama_embedder()
ctx = DocsContext(d, VectorIndex(DATA_PROCESSED_DIR / "vector_index", _emb), AuditLog(Path("/tmp/explain_eval_audit.jsonl")), generate=gen,
                  router=CategoryRouter(d.category.unique().tolist(), _emb) if "--route" in sys.argv else None, scorer=ollama_scorer() if "--rerank" in sys.argv else None)
rng = np.random.default_rng(77); syms = {i.instrument_id: i.symbol for i in NIFTY50.values()}; ids = sorted(d.instrument_id.unique()); dates = pd.date_range("2023-06-01", "2026-10-31", tz="Asia/Kolkata")
Qs = [{"k": k, "inst": (inst := ids[rng.integers(len(ids))]), "topic": (t := TOPICS[k % len(TOPICS)]), "as_of": dates[rng.integers(len(dates))], "win": 365 if k < 30 else 14,
       "q": f"What did {syms.get(inst, inst)} announce about {t}?"} for k in range(50)]
FP = ("You verify answers against evidence.\nQuestion: {q}\nEvidence:\n{e}\nAnswer to check: {a}\n"
      "Grade the answer: 2 = every factual claim is supported by the evidence (or it correctly says the evidence is insufficient); 1 = mostly supported with a minor unsupported claim; "
      "0 = contains claims that are unsupported or contradicted by the evidence. Reply with one digit.")
def faith(q, ev, a):
    p = json.dumps({"model": "qwen3:14b", "think": False, "stream": False, "prompt": FP.format(q=q, e=ev, a=a), "options": {"temperature": 0, "num_predict": 4}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(OLLAMA_GENERATE_URL, data=p, headers={"Content-Type": "application/json"}), timeout=180)
    m = re.search(r"[0-2]", json.loads(r.read()).get("response", "")); return int(m.group()) if m else 0
judge = ollama_judge("qwen3:14b"); ABST = re.compile(r"insufficient|no (relevant )?(information|evidence)|not (mention|provide|contain|include)|does not (mention|provide|contain|include)|cannot (answer|find)", re.I)
rows = []
for q in Qs:
    key = str(q["k"])
    if key not in cache:
        last.clear(); r = explain(EvidenceRequest(query=q["q"], instrument_id=q["inst"], as_of=q["as_of"].to_pydatetime(), top_k=4, window_days=q["win"], rerank="--rerank" in sys.argv), ctx)
        items = r.evidence.pack["evidence_items"] if r.evidence.pack else []; text = r.answer if r.answer_source == "digest" else last.get("text")
        ev = "\n".join(f"[{i+1}] {e['claim_span'][:400]}" for i, e in enumerate(items))
        grades = judge(q["q"], [{"text": e["claim_span"], "when": e["publication_time"][:10], "company": syms.get(q["inst"], "?")} for e in items]) if items else []
        cache[key] = {"status": r.status, "source": r.answer_source, "n_ev": len(items), "text": text, "reason": (r.verification or {}).get("reason"), "answerable": bool(grades and max(grades) >= 2),
                      "faith": faith(q["q"], ev, text) if text else None}
        CACHE.write_text(json.dumps(cache))
    rows.append({**q, **cache[key]}); print(key, cache[key]["status"], cache[key]["answerable"], cache[key]["faith"], flush=True)
R = pd.DataFrame(rows); R["released"] = R.status.isin(["OK", "ABSTAIN"]); R["abst_phrase"] = R.text.fillna("").map(lambda t: bool(ABST.search(t)))
def f(x): return {"n": int(len(x)), "mean_faith": round(float(x.faith.mean()), 3) if len(x) and x.faith.notna().any() else None, "faith_0_rate": round(float((x.faith == 0).mean()), 3) if len(x) else None}
out = {"n_questions": len(R), "answer_source": R.get("source", pd.Series(dtype=str)).fillna("none").value_counts().to_dict(), "faith_of_digest_answers": f(R[R.get("source") == "digest"]) if "source" in R else None, "status": R.status.value_counts().to_dict(), "answerable_packs": int(R.answerable.sum()),
       "released": f(R[R.released & R.text.notna()]), "withheld_BLOCKED": f(R[R.status == "BLOCKED"]),
       "released_when_answerable": f(R[R.released & R.answerable]), "released_when_NOT_answerable": f(R[R.released & ~R.answerable & R.text.notna()]),
       "abstained_when_not_answerable": {"n": int((~R.answerable).sum()), "abstain_phrase_rate": round(float(R[~R.answerable].abst_phrase.mean()), 3) if (~R.answerable).any() else None},
       "leakage_unfaithful_released": int(((R.faith == 0) & R.released).sum()), "blocked_but_faithful": int(((R.faith >= 1) & (R.status == "BLOCKED")).sum()),
       "by_window": {str(w): R[R.win == w].status.value_counts().to_dict() for w in (365, 14)}}
out["generator"] = MODEL; out["answer_rate_when_answerable"] = round(float(((R.status == "OK") & R.answerable).sum() / max(R.answerable.sum(), 1)), 3)
Path(f"reports/explain_eval{SUF}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
