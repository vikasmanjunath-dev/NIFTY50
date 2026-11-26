"""Filing QA accuracy with machine-verifiable gold answers: (1) dividend RECORD DATE, (2) financial-results PERIOD-END date. Questions are generated from announcements
whose text states the fact (regex), asked as a user would AFTER publication (as_of = publication + 1 day, instrument-filtered, window argv[3] days (default 45; 400 forces choosing the latest among several)) through the real
/v3/explain pipeline (retrieve -> qwen3:8b -> guardrails). Correct = the gold date appears among the dates in the released answer. Reports accuracy over all
questions (abstentions count as wrong), precision when answered, abstain rate, and retrieval hit-rate (gold-bearing doc in the evidence)."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.instrument_master import NIFTY50
from nifty_analyser.documents.index import VectorIndex, make_ollama_embedder
from nifty_analyser.documents.llm import generate_explanation
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.evidence_service import DocsContext, EvidenceRequest, explain
import json
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"; NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 40; WIN = int(sys.argv[3]) if len(sys.argv) > 3 else 45; SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 3; SUF = ("" if WIN == 45 else f"_w{WIN}") + ("" if SEED == 3 else f"_s{SEED}") + ("_recent" if "--recent" in sys.argv else "") + ("_hist" if "--historical" in sys.argv else "")
MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DATE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-/,]+([A-Za-z]{3,9})[\s\-/,]+(\d{4})")
DATE2 = re.compile(r"([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})")          # month-first: "December 31, 2022"
def dates(t):
    out = set()
    for mm, dd, yy in DATE2.findall(t or ""):
        if mm[:3].lower() in MON: out.add((int(yy), MON[mm[:3].lower()], int(dd)))
    for dd, mm, yy in DATE.findall(t or ""):
        if mm[:3].lower() in MON: out.add((int(yy), MON[mm[:3].lower()], int(dd)))
    return out
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
d["available_time"] = pd.to_datetime(d["available_time"], utc=True).dt.tz_convert("Asia/Kolkata"); d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
syms = {i.instrument_id: i.symbol for i in NIFTY50.values()}; rng = np.random.default_rng(9); qs = []
rec = d[d.text.str.contains(r"record date", case=False) & d.text.str.contains(r"dividend", case=False)]
res_ = d[d.text.str.contains(r"financial results for the period ended", case=False)]
for kind, pool, pat, tmpl in (("record_date", rec, r"record date[^.]*?\bis\s*(\d{1,2}[\s\-][A-Za-z]{3,9}[\s\-]\d{4})", "What is the record date for the dividend announced by {s}?"),
                              ("results_period", res_, r"period ended\s+(\w+\s+\d{1,2},?\s+\d{4})", "For which period did {s} submit its latest financial results?")):
    for r in pool.sample(len(pool), random_state=SEED).itertuples():
        m = re.search(pat, r.text, re.I); g = dates(m.group(1)) if m else set()
        if len(g) == 1 and r.available_time >= pd.Timestamp("2023-01-01", tz="Asia/Kolkata") and r.available_time < pd.Timestamp("2026-10-01", tz="Asia/Kolkata"):
            qs.append({"kind": kind, "inst": r.instrument_id, "doc": r.doc_id, "gold": next(iter(g)), "as_of": r.available_time + pd.Timedelta(days=1), "q": tmpl.format(s=syms.get(r.instrument_id, r.instrument_id))})
        if sum(x["kind"] == kind for x in qs) >= NQ: break
HIST = "--historical" in sys.argv        # questions name the filing's month+year; asked ~200 days later, so newer filings of the same kind exist
if HIST:
    MN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    dm = d.set_index("doc_id").available_time
    for x in qs:
        t = dm[x["doc"]]; x["when"] = f"{MN[t.month - 1]} {t.year}"; x["as_of"] = t + pd.Timedelta(days=200)
        x["q"] = x["q"].replace("announced by", "announced by").replace("latest financial results", "financial results") + f" (published in {x['when']})" if False else (
            f"What is the record date for the dividend announced by {syms.get(x['inst'], x['inst'])} in {x['when']}?" if x["kind"] == "record_date" else f"For which period did {syms.get(x['inst'], x['inst'])} submit its financial results in {x['when']}?")
    qs = [x for x in qs if x["as_of"] < pd.Timestamp("2026-11-24", tz="Asia/Kolkata")]
last = {}
def gen(q, pack): last["t"] = generate_explanation(q, pack, model=MODEL); return last["t"]
ctx = DocsContext(d, VectorIndex(DATA_PROCESSED_DIR / "vector_index", make_ollama_embedder()), AuditLog(Path("/tmp/filing_qa_audit.jsonl")), generate=gen)
rows = []
for q in qs:
    last.clear(); r = explain(EvidenceRequest(query=q["q"], instrument_id=q["inst"], as_of=q["as_of"].to_pydatetime(), top_k=5, window_days=WIN, prefer_recent=True if "--recent" in sys.argv else None), ctx)
    ids = [e["doc_id"] for e in (r.evidence.pack or {"evidence_items": []})["evidence_items"]]; txt = last.get("t") or ""
    rows.append({"kind": q["kind"], "hit": q["doc"] in ids, "status": r.status, "released": r.status in ("OK", "ABSTAIN") and bool(r.answer), "ok": r.status == "OK" and q["gold"] in dates(r.answer or ""), "strict": r.status == "OK" and q["gold"] in dates(r.answer or "") and len(dates(r.answer or "")) <= 2,
                 "withheld_correct": r.status == "BLOCKED" and q["gold"] in dates(txt), "answered": r.status == "OK", "q": q["q"], "gold": q["gold"], "ans": (r.answer or txt)[:140]}); print(len(rows), rows[-1]["kind"], rows[-1]["ok"], flush=True)
R = pd.DataFrame(rows)
def summ(x): return {"n": len(x), "accuracy_all": round(float(x.ok.mean()), 3), "strict_accuracy_all(<=2 dates in answer)": round(float(x.strict.mean()), 3), "precision_when_answered": round(float(x[x.answered].ok.mean()), 3) if x.answered.any() else None, "answered_rate": round(float(x.answered.mean()), 3),
                     "retrieval_hit_rate": round(float(x.hit.mean()), 3), "accuracy_given_retrieved": round(float(x[x.hit].ok.mean()), 3) if x.hit.any() else None, "blocked_but_correct": int(x.withheld_correct.sum())}
out = {"generator": MODEL, "overall": summ(R), **{k: summ(g) for k, g in R.groupby("kind")}, "status": R.status.value_counts().to_dict()}
Path(f"reports/filing_qa_{MODEL.replace(':', '-')}{SUF}.json").write_text(json.dumps(out, indent=1)); R.to_csv(f"reports/filing_qa_{MODEL.replace(':', '-')}{SUF}_rows.csv", index=False); print(json.dumps(out, indent=1))
