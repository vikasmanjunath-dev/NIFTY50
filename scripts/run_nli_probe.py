"""Probe: lexical guardrail vs pretrained-NLI guardrail vs both, on 150 real evidence packs x 8 planted answers (3 faithful, 5 fabricated). Threshold sweep for the NLI entailment minimum."""
"""Offline probe of the guardrails (no LLM): real evidence packs (4 filings of one company) + planted answers.
Faithful answers must pass; planted fabrications (invented event, wrong category, wrong company, invented figure) must be blocked. Reports pass/block rates per type."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.instrument_master import NIFTY50
from nifty_analyser.documents.evidence import build_evidence_pack
from nifty_analyser.documents.guardrails import verify_response
from nifty_analyser.documents.nli import nli_check
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
d["available_time"] = pd.to_datetime(d["available_time"], utc=True).dt.tz_convert("Asia/Kolkata"); d = d[d.text.str.len() > 20]
syms = {i.instrument_id: i.symbol for i in NIFTY50.values()}; rng = np.random.default_rng(5); ids = sorted(d.instrument_id.unique()); rows = []
for n in range(150):
    inst = ids[rng.integers(len(ids))]; g = d[d.instrument_id == inst].sort_values("available_time", ascending=False)
    cat = g.category.value_counts().index[rng.integers(min(5, g.category.nunique()))]; g = g[g.category == cat].head(4)
    if len(g) < 2: continue
    as_of = g.available_time.max() + pd.Timedelta(days=2); s = syms.get(inst, inst); c = cat; ds = [str(t.date()) for t in g.available_time]
    q = f"What did {s} announce about {c}?"; pack = build_evidence_pack(q, as_of, g.assign(retrieval_score=1.0), 4)
    other = syms[ids[(ids.index(inst) + 7) % len(ids)]] if ids[(ids.index(inst) + 7) % len(ids)] in syms else "ZZZCORP"
    base = f"{s} announced {c} on {ds[0]} and {ds[1]}."
    cases = {"faithful_dates": base, "faithful_partial": f"The most recent announcement was on {ds[0]} ({c}). However, the specific details are not stated in the evidence.",
             "faithful_cited": f"{s} made {len(g)} {c} announcements, the latest on {ds[0]} [1].",
             "fab_invented_event": base + " It also announced a merger with a rival firm and a major profit warning.", "fab_wrong_category": f"{s} announced a share buyback and a stock split on {ds[0]}.",
             "fab_wrong_company": f"{other} announced {c} on {ds[0]}.", "fab_invented_figure": base + " The amount involved was Rs 5000 crore.",
             "fab_unrelated": f"{s} opened a new steel plant in Gujarat last year."}
    for k, a in cases.items():
        r = nli_check(a, pack, entail_min=0.0, contra_max=1.0)
        rows.append({"type": k, "lex": verify_response(a, pack).passed, "min_entail": 1.0 if r["min_entail"] is None else r["min_entail"], "max_contra": 0.0 if r["max_contra"] is None else r["max_contra"]})

R = pd.DataFrame(rows); R.to_csv("reports/nli_probe_rows.csv", index=False); faith = R.type.str.startswith("faithful"); out = {}
def rates(mask): return {"faithful_pass": round(float(mask[faith].mean()), 3), "fabrication_block": round(float(1 - mask[~faith].mean()), 3), "per_type_pass": {t: round(float(mask[R.type == t].mean()), 3) for t in sorted(R.type.unique())}}
out["lexical_only"] = rates(R.lex)
for th in (0.1, 0.3, 0.5, 0.7, 0.9):
    nli = (R.min_entail >= th) & (R.max_contra <= 0.5); out[f"nli_only@{th}"] = rates(nli); out[f"lexical+nli@{th}"] = rates(R.lex & nli)
Path("reports/nli_probe.json").write_text(json.dumps(out, indent=1))
for k, v in out.items(): print(k, v["faithful_pass"], v["fabrication_block"], {t: p for t, p in v["per_type_pass"].items() if t.startswith("fab")})
