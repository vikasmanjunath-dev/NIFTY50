"""After labelling: agreement of the human grades with the local LLM judges (and with the third LLM rater's grades where both exist), plus how much the
retrieval ranking conclusions change when human grades override model grades on the labelled pairs. Run: python3 scripts/human_agreement.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from nifty_analyser.documents.judge import load_human_labels, weighted_kappa
R = Path(__file__).resolve().parents[1] / "reports"
h = load_human_labels(R / "label_sheet.csv")
s = pd.read_csv(R / "label_sheet.csv", dtype={"query_id": str, "doc_id": str})
s["human"] = [h.get(q, {}).get(d, np.nan) for q, d in zip(s.query_id, s.doc_id)]
s = s.dropna(subset=["human"])
if len(s) < 30: raise SystemExit(f"only {len(s)} human labels; need >= 30 for a meaningful agreement estimate (python3 scripts/label_cli.py)")
out = {"n_human": len(s), "spearman_vs_local_judges": round(float(spearmanr(s.human, s.llm_grade)[0]), 3),
       "weighted_kappa_vs_rounded_judges": round(weighted_kappa(s.human.astype(int), np.rint(s.llm_grade).astype(int)), 3),
       "within1": round(float((np.abs(s.human - np.rint(s.llm_grade)) <= 1).mean()), 3), "human_mean": round(float(s.human.mean()), 3), "judge_mean": round(float(s.llm_grade.mean()), 3)}
c = pd.read_csv(R / "rater3_sample.csv", dtype={"query_id": str, "doc_id": str})[["query_id", "doc_id", "rater3_grade"]].merge(s, on=["query_id", "doc_id"])
if len(c) >= 20:
    out["n_overlap_with_rater3"] = len(c); out["spearman_human_vs_rater3"] = round(float(spearmanr(c.human, c.rater3_grade)[0]), 3)
print(json.dumps(out, indent=1)); (R / "human_agreement.json").write_text(json.dumps(out, indent=1))
