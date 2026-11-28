"""Train + evaluate + save the production event classifier.
1) fit on <= 2024-12 -> choose the abstain threshold on 2025-01..05 (smallest threshold with >= 99% accepted accuracy);
2) refit on <= 2025-05 with that threshold -> score the untouched test period (>= 2025-06): accuracy without abstention, and accuracy/coverage at the threshold;
3) save data/processed/models/event_classifier.joblib (metrics inside). Use --final to refit on ALL data (no test) for deployment after evaluating."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.documents.event_classifier import MIN_CONTENT_CHARS, VAGUE, EventClassifier, label_for, mask_text
FINAL = "--final" in sys.argv
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[(d.quality_flag == "PASS") & ~d.category.isin(VAGUE)].copy()
d["y"] = d.category.map(label_for); d = d.dropna(subset=["y"]); d["t"] = pd.to_datetime(d.available_time, utc=True).dt.tz_convert("Asia/Kolkata")
d["m"] = [mask_text(t, c) for t, c in zip(d.text, d.category)]; d = d[d.m.str.len() >= MIN_CONTENT_CHARS]
cap = lambda x, n, s: x.groupby("y", group_keys=False).apply(lambda g: g.sample(min(len(g), n), random_state=s))
A, V = cap(d[d.t < "2025-01-01"], 2500, 1), d[(d.t >= "2025-01-01") & (d.t < "2025-06-01")]
a = EventClassifier().fit(A.m, A.y); PV = a.predict_proba(V.m); pv, cv = a.classes_ and np.array(a.classes_)[PV.argmax(1)], PV.max(1)
ok = (pv == V.y.to_numpy())
tau = next((t for t in np.arange(0.30, 0.995, 0.01) if (cv >= t).sum() >= 50 and ok[cv >= t].mean() >= 0.99), 0.9)
B = cap(d[d.t < ("2099-01-01" if FINAL else "2025-06-01")], 2500, 1); m = EventClassifier(threshold=float(tau)).fit(B.m, B.y)
out = {"final_refit_on_all_data": FINAL, "n_train": len(B), "threshold": round(float(tau), 3), "val_accepted_accuracy_at_threshold": round(float(ok[cv >= tau].mean()), 4), "val_coverage": round(float((cv >= tau).mean()), 4)}
if not FINAL:
    T = cap(d[d.t >= "2025-06-01"], 150, 3); PT = m.predict_proba(T.m); pt = np.array(m.classes_)[PT.argmax(1)]; ct = PT.max(1); okt = pt == T.y.to_numpy(); acc = ct >= tau
    nat = d[d.t >= "2025-06-01"].sample(4000, random_state=4); PN = m.predict_proba(nat.m); pn = np.array(m.classes_)[PN.argmax(1)]; cn = PN.max(1)
    out["TEST"] = {"n": len(T), "accuracy_no_abstain": round(float(okt.mean()), 4), "accuracy_when_accepted": round(float(okt[acc].mean()), 4), "coverage": round(float(acc.mean()), 4),
                   "natural_n": len(nat), "natural_accuracy_no_abstain": round(float((pn == nat.y.to_numpy()).mean()), 4), "natural_accuracy_when_accepted": round(float((pn == nat.y.to_numpy())[cn >= tau].mean()), 4), "natural_coverage": round(float((cn >= tau).mean()), 4)}
    out["TEST"]["accuracy_vs_coverage"] = {f"conf>={t}": {"coverage": round(float((ct >= t).mean()), 3), "accuracy": round(float(okt[ct >= t].mean()), 4)} for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
m.meta = out; m.save(DATA_PROCESSED_DIR / "models" / ("event_classifier_final.joblib" if FINAL else "event_classifier.joblib"))
Path("reports/event_classifier_prod" + ("_final" if FINAL else "") + ".json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
