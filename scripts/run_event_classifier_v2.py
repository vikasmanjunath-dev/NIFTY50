"""v2 (train<=2024-12, val 2025-01..05 for ALL tuning, test>=2025-06 touched once). Announcement EVENT-TYPE classification with the category label masked out of the text (so it cannot be solved by lookup), temporal holdout
(train <= 2025-05-31, test after), three models: TF-IDF+LR, mxbai-embedding+LR, and a zero-shot Ollama LLM on a test subsample.
Labels = NSE `category` merged into coarse classes (documented in CLASSES). Rows whose text is empty after masking are excluded (reported)."""
import json, re, sys

from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.documents.index import make_ollama_embedder

CLASSES = {"investor_meet": r"investor meet|con\. call|analyst|investor presentation|link of recording|earnings call|transcript", "share_admin": r"share certificate|letter of confirmation|duplicate|depositories and participants|74\(5\)",
           "results": r"financial result|result updates|quarterly|audited|unaudited", "board_meeting": r"outcome of board|board meeting", "press_release": r"press release", "newspaper": r"newspaper",
           "esop_allotment": r"esop|esos|esps|allotment|stock option|restricted stock|rsu", "trading_window": r"trading window", "takeover_disclosure": r"takeover|regulation 31|substantial acquisition|disclosure under sebi",
           "credit_rating": r"credit rating|rating", "dividend_record_date": r"dividend|record date|book closure", "acquisition_mna": r"acquisition|amalgamation|merger|scheme of arrangement|demerger",
           "personnel": r"appointment|change in director|cessation|resignation|senior management|change in management", "shareholder_meeting": r"shareholders meeting|postal ballot|agm|egm|general meeting|scrutinizer|srutinizer"}
def group(c):
    c = c.lower()
    hits = [k for k, p in CLASSES.items() if re.search(p, c)]; return hits[0] if hits else None
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
d = d[~d.category.isin(["Updates", "General Updates", "General updates"])]; d["y"] = d.category.map(group); d = d.dropna(subset=["y"])
MERGE = "--merge-results-board" in sys.argv       # NSE moved results filings from "Financial Result Updates" to "Outcome of Board Meeting" in 2025 (taxonomy drift)
if MERGE: d["y"] = d.y.replace({"results": "results_or_board_outcome", "board_meeting": "results_or_board_outcome"})
d["t"] = pd.to_datetime(d.available_time, utc=True).dt.tz_convert("Asia/Kolkata")
def mask(r):
    t = r.text; t = re.sub(re.escape(r.category), " ", t, flags=re.I); t = re.sub(r"\b(has|have) (informed|submitted)( to)? the exchange( about| regarding| that)?\b", " ", t, flags=re.I)
    return " ".join(re.sub(r"[()'\"]", " ", t).split())
d["m"] = d.apply(mask, axis=1)
d["m"] = d.m.map(lambda t: " ".join(re.sub(r"^.*?\b(Limited|Ltd\.?|Corporation|Bank)\b[ ,.\-:]*", "", t, count=1, flags=re.I).split()) if re.search(r"\b(Limited|Ltd\.?)\b", t[:80], re.I) else t)   # strip leading company name: it carries no event information
n0 = len(d); d = d[d.m.str.len() >= 25]; excl = n0 - len(d)

from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
def cap(x, n, seed): return x.groupby("y", group_keys=False).apply(lambda g: g.sample(min(len(g), n), random_state=seed))
train, val, test = cap(d[d.t < "2025-01-01"], 2500, 1), cap(d[(d.t >= "2025-01-01") & (d.t < "2025-06-01")], 200, 2), cap(d[d.t >= "2025-06-01"], 150, 3)
res = {"excluded_empty_after_masking": int(excl), "n_train": len(train), "n_val": len(val), "n_test": len(test), "classes": sorted(d.y.unique())}
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300000)
tr = hstack([w.fit_transform(train.m), c.fit_transform(train.m)]).tocsr(); T = lambda x: hstack([w.transform(x.m), c.transform(x.m)]).tocsr()
emb = make_ollama_embedder("mxbai-embed-large"); E = lambda s_: np.vstack([emb(list(s_.iloc[i:i + 64])) for i in range(0, len(s_), 64)])
Etr, Eva, Ete = E(train.m), E(val.m), E(test.m)
models = {"lr_tfidf": (LogisticRegression(max_iter=3000, C=20).fit(tr, train.y), T), "svm_tfidf": (CalibratedClassifierCV(LinearSVC(C=1.0), cv=3).fit(tr, train.y), T),
          "lr_emb": (LogisticRegression(max_iter=4000, C=20).fit(Etr, train.y), None)}
cls = models["lr_tfidf"][0].classes_
def P(name, x, e): m, f = models[name]; return m.predict_proba(f(x) if f else e)
Pv = {n: P(n, val, Eva) for n in models}; Pt = {n: P(n, test, Ete) for n in models}
f1 = lambda y, p: f1_score(y, cls[p.argmax(1)], average="macro", labels=[k for k, v in val.y.value_counts().items() if v >= 30])
res["val_macro_f1"] = {n: round(f1(val.y, Pv[n]), 4) for n in models}
best = max(((a, b, 1 - a - b) for a in np.arange(0, 1.01, .1) for b in np.arange(0, 1.01 - a, .1)), key=lambda wts: f1(val.y, wts[0] * Pv["lr_tfidf"] + wts[1] * Pv["svm_tfidf"] + wts[2] * Pv["lr_emb"]))
res["ensemble_weights_chosen_on_val(lr_tfidf,svm,emb)"] = [round(x, 1) for x in best]; ens = lambda Pd: best[0] * Pd["lr_tfidf"] + best[1] * Pd["svm_tfidf"] + best[2] * Pd["lr_emb"]
res["val_ensemble_macro_f1"] = round(f1(val.y, ens(Pv)), 4); pt = cls[ens(Pt).argmax(1)]
ok_cls = [k for k, v in test.y.value_counts().items() if v >= 30]; res["classes_with_>=30_test_examples"] = sorted(ok_cls); res["classes_dropped_from_macro(<30 test examples)"] = sorted(set(cls) - set(ok_cls))
mf = lambda y, p: f1_score(y, p, average="macro", labels=ok_cls)
res["TEST"] = {"n": len(test), **{n: {"accuracy": round(accuracy_score(test.y, cls[Pt[n].argmax(1)]), 4), "macro_f1": round(mf(test.y, cls[Pt[n].argmax(1)]), 4)} for n in models}, "ensemble": {"accuracy": round(accuracy_score(test.y, pt), 4), "macro_f1": round(mf(test.y, pt), 4)}}
nat = d[d.t >= "2025-06-01"].sample(4000, random_state=4); Pn = ens({k: (models[k][0].predict_proba(models[k][1](nat)) if models[k][1] else models[k][0].predict_proba(E(nat.m))) for k in models})
res["TEST_natural_distribution"] = {"n": len(nat), "accuracy": round(accuracy_score(nat.y, cls[Pn.argmax(1)]), 4), "macro_f1": round(mf(nat.y, cls[Pn.argmax(1)]), 4)}
res["per_class_f1"] = {k: round(v["f1-score"], 3) for k, v in classification_report(test.y, pt, output_dict=True).items() if k in res["classes"]}
cm = pd.DataFrame(confusion_matrix(test.y, pt, labels=cls), index=cls, columns=cls); pairs = [(a, b, int(cm.loc[a, b])) for a in cls for b in cls if a != b and cm.loc[a, b] >= 8]
res["top_confusions(true,pred,n)"] = sorted(pairs, key=lambda x: -x[2])[:8]
Path("reports/event_classifier_v2" + ("_merged" if MERGE else "") + ".json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
