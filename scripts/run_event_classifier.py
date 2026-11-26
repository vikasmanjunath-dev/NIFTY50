"""Announcement EVENT-TYPE classification with the category label masked out of the text (so it cannot be solved by lookup), temporal holdout
(train <= 2025-05-31, test after), three models: TF-IDF+LR, mxbai-embedding+LR, and a zero-shot Ollama LLM on a test subsample.
Labels = NSE `category` merged into coarse classes (documented in CLASSES). Rows whose text is empty after masking are excluded (reported)."""
import json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.documents.index import make_ollama_embedder
from nifty_analyser.documents.llm import OLLAMA_GENERATE_URL
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
d["t"] = pd.to_datetime(d.available_time, utc=True).dt.tz_convert("Asia/Kolkata")
def mask(r):
    t = r.text; t = re.sub(re.escape(r.category), " ", t, flags=re.I); t = re.sub(r"\b(has|have) (informed|submitted)( to)? the exchange( about| regarding| that)?\b", " ", t, flags=re.I)
    return " ".join(re.sub(r"[()'\"]", " ", t).split())
d["m"] = d.apply(mask, axis=1); n0 = len(d); d = d[d.m.str.len() >= 25]; excl = n0 - len(d)
train, test = d[d.t < "2025-06-01"], d[d.t >= "2025-06-01"]
train = train.groupby("y", group_keys=False).apply(lambda g: g.sample(min(len(g), 600), random_state=1)); test = test.groupby("y", group_keys=False).apply(lambda g: g.sample(min(len(g), 150), random_state=2))
res = {"excluded_empty_after_masking": int(excl), "n_train": len(train), "n_test": len(test), "classes": sorted(d.y.unique()), "majority_baseline": round(float(test.y.value_counts(normalize=True).max()), 3)}
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); Xtr = tf.fit_transform(train.m); lr = LogisticRegression(max_iter=2000, C=10).fit(Xtr, train.y)
p = lr.predict(tf.transform(test.m)); res["tfidf_lr"] = {"accuracy": round(accuracy_score(test.y, p), 4), "macro_f1": round(f1_score(test.y, p, average="macro"), 4)}
emb = make_ollama_embedder("mxbai-embed-large"); E = lambda s: np.vstack([emb(list(s.iloc[i:i + 64])) for i in range(0, len(s), 64)])
Etr, Ete = E(train.m), E(test.m); lre = LogisticRegression(max_iter=3000, C=10).fit(Etr, train.y); pe = lre.predict(Ete)
res["embedding_lr"] = {"accuracy": round(accuracy_score(test.y, pe), 4), "macro_f1": round(f1_score(test.y, pe, average="macro"), 4)}
Xs = np.hstack([Etr, Xtr.toarray()[:, :0]]); proba = (lre.predict_proba(Ete) + lr.predict_proba(tf.transform(test.m))) / 2; pc = lr.classes_[proba.argmax(1)]
res["tfidf+embedding_ensemble"] = {"accuracy": round(accuracy_score(test.y, pc), 4), "macro_f1": round(f1_score(test.y, pc, average="macro"), 4)}
res["per_class_f1_ensemble"] = {k: round(v["f1-score"], 3) for k, v in classification_report(test.y, pc, output_dict=True).items() if k in res["classes"]}
LP = "Classify this Indian stock-exchange announcement into exactly one type from: {c}.\nAnnouncement: {t}\nAnswer with only the type name."
def ask(t, model="qwen3:14b"):
    q = json.dumps({"model": model, "think": False, "stream": False, "prompt": LP.format(c=", ".join(res["classes"]), t=t[:500]), "options": {"temperature": 0, "num_predict": 8}}).encode()
    a = json.loads(urllib.request.urlopen(urllib.request.Request(OLLAMA_GENERATE_URL, data=q, headers={"Content-Type": "application/json"}), timeout=180).read())["response"].strip().lower()
    return next((c for c in res["classes"] if c in a), "other")
sub = test.sample(min(300, len(test)), random_state=3)
with ThreadPoolExecutor(3) as ex: pl = list(ex.map(ask, sub.m))
res["llm_zero_shot_qwen3_14b"] = {"n": len(sub), "accuracy": round(accuracy_score(sub.y, pl), 4)}
res["ensemble_on_same_300"] = round(accuracy_score(sub.y, pd.Series(pc, index=test.index).loc[sub.index]), 4)
Path("reports/event_classifier.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
