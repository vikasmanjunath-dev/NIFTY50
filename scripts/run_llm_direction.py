"""Can a local LLM call next-day direction from a company announcement? Strictly point-in-time: the model sees only the announcement text, company and
date; entry = first close at/after publication (after 15:30 -> next day's close), exit = the following close (1 trading day). N announcements in
market-relevant categories, 2023-06..2026-10. Reports accuracy, AUC, accuracy by stated confidence, vs the 'always-majority' baseline."""
import json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.instrument_master import NIFTY50
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.documents.llm import OLLAMA_GENERATE_URL
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:14b"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
CATS = ["Financial Result Updates", "Dividend", "Acquisition", "Credit Rating", "Outcome of Board Meeting", "Press Release", "Allotment of Securities", "Appointment", "Change in Director(s)"]
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[(d.quality_flag == "PASS") & d.category.isin(CATS) & (d.text.str.len() > 60)].copy()
d["t"] = pd.to_datetime(d.available_time, utc=True).dt.tz_convert("Asia/Kolkata"); d = d[(d.t >= "2023-06-01") & (d.t < "2026-10-25")]
b = load_universe_bars().pivot_table(index="event_time", columns="instrument_id", values="close").sort_index(); b.index = pd.DatetimeIndex(b.index).tz_convert("Asia/Kolkata").normalize().tz_localize(None)
sym = {i.instrument_id: i.symbol for i in NIFTY50.values()}; d = d.sample(min(N * 3, len(d)), random_state=5); rows = []
for r in d.itertuples():
    if r.instrument_id not in b.columns: continue
    day = r.t.tz_localize(None).normalize() + pd.Timedelta(days=(1 if r.t.hour * 60 + r.t.minute >= 15 * 60 + 30 else 0))
    i = b.index.searchsorted(day)
    if i + 1 >= len(b) or b.iloc[i][r.instrument_id] != b.iloc[i][r.instrument_id] or b.iloc[i + 1][r.instrument_id] != b.iloc[i + 1][r.instrument_id]: continue
    rows.append({"inst": r.instrument_id, "cat": r.category, "when": str(r.t.date()), "text": r.text[:600], "ret": float(np.log(b.iloc[i + 1][r.instrument_id] / b.iloc[i][r.instrument_id]))})
    if len(rows) >= N: break
P = ("You are an equity analyst. A Nifty 50 company published this exchange announcement on {when}.\nCompany: {c}\nCategory: {cat}\nAnnouncement: {t}\n"
     "Estimate the probability (0-100) that the stock's next-day closing return (from the first close after this announcement to the following close) is POSITIVE. "
     "Use only this information. Reply with just an integer.")
def ask(x):
    p = json.dumps({"model": MODEL, "think": False, "stream": False, "prompt": P.format(when=x["when"], c=sym.get(x["inst"], x["inst"]), cat=x["cat"], t=x["text"]), "options": {"temperature": 0, "num_predict": 6}}).encode()
    m = re.search(r"\d+", json.loads(urllib.request.urlopen(urllib.request.Request(OLLAMA_GENERATE_URL, data=p, headers={"Content-Type": "application/json"}), timeout=180).read()).get("response", ""))
    return min(int(m.group()), 100) if m else 50
with ThreadPoolExecutor(3) as ex: probs = list(ex.map(ask, rows))
R = pd.DataFrame(rows).assign(p=probs); R["y"] = (R.ret > 0).astype(int); R["pred"] = (R.p > 50).astype(int); R["conf"] = (R.p - 50).abs()
out = {"model": MODEL, "n": len(R), "base_rate_up": round(float(R.y.mean()), 3), "majority_baseline_acc": round(float(max(R.y.mean(), 1 - R.y.mean())), 3),
       "accuracy": round(float((R.pred == R.y).mean()), 4), "auc": round(float(roc_auc_score(R.y, R.p)), 4) if R.p.nunique() > 1 else None, "pct_called_up": round(float(R.pred.mean()), 3)}
for lo in (10, 20, 30):
    s = R[R.conf >= lo]; out[f"conf>={lo}"] = {"n": len(s), "accuracy": round(float((s.pred == s.y).mean()), 3) if len(s) else None}
out["by_category_accuracy"] = {k: round(float((g.pred == g.y).mean()), 3) for k, g in R.groupby("cat") if len(g) >= 15}
Path(f"reports/llm_direction_{MODEL.replace(':', '-')}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
