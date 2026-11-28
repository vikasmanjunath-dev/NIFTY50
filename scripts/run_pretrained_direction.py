"""Pretrained models on the direction problem, honest zero-shot tests:
(A) ProsusAI/finbert sentiment of 400 real announcements -> next-day direction (same sample/entry/exit as run_llm_direction.py);
(B) amazon/chronos-bolt-small time-series foundation model, zero-shot on the last 256 closes of 3,000 random (stock, day) points -> next-1-day and 5-day direction.
Metrics: accuracy of sign, AUC of the score, rank IC; vs the majority baseline."""
"""Can a local LLM call next-day direction from a company announcement? Strictly point-in-time: the model sees only the announcement text, company and
date; entry = first close at/after publication (after 15:30 -> next day's close), exit = the following close (1 trading day). N announcements in
market-relevant categories, 2023-06..2026-10. Reports accuracy, AUC, accuracy by stated confidence, vs the 'always-majority' baseline."""
import json, sys

from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.instrument_master import NIFTY50
from nifty_analyser.data.store import load_universe_bars

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

import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tk = AutoTokenizer.from_pretrained("ProsusAI/finbert"); fm = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", use_safetensors=True).eval(); lab = {v.lower(): k for k, v in fm.config.id2label.items()}
R = pd.DataFrame(rows); sc = []
for i in range(0, len(R), 32):
    enc = tk(list(R.text.iloc[i:i + 32]), truncation=True, max_length=256, padding=True, return_tensors="pt")
    with torch.no_grad(): p = torch.softmax(fm(**enc).logits, -1).numpy()
    sc += (p[:, lab["positive"]] - p[:, lab["negative"]]).tolist()
R["s"] = sc; R["y"] = (R.ret > 0).astype(int); out = {"A_finbert": {"n": len(R), "majority_baseline": round(float(max(R.y.mean(), 1 - R.y.mean())), 3), "accuracy_sign": round(float(((R.s > 0).astype(int) == R.y).mean()), 4),
       "auc": round(float(roc_auc_score(R.y, R.s)), 4), "rank_ic": round(float(spearmanr(R.s, R.ret)[0]), 4), "pct_positive": round(float((R.s > 0).mean()), 3)}}
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="mps" if torch.backends.mps.is_available() else "cpu", torch_dtype=torch.float32)
bars = load_universe_bars().pivot_table(index="event_time", columns="instrument_id", values="close").sort_index(); rng = np.random.default_rng(11); samples = []
cols = [c for c in bars.columns if bars[c].notna().sum() > 600]
while len(samples) < 3000:
    c = cols[rng.integers(len(cols))]; s_ = bars[c].dropna(); i = int(rng.integers(300, len(s_) - 6))
    if s_.index[i] >= pd.Timestamp("2023-01-01", tz=s_.index.tz): samples.append((c, s_.index[i], s_.iloc[i - 255:i + 1].to_numpy(np.float32), float(np.log(s_.iloc[i + 1] / s_.iloc[i])), float(np.log(s_.iloc[i + 5] / s_.iloc[i]))))
ctxs = [torch.tensor(x[2]) for x in samples]; f1, f5 = [], []
for i in range(0, len(ctxs), 128):
    q, mean = pipe.predict_quantiles(ctxs[i:i + 128], prediction_length=5, quantile_levels=[0.5])
    med = q[:, :, 0].numpy(); last = np.array([c[-1].item() for c in ctxs[i:i + 128]])
    f1 += (np.log(med[:, 0] / last)).tolist(); f5 += (np.log(med[:, 4] / last)).tolist()
S = pd.DataFrame({"f1": f1, "f5": f5, "r1": [x[3] for x in samples], "r5": [x[4] for x in samples], "t": [x[1] for x in samples]})
for h, f, r in ((1, "f1", "r1"), (5, "f5", "r5")):
    y = (S[r] > 0).astype(int); out[f"B_chronos_h{h}"] = {"n": len(S), "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 3), "accuracy_sign": round(float(((S[f] > 0).astype(int) == y).mean()), 4), "auc": round(float(roc_auc_score(y, S[f])), 4),
                                                           "rank_ic": round(float(spearmanr(S[f], S[r])[0]), 4), "pct_up_calls": round(float((S[f] > 0).mean()), 3)}
Path("reports/pretrained_direction.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
