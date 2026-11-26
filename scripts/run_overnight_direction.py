"""Does the US close predict Indian direction? Strictly point-in-time: the US session of calendar date d closes (~02:00 IST) BEFORE the next Indian open, so
r_us(d) is known when Indian trading day T (first Indian trading date > d) opens. For the Nifty index and for each stock we measure, walk-forward by year,
the accuracy of predicting the sign of
  gap(T)       = open(T)/close(T-1) - 1     (NOT tradable: the opening auction already contains the US move),
  intraday(T)  = close(T)/open(T) - 1       (tradable from the open),
  close2close  = close(T)/close(T-1) - 1    (mixes the two).
Model: logistic regression on [S&P500 return, Nasdaq return, previous Indian return]; also naive 'follow the S&P sign'. Reports accuracy, AUC, majority baseline."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from nifty_analyser.data.store import load_universe_bars
us = yf.download(["^GSPC", "^IXIC"], start="2020-12-01", progress=False, auto_adjust=True)["Close"].dropna()
us_ret = np.log(us / us.shift(1)).dropna(); us_ret.index = pd.DatetimeIndex(us_ret.index).tz_localize(None).normalize()
b = load_universe_bars(); _ix = pd.read_parquet(Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'NSE_NIFTY50_INDEX.parquet'); _ix = _ix[_ix.quality_flag != 'FAIL']; b = pd.concat([b, _ix[[c for c in b.columns if c in _ix.columns]]], ignore_index=True); b["d"] = pd.DatetimeIndex(b.event_time).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
def frame(g):
    g = g.sort_values("d").drop_duplicates("d").set_index("d")
    f = pd.DataFrame({"gap": np.log(g.open / g.close.shift(1)), "intra": np.log(g.close / g.open), "c2c": np.log(g.close / g.close.shift(1)), "prev": np.log(g.close / g.close.shift(1)).shift(1)})
    idx = f.index; pos = np.searchsorted(us_ret.index.values, idx.values, side="left") - 1          # last US session with date < T  (date <= T-1)
    ok = pos >= 0; f["spx"] = np.where(ok, us_ret["^GSPC"].values[np.clip(pos, 0, None)], np.nan); f["ndx"] = np.where(ok, us_ret["^IXIC"].values[np.clip(pos, 0, None)], np.nan)
    f["lag_days"] = np.where(ok, (idx.values - us_ret.index.values[np.clip(pos, 0, None)]).astype("timedelta64[D]").astype(int), 99)
    return f[f.lag_days <= 4].dropna()
def evaluate(f, target):
    y = (f[target] > 0).astype(int); X = f[["spx", "ndx", "prev"]].to_numpy(); yrs = f.index.year; P = np.full(len(f), np.nan)
    for yr in sorted(set(yrs))[1:]:
        tr, te = yrs < yr, yrs == yr
        if tr.sum() > 200 and te.sum() > 20 and y[tr].nunique() == 2: P[te] = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    m = ~np.isnan(P); yy = y[m]
    return {"n": int(m.sum()), "majority_baseline": round(float(max(yy.mean(), 1 - yy.mean())), 3), "logit_accuracy": round(float(((P[m] > 0.5) == yy).mean()), 4), "logit_auc": round(float(roc_auc_score(yy, P[m])), 4),
            "naive_follow_spx_accuracy": round(float(((f.spx[m] > 0).astype(int) == yy).mean()), 4)}
out = {}
from nifty_analyser.data.instrument_master import NIFTY_INDEX
idx_inst = [NIFTY_INDEX.instrument_id] if NIFTY_INDEX.instrument_id in set(b.instrument_id) else []
nf = frame(b[b.instrument_id.isin(idx_inst)]) if idx_inst else None
out["index_ids"] = idx_inst
if nf is not None and len(nf) > 300: out["nifty_index"] = {t: evaluate(nf, t) for t in ("gap", "intra", "c2c")}
stocks = [i for i in b.instrument_id.unique() if i not in idx_inst]; rows = {t: [] for t in ("gap", "intra", "c2c")}
for i in stocks:
    f = frame(b[b.instrument_id == i])
    if len(f) < 600: continue
    for t in rows: rows[t].append(evaluate(f, t))
out["stocks_mean_over_%d_names" % len(rows["gap"])] = {t: {k: round(float(np.mean([r[k] for r in v])), 4) for k in ("majority_baseline", "logit_accuracy", "logit_auc", "naive_follow_spx_accuracy")} for t, v in rows.items()}
Path("reports/overnight_direction.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
