"""Tradable intraday direction test on Yahoo HOURLY bars (~2 years). Decision at the close of hourly bar k of the day (k=0 -> 10:15 IST, k=3 -> 13:15); target = sign of the
return from that decision price to the day's last bar close. Features known at the decision: overnight gap, return so far today, previous-day return, previous US (S&P) return,
number of bars so far. Walk-forward by calendar quarter, logistic regression pooled over instruments (+ per-instrument for the index). Reports accuracy, AUC, majority baseline,
and accuracy in the most confident decile."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
syms = ["^NSEI", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "TATASTEEL.NS", "NTPC.NS", "ONGC.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "TITAN.NS", "WIPRO.NS", "POWERGRID.NS", "COALINDIA.NS"]
raw = yf.download(syms, period="730d", interval="1h", progress=False, auto_adjust=True, group_by="ticker")
us = yf.download("^GSPC", start="2023-10-01", progress=False, auto_adjust=True)["Close"].dropna().squeeze(); usr = np.log(us / us.shift(1)).dropna(); usr.index = pd.DatetimeIndex(usr.index).tz_localize(None).normalize()
rows = []
for s in syms:
    try: h = raw[s].dropna(subset=["Close"])
    except Exception: continue
    if len(h) < 500: continue
    h.index = pd.DatetimeIndex(h.index).tz_convert("Asia/Kolkata"); h = h.assign(d=h.index.normalize().tz_localize(None))
    days = h.groupby("d"); prev_close = None; prev_ret = np.nan
    for d, g in days:
        if len(g) < 5: continue
        g = g.sort_index(); o, cl = g.Open.iloc[0], g.Close.to_numpy(); last = cl[-1]
        pos = np.searchsorted(usr.index.values, np.datetime64(d), side="left") - 1; spx = usr.iloc[pos] if pos >= 0 and (d - usr.index[pos]).days <= 4 else np.nan
        for k in (0, 3):
            if k >= len(cl) - 1: continue
            rows.append({"inst": s, "d": d, "k": k, "gap": np.log(o / prev_close) if prev_close else np.nan, "sofar": np.log(cl[k] / o), "prev": prev_ret, "spx": spx, "y_ret": np.log(last / cl[k])})
        prev_ret = np.log(last / prev_close) if prev_close else np.nan; prev_close = last
D = pd.DataFrame(rows).dropna(); D["q"] = D.d.dt.to_period("Q").astype(str); out = {"n_days_instrument": int(D[D.k == 0].shape[0]), "instruments": int(D.inst.nunique()), "span": [str(D.d.min().date()), str(D.d.max().date())]}
def run(df):
    y = (df.y_ret > 0).astype(int).to_numpy(); X = df[["gap", "sofar", "prev", "spx"]].to_numpy(); qs = sorted(df.q.unique()); P = np.full(len(df), np.nan)
    for q in qs[2:]:
        tr, te = (df.q < q).to_numpy(), (df.q == q).to_numpy()
        if tr.sum() > 500 and te.sum() > 20 and len(set(y[tr])) == 2: P[te] = LogisticRegression(C=1.0, max_iter=500).fit(X[tr] * 100, y[tr]).predict_proba(X[te] * 100)[:, 1]
    m = ~np.isnan(P); yy, pp = y[m], P[m]; conf = np.abs(pp - 0.5); top = conf >= np.quantile(conf, 0.9)
    return {"n": int(m.sum()), "majority_baseline": round(float(max(yy.mean(), 1 - yy.mean())), 3), "accuracy": round(float(((pp > 0.5) == yy).mean()), 4), "auc": round(float(roc_auc_score(yy, pp)), 4),
            "accuracy_top_decile_confidence": round(float(((pp[top] > 0.5) == yy[top]).mean()), 3)}
for k in (0, 3):
    out[f"k{k}_pooled_all_instruments"] = run(D[D.k == k]); out[f"k{k}_index_only"] = run(D[(D.k == k) & (D.inst == "^NSEI")])
    # simple intraday-momentum rule: continue the direction so far
    s = D[D.k == k]; out[f"k{k}_naive_continue_sofar_accuracy"] = round(float(((s.sofar > 0) == (s.y_ret > 0)).mean()), 4)
Path("reports/intraday_direction.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
