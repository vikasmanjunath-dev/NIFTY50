"""Pre-open direction model: everything here is known at 09:00 IST of Indian trading day T (as-of strictly BEFORE T's calendar date):
US indices (S&P, Nasdaq), US VIX change, dollar index, Brent, gold, US 10y yield change, USD/INR, plus Indian previous-day return/gap. Targets for day T:
gap = sign(open_T/close_{T-1}), c2c = sign(close_T/close_{T-1}), intra = sign(close_T/open_T) (the only part tradable after the open).
Walk-forward by calendar year (train on all earlier years). Models: logistic regression and LightGBM. Index + pooled stocks. Reports AUC, accuracy, majority baseline,
and accuracy/coverage when only the most confident predictions are kept."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from nifty_analyser.data.instrument_master import NIFTY_INDEX
from nifty_analyser.data.store import load_universe_bars
T = {"spx": "^GSPC", "ndx": "^IXIC", "vix": "^VIX", "dxy": "DX-Y.NYB", "brent": "BZ=F", "gold": "GC=F", "tnx": "^TNX", "usdinr": "INR=X"}
raw = yf.download(list(T.values()), start="2020-11-01", progress=False, auto_adjust=True)["Close"]
G = pd.DataFrame(index=pd.DatetimeIndex(raw.index).tz_localize(None).normalize())
for k, t in T.items():
    s = raw[t].copy(); s.index = G.index; s = s[~s.index.duplicated()].dropna()
    G[k] = (s.diff() if k in ("tnx",) else np.log(s / s.shift(1))).reindex(G.index)
G = G.dropna(how="all").reset_index().rename(columns={"index": "gd", "Date": "gd"}); G["gd"] = pd.to_datetime(G.iloc[:, 0]).dt.normalize()
b = load_universe_bars(); ix = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "raw" / "NSE_NIFTY50_INDEX.parquet"); ix = ix[ix.quality_flag != "FAIL"]
b = pd.concat([b, ix[[c for c in b.columns if c in ix.columns]]], ignore_index=True); b["d"] = pd.DatetimeIndex(b.event_time).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
FE = ["spx", "ndx", "vix", "dxy", "brent", "gold", "tnx", "usdinr", "prev", "prev_gap"]
def build(g):
    g = g.sort_values("d").drop_duplicates("d"); f = pd.DataFrame({"d": g.d.values, "gap": np.log(g.open.values / g.close.shift(1).values), "intra": np.log(g.close.values / g.open.values), "c2c": np.log(g.close.values / g.close.shift(1).values)})
    f["prev"] = f.c2c.shift(1); f["prev_gap"] = f.gap.shift(1)
    f = pd.merge_asof(f.sort_values("d"), G.sort_values("gd"), left_on="d", right_on="gd", direction="backward", allow_exact_matches=False)     # last session strictly before T
    f = f[(f.d - f.gd).dt.days <= 5]
    return f.dropna(subset=FE + ["gap", "intra", "c2c"])
def walk(df, target, kind):
    y = (df[target] > 0).astype(int).to_numpy(); X = df[FE].to_numpy(); yr = df.d.dt.year.to_numpy(); P = np.full(len(df), np.nan)
    for Y in sorted(set(yr))[1:]:
        tr, te = yr < Y, yr == Y
        if tr.sum() < 250 or te.sum() < 20 or len(set(y[tr])) < 2: continue
        if kind == "logit":
            sc = StandardScaler().fit(X[tr]); P[te] = LogisticRegression(C=0.5, max_iter=500).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
        else:
            P[te] = LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    m = ~np.isnan(P); yy, pp = y[m], P[m]; conf = np.abs(pp - 0.5); out = {"n": int(m.sum()), "base": round(float(max(yy.mean(), 1 - yy.mean())), 3), "auc": round(float(roc_auc_score(yy, pp)), 4), "acc": round(float(((pp > .5) == yy).mean()), 4)}
    for q in (0.5, 0.25):
        top = conf >= np.quantile(conf, 1 - q); out[f"acc_top{int(q*100)}pct_conf"] = round(float(((pp[top] > .5) == yy[top]).mean()), 3)
    return out
res = {}
nf = build(b[b.instrument_id == NIFTY_INDEX.instrument_id])
res["nifty_index"] = {t: {k: walk(nf, t, k) for k in ("logit", "lgbm")} for t in ("gap", "c2c", "intra")}
pool = pd.concat([build(b[b.instrument_id == i]).assign(inst=i) for i in b.instrument_id.unique() if i != NIFTY_INDEX.instrument_id]).sort_values("d").reset_index(drop=True)
res["stocks_pooled"] = {"n_stocks": int(pool.inst.nunique()), **{t: {k: walk(pool, t, k) for k in ("logit", "lgbm")} for t in ("gap", "c2c", "intra")}}
Path("reports/preopen_direction.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
