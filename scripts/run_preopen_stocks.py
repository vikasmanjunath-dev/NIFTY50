"""Stock-level pre-open model v2: adds per-stock trailing betas (60 prior days, current day excluded) of the stock's own gap to S&P, Nasdaq, dollar index, Brent and USD/INR,
beta x same-night driver interaction terms, sector code, weekday, and the stock's trailing 20-day mean gap. Same walk-forward by year as run_preopen_direction.py.
Targets: gap, close-to-close, open-to-close. Pooled over 59 stocks."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
__file__ = str(Path(__file__).resolve())
head = open(__file__.replace("run_preopen_stocks.py", "run_preopen_direction.py")).read().split("res = {}")[0]
exec(head)
from nifty_analyser.data.instrument_master import get_instrument
DRV = ["spx", "ndx", "dxy", "brent", "usdinr"]
def build2(g, inst):
    f = build(g)
    if len(f) < 400: return None
    for k in DRV:      # trailing beta of the stock's gap to the driver, using ONLY earlier days (shift(1) excludes today)
        cov = f.gap.rolling(60, min_periods=40).cov(f[k]).shift(1); var = f[k].rolling(60, min_periods=40).var().shift(1)
        f[f"b_{k}"] = cov / var; f[f"x_{k}"] = f[f"b_{k}"] * f[k]
    f["drift20"] = f.gap.rolling(20, min_periods=10).mean().shift(1); f["dow"] = f.d.dt.dayofweek; f["sector"] = hash(get_instrument(inst).sector) % 997
    f["inst"] = inst
    return f
sec = {s: i for i, s in enumerate(sorted({get_instrument(i).sector for i in b.instrument_id.unique() if i != NIFTY_INDEX.instrument_id}))}
parts = [build2(b[b.instrument_id == i], i) for i in b.instrument_id.unique() if i != NIFTY_INDEX.instrument_id]
pool = pd.concat([p for p in parts if p is not None]).sort_values("d").reset_index(drop=True); pool["sector"] = pool.inst.map(lambda i: sec[get_instrument(i).sector])
FE2 = FE + [f"b_{k}" for k in DRV] + [f"x_{k}" for k in DRV] + ["drift20", "dow", "sector"]; pool = pool.dropna(subset=FE2)
def walk2(df, target, kind, feats):
    y = (df[target] > 0).astype(int).to_numpy(); X = df[feats].to_numpy(); yr = df.d.dt.year.to_numpy(); P = np.full(len(df), np.nan)
    for Y in sorted(set(yr))[1:]:
        tr, te = yr < Y, yr == Y
        if tr.sum() < 2000 or te.sum() < 200 or len(set(y[tr])) < 2: continue
        if kind == "logit":
            sc = StandardScaler().fit(X[tr]); P[te] = LogisticRegression(C=0.5, max_iter=800).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
        else:
            P[te] = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    m = ~np.isnan(P); yy, pp = y[m], P[m]; conf = np.abs(pp - 0.5); o = {"n": int(m.sum()), "base": round(float(max(yy.mean(), 1 - yy.mean())), 3), "auc": round(float(roc_auc_score(yy, pp)), 4), "acc": round(float(((pp > .5) == yy).mean()), 4)}
    for q in (0.5, 0.25): top = conf >= np.quantile(conf, 1 - q); o[f"acc_top{int(q*100)}"] = round(float(((pp[top] > .5) == yy[top]).mean()), 3)
    return o
res = {"n_stocks": int(pool.inst.nunique()), "n_rows": len(pool)}
for t in ("gap", "c2c", "intra"):
    res[t] = {"v1_features_lgbm": walk2(pool, t, "lgbm", FE), "v2_features_logit": walk2(pool, t, "logit", FE2), "v2_features_lgbm": walk2(pool, t, "lgbm", FE2)}
    print(t, json.dumps(res[t]), flush=True)
Path("reports/preopen_stocks_v2.json").write_text(json.dumps(res, indent=1))
