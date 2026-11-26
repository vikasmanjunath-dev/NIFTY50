"""Predictable target study: will the NEXT-5-day realised volatility be above the stock's own trailing-year median 5-day vol?
Purged/embargoed walk-forward, LightGBM vs naive persistence (current 5d vol vs its own trailing median). Pooled accuracy/AUC + date-neutral AUC +
accuracy by confidence band. Strictly point-in-time: target uses returns t+1..t+5, threshold uses only 5d vols fully realised by t."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.dataset import load_pit_dataset
from nifty_analyser.models.baseline import fit_lightgbm
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward
H = int(sys.argv[1]) if len(sys.argv) > 1 else 5
df, cols = load_pit_dataset(5); OUT, EMB = "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=int(H * 1.6) + 3)   # embargo covers the H-day target window
b = load_universe_bars().sort_values(["instrument_id", "event_time"]); parts = []
for inst, g in b.groupby("instrument_id"):
    r2 = np.log(g.close / g.close.shift(1)).pow(2).to_numpy(); n = len(r2); fwd = np.full(n, np.nan)
    cs = np.concatenate([[0], np.nancumsum(r2)])
    for i in range(n - H): fwd[i] = np.sqrt(cs[i + 1 + H] - cs[i + 1])                 # sqrt(sum r2 over t+1..t+H)
    s = pd.Series(fwd, index=g.event_time.to_numpy()); past = s.shift(H)                # fully realised by t
    cur = pd.Series(np.sqrt(pd.Series(r2).rolling(H).sum().to_numpy()), index=s.index)  # current H-day realised vol (known at t)
    med = past.rolling(252, min_periods=120).median()
    parts.append(pd.DataFrame({"instrument_id": inst, "event_time": s.index, "fwd_rv": s.to_numpy(), "med": med.to_numpy(), "cur_rv": cur.to_numpy()}))
v = pd.concat(parts); df = df.merge(v, on=["instrument_id", "event_time"], how="inner").dropna(subset=["fwd_rv", "med", "cur_rv"]).reset_index(drop=True)
df["yv"] = (df.fwd_rv > df.med).astype(int); df["naive"] = (df.cur_rv > df.med).astype(float)
X, y = df[cols].to_numpy(float), df.yv.to_numpy(); P = np.full(len(df), np.nan)
for f in purged_embargoed_walk_forward(df, 6, OUT, EMB):
    P[f.val_idx] = fit_lightgbm(X[f.train_idx], y[f.train_idx], 42).predict_proba_up(X[f.val_idx])
m = ~np.isnan(P); d = df[m].assign(p=P[m]); pred = (d.p > 0.5).astype(int)
def dateauc(col): return float(np.nanmean([roc_auc_score(g.yv, g[col]) for _, g in d.groupby("event_time") if g.yv.nunique() == 2]))
out = {"horizon_days": H, "n_oos": int(m.sum()), "base_rate": round(float(d.yv.mean()), 3), "lgbm_accuracy": round(float((pred == d.yv).mean()), 4),
       "lgbm_auc": round(float(roc_auc_score(d.yv, d.p)), 4), "lgbm_within_date_auc": round(dateauc("p"), 4),
       "naive_accuracy": round(float((d.naive == d.yv).mean()), 4), "naive_auc": round(float(roc_auc_score(d.yv, d.cur_rv / d.med)), 4)}
d["conf"] = (d.p - 0.5).abs() * 2; d["band"] = pd.qcut(d.conf, 5, labels=False)
out["accuracy_by_confidence_quintile"] = {int(k): round(float(((g.p > 0.5).astype(int) == g.yv).mean()), 3) for k, g in d.groupby("band")}
for lo in (0.5, 0.7, 0.9):
    s = d[d.conf >= lo]; out[f"conf>={lo}"] = {"coverage": round(len(s) / len(d), 3), "accuracy": round(float(((s.p > 0.5).astype(int) == s.yv).mean()), 3) if len(s) else None}
Path(f"reports/vol_forecast_h{H}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
