"""Volatility forecasting v2 on the point-in-time universe, purged/embargoed walk-forward.

Pre-registered primary model, fixed before the first run: `blend` (HAR linear + shallow LightGBM).
Benchmarks: persistence of current volatility. Differences carry a date-block bootstrap interval."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.data.index_membership import membership_mask
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.features.volatility import build_volatility_frame
from nifty_analyser.models.volatility import fit_predict_classifiers, fit_predict_regressors
from nifty_analyser.ops.vol_forecast import load_inputs
from nifty_analyser.validation.walk_forward import assert_folds_are_leakage_free, purged_embargoed_walk_forward


def load(horizon):
    bars, vix, announcements = load_inputs()
    df, cols = build_volatility_frame(bars, horizon, vix, announcements)
    return df[membership_mask(df).to_numpy()].reset_index(drop=True), cols


def block_bootstrap(dates, stat, block, reps=500, seed=0):
    uniq, inv = np.unique(dates, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        starts = rng.integers(0, len(uniq) - block + 1, size=len(uniq) // block + 1)
        vals.append(stat(np.concatenate([rows[i] for s in starts for i in range(s, s + block)])))
    return [round(float(q), 4) for q in np.percentile(vals, [2.5, 97.5])]


def evaluate(horizon):
    df, cols = load(horizon)
    df = df.dropna(subset=["fwd_rv", "med_rv", "cur_rv", "outcome_time", "har_66"])
    df = df[(df.fwd_rv > 0) & (df.cur_rv > 0)].reset_index(drop=True)
    y = (df.fwd_rv > df.med_rv).astype(int).to_numpy()
    lv = np.log(df.fwd_rv).to_numpy()
    X = df[cols].to_numpy(float)
    folds = purged_embargoed_walk_forward(df, 6, "outcome_time", pd.Timedelta(days=int(horizon * 1.6) + 3))
    assert_folds_are_leakage_free(df, folds, "outcome_time")
    P, R = {}, {}
    mean_lv = np.full(len(df), np.nan)
    for f in folds:
        tr, va = f.train_idx, f.val_idx
        for store, preds in ((P, fit_predict_classifiers(X[tr], y[tr], X[va])), (R, fit_predict_regressors(X[tr], lv[tr], X[va]))):
            for k, v in preds.items():
                store.setdefault(k, np.full(len(df), np.nan))[va] = v
        mean_lv[va] = lv[tr].mean()
    m = ~np.isnan(P["blend"])
    d = df[m].reset_index(drop=True)
    y, lv, mean_lv = y[m], lv[m], mean_lv[m]
    naive = np.log(d.cur_rv / d.med_rv).to_numpy()
    dates = d.event_time.to_numpy()

    def within_date(score):
        t = pd.DataFrame({"d": dates, "y": y, "s": score})
        return float(np.mean([roc_auc_score(g.y, g.s) for _, g in t.groupby("d") if g.y.nunique() == 2]))

    out = {"horizon_days": horizon, "n_oos": int(m.sum()), "n_features": len(cols), "base_rate": round(float(y.mean()), 3),
           "classification": {"naive_persistence": {"auc": round(roc_auc_score(y, naive), 4), "within_date_auc": round(within_date(naive), 4),
                                                    "accuracy": round(float(((naive > 0) == y).mean()), 4)}},
           "regression_log_rv": {}}
    for k, p in P.items():
        p = p[m]
        conf = np.abs(p - 0.5) * 2
        top = conf >= np.quantile(conf, 0.8)
        out["classification"][k] = {"auc": round(roc_auc_score(y, p), 4), "within_date_auc": round(within_date(p), 4),
                                    "accuracy": round(float(((p > 0.5) == y).mean()), 4),
                                    "accuracy_top20pct_confidence": round(float(((p[top] > 0.5) == y[top]).mean()), 4)}
    d[["instrument_id", "event_time"]].assign(y=y, lv=lv, p=P["blend"][m], r=R["blend"][m]).to_parquet(
        DATA_PROCESSED_DIR / f"vol_oos_h{horizon}.parquet", index=False)
    best = P["blend"][m]
    out["auc_blend_minus_naive_ci95"] = block_bootstrap(
        dates, lambda i: roc_auc_score(y[i], best[i]) - roc_auc_score(y[i], naive[i]), max(2 * horizon, 20))

    def reg(pred):
        ratio = np.exp(2 * (lv - pred))  # realised variance / forecast variance
        return {"oos_r2": round(float(1 - ((lv - pred) ** 2).sum() / ((lv - mean_lv) ** 2).sum()), 4),
                "qlike": round(float(np.mean(ratio - np.log(ratio) - 1)), 4)}

    out["regression_log_rv"] = {"naive_persistence": reg(np.log(d.cur_rv).to_numpy()), **{k: reg(v[m]) for k, v in R.items()}}
    return out


if __name__ == "__main__":
    for h in (int(a) for a in (sys.argv[1:] or ["5", "20"])):
        out = evaluate(h)
        Path(f"reports/vol_forecast_v2_h{h}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, separators=(",", ":")))
