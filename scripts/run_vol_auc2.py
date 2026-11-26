"""Second AUC search, now with 2008-2020 training history. Run build_chronos_history.py first (separate process).

Same protocol as run_vol_auc.py: candidates are compared on the first 60% of out-of-sample dates, only the winner is scored on the
last 40% against the live model, and it is adopted only if that gain's 95% date-block interval is above zero. Direction is reported
for reference with the same data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_vol_auc
from run_vol_forecast_v2 import block_bootstrap, load

from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.features.volatility_extra import extra_features
from nifty_analyser.models.baseline import fit_catboost
from nifty_analyser.models.volatility import _TREE, _linear, fit_predict_classifiers
from nifty_analyser.ops.vol_forecast import load_history_training
from nifty_analyser.validation.walk_forward import assert_folds_are_leakage_free, purged_embargoed_walk_forward

KEY = ["instrument_id", "event_time"]
HIST_CHRONOS = DATA_PROCESSED_DIR / "chronos_vol_features_history.parquet"


def history_bars():
    b = pd.read_parquet(DATA_RAW_DIR / "history" / "bars.parquet")
    for c in ("event_time", "available_time"):
        b[c] = pd.to_datetime(b[c], utc=True).dt.tz_convert("Asia/Kolkata").astype("datetime64[ns, Asia/Kolkata]")
    return b[b.quality_flag != "FAIL"]


def fit(Xt, yt, Xv, kind):
    if kind == "big_trees":
        base = fit_predict_classifiers(Xt, yt, Xv)["har_logit"]
        tree = LGBMClassifier(**{**_TREE, "num_leaves": 31, "n_estimators": 500}).fit(Xt, yt).predict_proba(Xv)[:, 1]
        return (base + tree) / 2
    base = fit_predict_classifiers(Xt, yt, Xv)
    if kind == "blend":
        return base["blend"]
    cat = fit_catboost(Xt, yt).predict_proba_up(Xv)
    mlp = _linear(MLPClassifier((24,), alpha=3.0, batch_size=1024, max_iter=25, random_state=42)).fit(Xt, yt).predict_proba(Xv)[:, 1]
    return (base["har_logit"] + base["lgbm"] + cat + mlp) / 4


def main():
    live_bars, old_bars = load_universe_bars(), history_bars()
    extra_now, extra_cols = extra_features(live_bars)
    extra_old = extra_features(old_bars)[0]
    chron_now, chron_old = pd.read_parquet(run_vol_auc.CACHE), pd.read_parquet(HIST_CHRONOS)
    out = {}
    for h in (5, 20):
        df, cols = load(h)
        df = df.dropna(subset=["fwd_rv", "med_rv", "cur_rv", "outcome_time", "har_66"])
        df = df[(df.fwd_rv > 0) & (df.cur_rv > 0)]
        df = df.merge(extra_now, on=KEY, how="left").merge(chron_now, on=KEY, how="left").sort_values(KEY[::-1]).reset_index(drop=True)
        old = load_history_training(h, extra=False).merge(extra_old, on=KEY, how="left").merge(chron_old, on=KEY, how="left")
        assert old.outcome_time.max() < df.event_time.min()
        old = old.reindex(columns=df.columns)
        y, yo = (df.fwd_rv > df.med_rv).astype(int).to_numpy(), (old.fwd_rv > old.med_rv).astype(int).to_numpy()
        folds = {}
        for n in (6, 12):
            folds[n] = purged_embargoed_walk_forward(df, n, "outcome_time", pd.Timedelta(days=int(h * 1.6) + 3))
            assert_folds_are_leakage_free(df, folds[n], "outcome_time")
        c = [f"chronos_{h}"]
        candidates = {"live": (cols, 6, "blend"), "refit_often": (cols, 12, "blend"), "big_trees": (cols, 12, "big_trees"),
                      "more_features": (cols + extra_cols, 12, "blend"), "plus_chronos": (cols + c, 12, "blend"),
                      "chronos_ensemble4": (cols + c, 12, "ensemble")}
        P = {}
        for name, (fc, n, kind) in candidates.items():
            X, Xo = df[fc].to_numpy(float), old[fc].to_numpy(float)
            p = np.full(len(df), np.nan)
            for f in folds[n]:
                p[f.val_idx] = fit(np.vstack([Xo, X[f.train_idx]]), np.r_[yo, y[f.train_idx]], X[f.val_idx], kind)
            P[name] = p
        m = np.all([~np.isnan(p) for p in P.values()], axis=0)
        dates = df.event_time.to_numpy()
        uniq = np.unique(dates[m])
        k = int(len(uniq) * 0.6)
        dev, hold = m & (dates < uniq[k - h]), m & (dates >= uniq[k])
        dev_auc = {n: round(float(roc_auc_score(y[dev], p[dev])), 4) for n, p in P.items()}
        win = max(dev_auc, key=dev_auc.get)
        yy, a, b = y[hold], P["live"][hold], P[win][hold]
        res = {"dev_auc": dev_auc, "winner": win, "holdout_live": round(float(roc_auc_score(yy, a)), 4),
               "holdout_winner": round(float(roc_auc_score(yy, b)), 4),
               "gain_ci95": block_bootstrap(dates[hold], lambda i: roc_auc_score(yy[i], b[i]) - roc_auc_score(yy[i], a[i]), max(2 * h, 20), reps=400),
               "full_oos_live": round(float(roc_auc_score(y[m], P["live"][m])), 4), "full_oos_winner": round(float(roc_auc_score(y[m], P[win][m])), 4)}
        res["adopt"] = win != "live" and res["gain_ci95"][0] > 0
        # direction of the next h-day return with the same rows and history, for reference
        for frame, store in ((df, "now"), (old, "old")):
            src = live_bars if store == "now" else old_bars
            fwd = pd.concat([pd.DataFrame({"instrument_id": i, "event_time": g.event_time, "fret": g.close.shift(-h) / g.close - 1})
                             for i, g in src.sort_values(KEY).groupby("instrument_id")])
            frame["fret"] = frame[KEY].merge(fwd, on=KEY, how="left").fret.to_numpy()
        X, Xo = df[cols + extra_cols].to_numpy(float), old[cols + extra_cols].to_numpy(float)
        res["direction_auc"] = {}
        for target in ("up", "beats_median_stock"):
            labels = [((f.fret > 0) if target == "up" else (f.fret > f.groupby("event_time").fret.transform("median"))).astype(int).to_numpy()
                      for f in (df, old)]
            p = np.full(len(df), np.nan)
            for f in folds[6]:
                p[f.val_idx] = fit(np.vstack([Xo, X[f.train_idx]]), np.r_[labels[1], labels[0][f.train_idx]], X[f.val_idx], "blend")
            ok = ~np.isnan(p) & df.fret.notna().to_numpy()
            res["direction_auc"][target] = round(float(roc_auc_score(labels[0][ok], p[ok])), 4)
        out[str(h)] = res
        print(h, json.dumps(res, separators=(",", ":")), flush=True)
    Path("reports/vol_auc_search2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
