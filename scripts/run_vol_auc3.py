"""Round 4 (docs/AUC_PLAN.md): choose on 2008-2020 only, confirm once per horizon on the 2021+ rows against the live model.

Adoption rule: the 95% date-block interval of the 2021+ AUC gain over the live model is above zero."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vol_forecast_v2 import block_bootstrap, load

from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.features.volatility_extra import extra_features, stock_features
from nifty_analyser.models.baseline import fit_catboost
from nifty_analyser.models.volatility import _TREE, _linear, fit_predict_classifiers
from nifty_analyser.ops.vol_forecast import EXTRA_FEATURE_HORIZONS, load_history_training
from nifty_analyser.validation.walk_forward import assert_folds_are_leakage_free, purged_embargoed_walk_forward

KEY = ["instrument_id", "event_time"]
MODELS = ("blend", "big_trees", "cat3", "reg_target", "combo")


def rank(p):
    return pd.Series(p).rank(pct=True).to_numpy()


def score(X, y, t, V, kind):
    """y: above-usual label; t: log(next volatility / usual), the continuous version of the same question."""
    if kind == "reg_target":
        ridge = _linear(Ridge(alpha=10.0)).fit(X, t).predict(V)
        tree = LGBMRegressor(**_TREE).fit(X, t).predict(V)
        return (ridge + tree) / 2
    base = fit_predict_classifiers(X, y, V)
    if kind == "blend":
        return base["blend"]
    if kind == "big_trees":
        return (base["har_logit"] + LGBMClassifier(**{**_TREE, "num_leaves": 31, "n_estimators": 500}).fit(X, y).predict_proba(V)[:, 1]) / 2
    if kind == "cat3":
        return (base["har_logit"] + base["lgbm"] + fit_catboost(X, y).predict_proba_up(V)) / 3
    return (rank(base["blend"]) + rank(score(X, y, t, V, "reg_target"))) / 2


def targets(df):
    return (df.fwd_rv > df.med_rv).astype(int).to_numpy(), np.log(df.fwd_rv / df.med_rv).to_numpy()


def walk(df, cols, kind, n_folds, h, prior=None):
    """Out-of-sample scores on df; `prior` rows (all realised before df starts) are always in training."""
    folds = purged_embargoed_walk_forward(df, n_folds, "outcome_time", pd.Timedelta(days=int(h * 1.6) + 3))
    assert_folds_are_leakage_free(df, folds, "outcome_time")
    X, (y, t) = df[cols].to_numpy(float), targets(df)
    if prior is not None:
        Xo, (yo, to) = prior[cols].to_numpy(float), targets(prior)
    p = np.full(len(df), np.nan)
    for f in folds:
        tr, va = f.train_idx, f.val_idx
        if prior is None:
            p[va] = score(X[tr], y[tr], t[tr], X[va], kind)
        else:
            p[va] = score(np.vstack([Xo, X[tr]]), np.r_[yo, y[tr]], np.r_[to, t[tr]], X[va], kind)
    return p


def enrich(df, bars, h):
    extra, extra_cols = extra_features(bars)
    df = df.drop(columns=[c for c in extra_cols if c in df]).merge(extra, on=KEY, how="left")
    df, stock_cols = stock_features(df, h)
    return df, extra_cols, stock_cols


def main():
    hist_bars = pd.read_parquet(Path("data/raw/history/bars.parquet"))
    for c in ("event_time", "available_time"):
        hist_bars[c] = pd.to_datetime(hist_bars[c], utc=True).dt.tz_convert("Asia/Kolkata").astype("datetime64[ns, Asia/Kolkata]")
    hist_bars = hist_bars[hist_bars.quality_flag != "FAIL"]
    now_bars = load_universe_bars()
    out = {}
    for h in (5, 20):
        df, cols = load(h)
        df = df.dropna(subset=["fwd_rv", "med_rv", "cur_rv", "outcome_time", "har_66"])
        df = df[(df.fwd_rv > 0) & (df.cur_rv > 0)]
        df, extra_cols, stock_cols = enrich(df, now_bars, h)
        old = load_history_training(h, extra=False)
        old, _, _ = enrich(old, hist_bars, h)
        old = old.reindex(columns=df.columns)
        assert old.outcome_time.max() < df.event_time.min()
        feature_sets = {"base": cols, "extra": cols + extra_cols, "extra_stock": cols + extra_cols + stock_cols}
        dev = {}
        y_old = targets(old)[0]
        preds = {(f, k): walk(old, fc, k, 8, h) for f, fc in feature_sets.items() for k in MODELS}
        for key, p in preds.items():
            m = ~np.isnan(p)
            dev[key] = float(roc_auc_score(y_old[m], p[m]))
        win = max(dev, key=dev.get)
        live = ("extra" if h in EXTRA_FEATURE_HORIZONS else "base", "blend")
        a = walk(df, feature_sets[live[0]], live[1], 6, h, prior=old)
        b = walk(df, feature_sets[win[0]], win[1], 6, h, prior=old)
        m = ~np.isnan(a)
        y, dates = targets(df)[0][m], df.event_time.to_numpy()[m]
        a, b = a[m], b[m]
        res = {"dev_auc_2008_2020": {"/".join(k): round(v, 4) for k, v in dev.items()}, "winner": "/".join(win), "live": "/".join(live),
               "auc_live_2021": round(float(roc_auc_score(y, a)), 4), "auc_winner_2021": round(float(roc_auc_score(y, b)), 4),
               "gain_ci95": block_bootstrap(dates, lambda i: roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]), max(2 * h, 20), reps=400)}
        res["adopt"] = res["gain_ci95"][0] > 0
        out[str(h)] = res
        print(h, json.dumps(res, separators=(",", ":")), flush=True)
        Path("reports/vol_auc_search3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
