"""Search for higher volatility AUC without fooling ourselves.

Candidates are compared on the first 60% of out-of-sample dates (development); only the development winner of each horizon is
then scored on the last 40% (holdout) against the current model. Adopt if the holdout gain's 95% date-block interval is above zero."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vol_forecast_v2 import block_bootstrap, load

from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.features.volatility import EPS, daily_variance
from nifty_analyser.features.volatility_extra import extra_features
from nifty_analyser.models.baseline import fit_catboost
from nifty_analyser.models.volatility import _linear, fit_predict_classifiers
from nifty_analyser.validation.walk_forward import assert_folds_are_leakage_free, purged_embargoed_walk_forward

CACHE = DATA_PROCESSED_DIR / "chronos_vol_features.parquet"


def chronos_features(bars):
    """Zero-shot forecast made from the 256 days up to and including each date, relative to the trailing-year median."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import torch  # imported only here: PyTorch's OpenMP deadlocks LightGBM in the same macOS process
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="mps" if torch.backends.mps.is_available() else "cpu",
                                               torch_dtype=torch.float32)
    keys, contexts, medians = [], [], []
    for iid, g in bars.groupby("instrument_id"):
        g = g.sort_values("event_time")
        lv = np.log(daily_variance(g) + EPS)
        med = lv.rolling(252, min_periods=120).median().to_numpy()
        values, times = lv.to_numpy(), g.event_time.to_numpy()
        for i in range(130, len(g)):
            w = values[max(0, i - 255): i + 1]
            keys.append((iid, times[i]))
            contexts.append(torch.tensor(w[np.isfinite(w)], dtype=torch.float32))
            medians.append(med[i])
    q = []
    for s in range(0, len(contexts), 256):
        q.append(pipe.predict_quantiles(contexts[s: s + 256], prediction_length=20, quantile_levels=[0.5])[0][:, :, 0].cpu().numpy())
    q = np.concatenate(q)
    out = pd.DataFrame(keys, columns=["instrument_id", "event_time"])
    out["event_time"] = pd.to_datetime(out.event_time, utc=True).dt.tz_convert("Asia/Kolkata")
    for h in (5, 20):
        out[f"chronos_{h}"] = np.log(np.exp(q[:, :h]).mean(1)) - np.array(medians)
    out.to_parquet(CACHE, index=False)
    return out


def predict(X, y, folds, kind):
    p = np.full(len(y), np.nan)
    for f in folds:
        tr, va = f.train_idx, f.val_idx
        base = fit_predict_classifiers(X[tr], y[tr], X[va])
        if kind == "blend":
            p[va] = base["blend"]
        else:
            cat = fit_catboost(X[tr], y[tr]).predict_proba_up(X[va])
            mlp = _linear(MLPClassifier((24,), alpha=3.0, batch_size=512, max_iter=40, random_state=42)).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
            p[va] = (base["har_logit"] + base["lgbm"] + cat + mlp) / 4
    return p


def main():
    if not CACHE.exists():
        chronos_features(load_universe_bars())
        sys.exit("Chronos features cached; run again to search (kept in a separate process on purpose)")
    bars = load_universe_bars()
    extra, extra_cols = extra_features(bars, pd.read_parquet(DATA_RAW_DIR / "NSE_NIFTY50_INDEX.parquet"))
    chron = chronos_features(bars)
    out = {}
    for h in (5, 20):
        df, cols = load(h)
        df = df.dropna(subset=["fwd_rv", "med_rv", "cur_rv", "outcome_time", "har_66"])
        df = df[(df.fwd_rv > 0) & (df.cur_rv > 0)]
        df = df.merge(extra, on=["instrument_id", "event_time"], how="left").merge(chron, on=["instrument_id", "event_time"], how="left")
        df = df.sort_values(["event_time", "instrument_id"]).reset_index(drop=True)
        y = (df.fwd_rv > df.med_rv).astype(int).to_numpy()
        folds = {}
        for n in (6, 18):
            folds[n] = purged_embargoed_walk_forward(df, n, "outcome_time", pd.Timedelta(days=int(h * 1.6) + 3))
            assert_folds_are_leakage_free(df, folds[n], "outcome_time")
        c = [f"chronos_{h}"]
        candidates = {"current": (cols, 6, "blend"), "refit_often": (cols, 18, "blend"), "more_features": (cols + extra_cols, 18, "blend"),
                      "plus_chronos": (cols + c, 18, "blend"), "all_features": (cols + extra_cols + c, 18, "blend"),
                      "chronos_ensemble4": (cols + c, 18, "ensemble"), "all_ensemble4": (cols + extra_cols + c, 18, "ensemble")}
        P = {k: predict(df[fc].to_numpy(float), y, folds[n], kind) for k, (fc, n, kind) in candidates.items()}
        m = np.all([~np.isnan(p) for p in P.values()], axis=0)
        dates = df.event_time.to_numpy()
        uniq = np.unique(dates[m])
        cut = uniq[int(len(uniq) * 0.6)]
        dev, hold = m & (dates < uniq[int(len(uniq) * 0.6) - h]), m & (dates >= cut)
        dev_auc = {k: round(float(roc_auc_score(y[dev], p[dev])), 4) for k, p in P.items()}
        win = max(dev_auc, key=dev_auc.get)
        yy, a, b = y[hold], P["current"][hold], P[win][hold]
        res = {"dev_auc": dev_auc, "winner": win, "n_holdout": int(hold.sum()), "holdout_from": str(pd.Timestamp(cut).date()),
               "holdout_current": round(float(roc_auc_score(yy, a)), 4), "holdout_winner": round(float(roc_auc_score(yy, b)), 4),
               "gain_ci95": block_bootstrap(dates[hold], lambda i: roc_auc_score(yy[i], b[i]) - roc_auc_score(yy[i], a[i]), max(2 * h, 20), reps=400),
               "full_oos_winner": round(float(roc_auc_score(y[m], P[win][m])), 4), "full_oos_current": round(float(roc_auc_score(y[m], P["current"][m])), 4)}
        res["adopt"] = res["gain_ci95"][0] > 0
        out[str(h)] = res
        print(h, json.dumps(res, separators=(",", ":")), flush=True)
    Path("reports/vol_auc_search.json").write_text(json.dumps(out, indent=1))


main()
