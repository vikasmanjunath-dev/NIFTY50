"""Run after run_vol_forecast_v2.py and before run_vol_ranges.py.

Does pre-2021 training history raise volatility AUC? One pre-set variant, scored on exactly the same 2021+ rows.

Adoption rule, fixed before running: the 95% date-block interval of the AUC gain is above zero at BOTH horizons."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vol_forecast_v2 import block_bootstrap, load

from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.models.volatility import fit_predict_classifiers, fit_predict_regressors
from nifty_analyser.ops.vol_forecast import load_history_training
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward


def usable(df):
    df = df.dropna(subset=["fwd_rv", "med_rv", "cur_rv", "outcome_time", "har_66"])
    return df[(df.fwd_rv > 0) & (df.cur_rv > 0)].reset_index(drop=True)


def main():
    out, oos = {}, {}
    for h in (5, 20):
        df, cols = load(h)
        df = usable(df)
        old = load_history_training(h).reindex(columns=["outcome_time", "fwd_rv", "med_rv", *cols])
        assert old.outcome_time.max() < df.event_time.min()
        X, Xo = df[cols].to_numpy(float), old[cols].to_numpy(float)
        y, yo = (df.fwd_rv > df.med_rv).astype(int).to_numpy(), (old.fwd_rv > old.med_rv).astype(int).to_numpy()
        lv, lvo = np.log(df.fwd_rv).to_numpy(), np.log(old.fwd_rv).to_numpy()
        n = len(df)
        P = {k: np.full(n, np.nan) for k in ("cur", "new")}
        R = {k: np.full(n, np.nan) for k in ("cur", "new")}
        mean_lv = np.full(n, np.nan)
        for f in purged_embargoed_walk_forward(df, 6, "outcome_time", pd.Timedelta(days=int(h * 1.6) + 3)):
            tr, va = f.train_idx, f.val_idx
            mean_lv[va] = lv[tr].mean()
            for k, Xt, yt, lt in (("cur", X[tr], y[tr], lv[tr]),
                                  ("new", np.vstack([Xo, X[tr]]), np.r_[yo, y[tr]], np.r_[lvo, lv[tr]])):
                P[k][va] = fit_predict_classifiers(Xt, yt, X[va])["blend"]
                R[k][va] = fit_predict_regressors(Xt, lt, X[va])["blend"]
        m = ~np.isnan(P["cur"])
        dates, yy, a, b = df.event_time.to_numpy()[m], y[m], P["cur"][m], P["new"][m]
        r2 = lambda p: round(float(1 - ((lv - p)[m] ** 2).sum() / ((lv - mean_lv)[m] ** 2).sum()), 4)
        res = {"n_oos": int(m.sum()), "history_rows": len(old), "auc_current": round(float(roc_auc_score(yy, a)), 4),
               "auc_with_history": round(float(roc_auc_score(yy, b)), 4),
               "gain_ci95": block_bootstrap(dates, lambda i: roc_auc_score(yy[i], b[i]) - roc_auc_score(yy[i], a[i]), max(2 * h, 20), reps=400),
               "accuracy_current": round(float(((a > 0.5) == yy).mean()), 4), "accuracy_with_history": round(float(((b > 0.5) == yy).mean()), 4),
               "r2_current": r2(R["cur"]), "r2_with_history": r2(R["new"])}
        out[str(h)] = res
        oos[h] = df[m][["instrument_id", "event_time"]].assign(y=yy, lv=lv[m], p=b, r=R["new"][m])
        print(h, json.dumps(res, separators=(",", ":")), flush=True)
    out["adopted"] = all(out[h]["gain_ci95"][0] > 0 for h in ("5", "20"))
    print("adopted", out["adopted"])
    if out["adopted"]:  # the range study (run_vol_ranges.py) must be calibrated on the model that is live
        for h, frame in oos.items():
            frame.to_parquet(DATA_PROCESSED_DIR / f"vol_oos_h{h}.parquet", index=False)
    Path("reports/vol_history.json").write_text(json.dumps(out, indent=1))


main()
