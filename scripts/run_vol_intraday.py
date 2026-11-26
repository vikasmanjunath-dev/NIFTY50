"""Round 6 (docs/AUC_PLAN.md): intraday realised-volatility features. Scored from 2023; the candidate is chosen on the first half
of those dates and confirmed on the second. Adoption rule: the 95% date-block interval of the second-half AUC gain is above zero."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vol_forecast_v2 import block_bootstrap, load

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.features.volatility_extra import extra_features
from nifty_analyser.features.volatility_intraday import intraday_features
from nifty_analyser.models.volatility import fit_predict_classifiers
from nifty_analyser.ops.vol_forecast import EXTRA_FEATURE_HORIZONS, load_history_training

KEY = ["instrument_id", "event_time"]


def main():
    intra, intra_cols = intraday_features(pd.read_parquet(DATA_RAW_DIR / "intraday" / "daily_summary.parquet"))
    extra, extra_cols = extra_features(load_universe_bars())
    out = {}
    for h in (5, 20):
        df, cols = load(h)
        df = df.dropna(subset=["fwd_rv", "med_rv", "cur_rv", "outcome_time", "har_66"])
        df = df[(df.fwd_rv > 0) & (df.cur_rv > 0)].merge(extra, on=KEY, how="left")
        df["date"] = df.event_time.dt.strftime("%Y-%m-%d")
        df = df.merge(intra, on=["instrument_id", "date"], how="left").sort_values(["event_time", "instrument_id"]).reset_index(drop=True)
        old = load_history_training(h).reindex(columns=df.columns)
        live = cols + extra_cols if h in EXTRA_FEATURE_HORIZONS else cols
        y, yo = (df.fwd_rv > df.med_rv).astype(int).to_numpy(), (old.fwd_rv > old.med_rv).astype(int).to_numpy()
        last = pd.Timestamp(intra.date.max(), tz="Asia/Kolkata")  # score only dates the intraday data covers
        dates = np.sort(df.event_time[(df.event_time >= pd.Timestamp("2023-01-01", tz="Asia/Kolkata")) & (df.event_time <= last)].unique())
        embargo = pd.Timedelta(days=int(h * 1.6) + 3)
        candidates = {"live": (live, True, False), "intraday": (live + intra_cols, True, False), "intraday_recent_only": (live + intra_cols, False, True)}
        P = {}
        for name, (fc, with_history, recent) in candidates.items():
            X, Xo, p = df[fc].to_numpy(float), old[fc].to_numpy(float), np.full(len(df), np.nan)
            for block in np.array_split(dates, 8):
                start = pd.Timestamp(block[0])
                start = start if start.tz else start.tz_localize("UTC").tz_convert("Asia/Kolkata")
                va = df.event_time.isin(block).to_numpy()
                tr = ((df.event_time < start - embargo) & (df.outcome_time < start)).to_numpy()
                if recent:
                    tr &= (df.event_time >= pd.Timestamp("2022-04-01", tz="Asia/Kolkata")).to_numpy()
                assert df.outcome_time[tr].max() < df.event_time[va].min()
                Xt, yt = (np.vstack([Xo, X[tr]]), np.r_[yo, y[tr]]) if with_history else (X[tr], y[tr])
                p[va] = fit_predict_classifiers(Xt, yt, X[va])["blend"]
            P[name] = p
        m = ~np.isnan(P["live"])
        when = df.event_time.to_numpy()
        k = len(dates) // 2
        first = m & df.event_time.isin(dates[: k - h]).to_numpy()
        second = m & df.event_time.isin(dates[k:]).to_numpy()
        dev = {n: round(float(roc_auc_score(y[first], p[first])), 4) for n, p in P.items()}
        win = max(dev, key=dev.get)
        yy, a, b, w = y[second], P["live"][second], P[win][second], when[second]
        res = {"intraday_coverage": round(float(df.rv_5[m].notna().mean()), 3), "first_half_auc": dev, "winner": win,
               "second_half_live": round(float(roc_auc_score(yy, a)), 4), "second_half_winner": round(float(roc_auc_score(yy, b)), 4),
               "second_half_all": {n: round(float(roc_auc_score(yy, p[second])), 4) for n, p in P.items()},
               "gain_ci95": block_bootstrap(w, lambda i: roc_auc_score(yy[i], b[i]) - roc_auc_score(yy[i], a[i]), max(2 * h, 20), reps=400),
               "all_scored_live": round(float(roc_auc_score(y[m], P["live"][m])), 4), "all_scored_winner": round(float(roc_auc_score(y[m], P[win][m])), 4)}
        res["adopt"] = win != "live" and res["gain_ci95"][0] > 0
        out[str(h)] = res
        print(h, json.dumps(res, separators=(",", ":")), flush=True)
    Path("reports/vol_intraday.json").write_text(json.dumps(out, indent=1))


main()
