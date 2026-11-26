"""Zero-shot Hugging Face time-series model (amazon/chronos-bolt-small) versus the volatility blend, on identical rows.

Chronos forecasts the daily log-variance path from the last 256 days only. Dates are spaced one horizon apart (non-overlapping);
its linear calibration is fit on the first third of those dates and everything is scored on the rest. Run run_vol_forecast_v2.py first."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.features.volatility import EPS, daily_variance

pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="mps" if torch.backends.mps.is_available() else "cpu", torch_dtype=torch.float32)
series = {}
for iid, g in load_universe_bars().groupby("instrument_id"):
    g = g.sort_values("event_time")
    ret2 = np.log(g.close / g.close.shift(1)) ** 2
    series[iid] = (pd.DatetimeIndex(g.event_time), np.log(daily_variance(g) + EPS).to_numpy(), ret2.to_numpy())

for h in (5, 20):
    d = pd.read_parquet(DATA_PROCESSED_DIR / f"vol_oos_h{h}.parquet")
    d = d[d.event_time.isin(np.sort(d.event_time.unique())[::h])].reset_index(drop=True)
    contexts, offsets = [], []
    for r in d.itertuples():
        times, logvar, ret2 = series[r.instrument_id]
        i = times.get_loc(r.event_time)
        window = logvar[max(0, i - 255): i + 1]
        contexts.append(torch.tensor(window[np.isfinite(window)], dtype=torch.float32))
        # map range-based variance onto the close-to-close target using only the trailing year
        offsets.append(np.log(np.nanmean(ret2[max(0, i - 251): i + 1]) + EPS) - np.log(np.nanmean(np.exp(logvar[max(0, i - 251): i + 1])) + EPS))
    forecasts = []
    for s in range(0, len(contexts), 256):
        q, _ = pipe.predict_quantiles(contexts[s: s + 256], prediction_length=h, quantile_levels=[0.5])
        forecasts.append(q[:, :, 0].cpu().numpy())
    chronos = 0.5 * (np.log(np.exp(np.concatenate(forecasts)).sum(1)) + np.array(offsets))
    # the threshold (own median) is recovered from the target definition used by the study
    med = {}
    for iid, (times, _, ret2) in series.items():
        cur = np.sqrt(pd.Series(ret2, index=times).rolling(h).sum())
        med[iid] = cur.rolling(252, min_periods=120).median()
    threshold = np.log(np.array([med[r.instrument_id].loc[r.event_time] for r in d.itertuples()]))
    cut = np.sort(d.event_time.unique())[d.event_time.nunique() // 3]
    fit, test = (d.event_time < cut - pd.Timedelta(days=2 * h)).to_numpy(), (d.event_time >= cut).to_numpy()
    slope, intercept = np.polyfit(chronos[fit], d.lv[fit], 1)
    cal = intercept + slope * chronos
    both = (cal + d.r.to_numpy()) / 2
    base = d.lv[fit].mean()

    def row(score_cls, pred):
        out = {"auc": round(float(roc_auc_score(d.y[test], score_cls[test])), 4)}
        if pred is not None:
            out["r2"] = round(float(1 - ((d.lv - pred)[test] ** 2).sum() / ((d.lv[test] - base) ** 2).sum()), 4)
        return out

    res = {"horizon_days": h, "n_test": int(test.sum()), "n_test_dates": int(d.event_time[test].nunique()),
           "chronos_zero_shot": row(chronos - threshold, cal), "our_blend": row(d.p.to_numpy(), d.r.to_numpy()),
           "average_of_both_regression": row(both - threshold, both)}
    Path(f"reports/vol_chronos_h{h}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, separators=(",", ":")))
