"""Calibrated expected-move ranges and whether the volatility forecast helps a portfolio. Run run_vol_forecast_v2.py first.

(1) Return ranges scaled by the model forecast versus by trailing realised volatility and by EWMA, on identical rows.
(2) Long-only portfolios rebalanced every h days: equal weight, inverse trailing vol, inverse forecast vol (gross, 10 bps turnover cost shown)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vol_forecast_v2 import block_bootstrap

from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.models.ranges import LEVELS, fit_quantiles, pinball, walk_forward_ranges

bars = load_universe_bars().sort_values(["instrument_id", "event_time"])
for h in (5, 20):
    parts = []
    for iid, g in bars.groupby("instrument_id"):
        r = np.log(g.close).diff()
        parts.append(pd.DataFrame({"instrument_id": iid, "event_time": g.event_time,
                                   "ret": np.log(g.close.shift(-h) / g.close),
                                   "trailing": np.sqrt((r ** 2).rolling(h).sum()),
                                   "ewma": np.sqrt((r ** 2).ewm(alpha=0.06, min_periods=60).mean() * h)}))
    d = pd.read_parquet(DATA_PROCESSED_DIR / f"vol_oos_h{h}.parquet").merge(pd.concat(parts), on=["instrument_id", "event_time"])
    d = d.dropna().query("trailing > 0").sort_values("event_time").reset_index(drop=True)
    d["model"] = np.exp(d.r)
    dates = d.event_time.to_numpy()
    q = {name: walk_forward_ranges(dates, d.ret, d[name], h) for name in ("model", "trailing", "ewma")}
    m = ~np.isnan(q["model"][:, 0])
    ret = d.ret.to_numpy()
    out = {"horizon_days": h, "n": int(m.sum()), "ranges": {}}
    loss = {}
    for name, v in q.items():
        inside = lambda a, b: round(float(((ret >= v[:, a]) & (ret <= v[:, b]))[m].mean()), 4)
        loss[name] = sum(np.maximum(l * (ret - v[:, i]), (l - 1) * (ret - v[:, i])) for i, l in enumerate(LEVELS))
        out["ranges"][name] = {"coverage_90": inside(0, 3), "coverage_50": inside(1, 2),
                               "mean_width_90": round(float((v[m, 3] - v[m, 0]).mean()), 4),
                               "pinball_sum": round(sum(pinball(ret[m], v[m, i], l) for i, l in enumerate(LEVELS)), 6)}
    for other in ("trailing", "ewma"):
        diff = (loss["model"] - loss[other])[m]
        lo, hi = block_bootstrap(dates[m], lambda i, diff=diff: 1e4 * diff[i].mean(), max(2 * h, 20))
        out[f"pinball_model_minus_{other}_ci95_bps"] = [lo, hi]
    out["live_quantiles"] = {str(k): v for k, v in fit_quantiles(d.ret, d.model).items()}

    # portfolios on non-overlapping rebalance dates
    port = {}
    rebal = np.sort(d.event_time.unique())[::h]
    prev = {}
    series = {k: [] for k in ("equal", "inverse_trailing", "inverse_model")}
    turnover = {k: [] for k in series}
    for day in rebal:
        g = d[d.event_time == day].set_index("instrument_id")
        simple = np.expm1(g.ret)
        for name, raw in (("equal", pd.Series(1.0, index=g.index)), ("inverse_trailing", 1 / g.trailing), ("inverse_model", 1 / g.model)):
            w = raw / raw.sum()
            series[name].append(float((w * simple).sum()))
            turnover[name].append(float(w.sub(prev.get(name, w * 0), fill_value=0).abs().sum()))
            prev[name] = w
    for name, s in series.items():
        s, t = np.array(s), np.array(turnover[name])
        net = s - 0.001 * t
        wealth = np.cumprod(1 + net)
        ann = 252 / h
        port[name] = {"ann_return_net": round(float(net.mean() * ann), 4), "ann_vol": round(float(net.std(ddof=1) * np.sqrt(ann)), 4),
                      "sharpe_net": round(float(net.mean() / net.std(ddof=1) * np.sqrt(ann)), 3),
                      "max_drawdown": round(float((wealth / np.maximum.accumulate(wealth) - 1).min()), 4),
                      "mean_turnover": round(float(t.mean()), 3)}
    out["portfolios"] = {"n_rebalances": len(rebal), **port}
    Path(f"reports/vol_ranges_h{h}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "live_quantiles"}, separators=(",", ":")))
