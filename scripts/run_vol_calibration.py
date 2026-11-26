"""Two calibration fixes, rules fixed before running. Run run_vol_forecast_v2.py and run_vol_ranges.py first.

(1) Adaptive ranges replace static ones if 90% coverage is closer to 90% at both horizons and pinball loss is not more than 0.5% worse.
(2) Probability recalibration is adopted if the Brier-score change has a 95% date-block interval below zero at both horizons."""
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
from nifty_analyser.models.ranges import LEVELS, adaptive_ranges, pinball, walk_forward_ranges, walk_forward_recalibrate

bars = load_universe_bars().sort_values(["instrument_id", "event_time"])
out, finals = {}, {}
for h in (5, 20):
    fwd = pd.concat([pd.DataFrame({"instrument_id": i, "event_time": g.event_time, "ret": np.log(g.close.shift(-h) / g.close)})
                     for i, g in bars.groupby("instrument_id")])
    d = pd.read_parquet(DATA_PROCESSED_DIR / f"vol_oos_h{h}.parquet").merge(fwd, on=["instrument_id", "event_time"]).dropna()
    d = d.sort_values("event_time").reset_index(drop=True)
    dates, ret, scale = d.event_time.to_numpy(), d.ret.to_numpy(), np.exp(d.r.to_numpy())
    static = walk_forward_ranges(dates, ret, scale, h)
    adaptive, finals[h] = adaptive_ranges(dates, ret, scale, h)
    m = ~np.isnan(static[:, 0])
    res = {"ranges": {}}
    for name, v in (("static", static), ("adaptive", adaptive)):
        res["ranges"][name] = {"coverage_90": round(float(((ret >= v[:, 0]) & (ret <= v[:, 3]))[m].mean()), 4),
                               "coverage_50": round(float(((ret >= v[:, 1]) & (ret <= v[:, 2]))[m].mean()), 4),
                               "pinball_sum": round(sum(pinball(ret[m], v[m, i], q) for i, q in enumerate(LEVELS)), 6)}
    cal = walk_forward_recalibrate(dates, d.p.to_numpy(), d.y.to_numpy(), h)
    c = ~np.isnan(cal)
    y, raw, new = d.y.to_numpy()[c], d.p.to_numpy()[c], cal[c]
    diff = (new - y) ** 2 - (raw - y) ** 2
    res["recalibration"] = {"n": int(c.sum()), "brier_raw": round(float(((raw - y) ** 2).mean()), 4), "brier_new": round(float(((new - y) ** 2).mean()), 4),
                            "accuracy_raw": round(float(((raw > 0.5) == y).mean()), 4), "accuracy_new": round(float(((new > 0.5) == y).mean()), 4),
                            "brier_change_ci95": block_bootstrap(dates[c], lambda i: diff[i].mean(), max(2 * h, 20))}
    out[str(h)] = res
    print(h, json.dumps(res, separators=(",", ":")))
r = [out[h]["ranges"] for h in ("5", "20")]
out["adopt_adaptive_ranges"] = all(abs(x["adaptive"]["coverage_90"] - 0.9) < abs(x["static"]["coverage_90"] - 0.9) and x["adaptive"]["pinball_sum"] <= 1.005 * x["static"]["pinball_sum"] for x in r)
out["adopt_recalibration"] = all(out[h]["recalibration"]["brier_change_ci95"][1] < 0 for h in ("5", "20"))
print("adopt ranges", out["adopt_adaptive_ranges"], "recal", out["adopt_recalibration"])
Path("reports/vol_calibration.json").write_text(json.dumps(out, indent=1))
if out["adopt_adaptive_ranges"]:
    for h, q in finals.items():
        p = Path(f"reports/vol_ranges_h{h}.json")
        j = json.loads(p.read_text())
        j["live_quantiles"], j["live_method"] = {str(k): v for k, v in q.items()}, "adaptive"
        p.write_text(json.dumps(j, indent=1))
