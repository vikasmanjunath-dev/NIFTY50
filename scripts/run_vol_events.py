"""Round 7 (docs/AUC_PLAN.md): announced board-meeting dates. Chosen on 2010-2020, confirmed once on 2021+ against the live model.
Adoption rule: the 95% date-block interval of the 2021+ AUC gain is above zero."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vol_auc3 import enrich, targets, walk
from run_vol_forecast_v2 import block_bootstrap, load

from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.features.volatility_events import add_event_features
from nifty_analyser.ops.vol_forecast import EXTRA_FEATURE_HORIZONS, load_history_training


def main():
    meetings = pd.read_parquet(DATA_RAW_DIR / "board_meetings.parquet")
    hist_bars = pd.read_parquet(DATA_RAW_DIR / "history" / "bars.parquet")
    for c in ("event_time", "available_time"):
        hist_bars[c] = pd.to_datetime(hist_bars[c], utc=True).dt.tz_convert("Asia/Kolkata").astype("datetime64[ns, Asia/Kolkata]")
    hist_bars = hist_bars[hist_bars.quality_flag != "FAIL"]
    now_bars = load_universe_bars()
    out = {}
    for h in (5, 20):
        df, cols = load(h)
        df = df.dropna(subset=["fwd_rv", "med_rv", "cur_rv", "outcome_time", "har_66"])
        df = df[(df.fwd_rv > 0) & (df.cur_rv > 0)]
        df, extra_cols, _ = enrich(df, now_bars, h)
        old, _, _ = enrich(load_history_training(h, extra=False), hist_bars, h)
        df, ev = add_event_features(df, meetings, h)
        old, _ = add_event_features(old, meetings, h)
        old = old.reindex(columns=df.columns)
        assert old.outcome_time.max() < df.event_time.min()
        live = cols + extra_cols if h in EXTRA_FEATURE_HORIZONS else cols
        sets = {"live": live, "events": live + ev}
        y_old = targets(old)[0]
        dev = {}
        for name, fc in sets.items():
            for kind in ("blend", "big_trees"):
                p = walk(old, fc, kind, 8, h)
                m = ~np.isnan(p)
                dev[(name, kind)] = float(roc_auc_score(y_old[m], p[m]))
        win = max(dev, key=dev.get)
        a = walk(df, live, "blend", 6, h, prior=old)
        b = walk(df, sets[win[0]], win[1], 6, h, prior=old)
        m = ~np.isnan(a)
        y, dates = targets(df)[0][m], df.event_time.to_numpy()[m]
        a, b = a[m], b[m]
        res = {"dev_auc_2010_2020": {"/".join(k): round(v, 4) for k, v in dev.items()}, "winner": "/".join(win),
               "auc_live_2021": round(float(roc_auc_score(y, a)), 4), "auc_winner_2021": round(float(roc_auc_score(y, b)), 4),
               "accuracy_live": round(float(((a > 0.5) == y).mean()), 4), "accuracy_winner": round(float(((b > 0.5) == y).mean()), 4),
               "rows_with_results_due": round(float(df.ev_results_in_horizon[m].mean()), 3),
               "gain_ci95": block_bootstrap(dates, lambda i: roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]), max(2 * h, 20), reps=400)}
        res["adopt"] = res["gain_ci95"][0] > 0
        out[str(h)] = res
        print(h, json.dumps(res, separators=(",", ":")), flush=True)
    Path("reports/vol_events.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
