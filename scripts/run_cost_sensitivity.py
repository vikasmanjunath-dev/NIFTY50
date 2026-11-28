"""How wrong can the (uncalibrated) impact coefficient be before the conclusion flips? LightGBM walk-forward long/short, net Sharpe grid."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.costs import ImpactCostModel
from nifty_analyser.eval.dataset import liquidity_panel, load_pit_dataset
from nifty_analyser.eval.longshort import longshort_returns, pivots, sharpe
from nifty_analyser.models.baseline import fit_lightgbm
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward
OUT, EMB = "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5)
df, cols = load_pit_dataset(5); X, y = df[cols].to_numpy(float), df["y"].to_numpy(); parts = []
for f in purged_embargoed_walk_forward(df, 6, OUT, EMB):
    parts.append(df.iloc[f.val_idx][["instrument_id", "event_time", "fwd_ret_1d"]].assign(score=fit_lightgbm(X[f.train_idx], y[f.train_idx], 42).predict_proba_up(X[f.val_idx])))
pv = pivots(pd.concat(parts, ignore_index=True), "score", "fwd_ret_1d", liquidity_panel(load_universe_bars())); res = {}
for cap in (1e8, 1e9):
    for hold in (5, 20):
        for fx in (10.0, 5.0, 2.0):          # explicit fees+taxes bps/side: the other big unknown (STT alone is ~10 bps on delivery sells)
            for ic in (0.0, 0.25, 0.5, 1.0):
                r = longshort_returns(pv, q=0.2, hold=hold, model=ImpactCostModel(fx, 5.0, ic, cap))
                res[f"cap={cap:.0e}|hold={hold}|fixed={fx:g}|impact={ic}"] = round(sharpe(r["net"]), 3)
Path("reports/cost_sensitivity.json").write_text(json.dumps(res, indent=1))
pos = {k: v for k, v in res.items() if v > 0}; print(len(pos), "of", len(res), "cells have positive net Sharpe"); print(json.dumps(pos, indent=0))
