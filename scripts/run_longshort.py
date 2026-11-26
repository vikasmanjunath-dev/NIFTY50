"""Market-neutral cross-sectional long/short on walk-forward OOS scores, net of impact costs, vs a within-date permutation null."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.costs import ImpactCostModel
from nifty_analyser.eval.dataset import liquidity_panel, load_pit_dataset
from nifty_analyser.eval.longshort import longshort_returns, permutation_null, pivots, sharpe
from nifty_analyser.models.baseline import fit_lightgbm
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward

OUT, EMB, SEED = "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5), 42
df, cols = load_pit_dataset(5)
X, y = df[cols].to_numpy(float), df["y"].to_numpy()
parts = []
for f in purged_embargoed_walk_forward(df, 6, OUT, EMB):
    m = fit_lightgbm(X[f.train_idx], y[f.train_idx], SEED)
    parts.append(df.iloc[f.val_idx][["instrument_id", "event_time", "fwd_ret_1d"]].assign(score=m.predict_proba_up(X[f.val_idx])))
p = pd.concat(parts, ignore_index=True)
print("OOS", p.shape, p.event_time.min().date(), p.event_time.max().date(), flush=True)
pv = pivots(p, "score", "fwd_ret_1d", liquidity_panel(load_universe_bars()))
res = {"oos_days": len(pv["score"]), "n_names": pv["score"].shape[1]}
for q in (0.1, 0.2, 0.3):
    row = {}
    for label, model in (("gross", ImpactCostModel(0, 0, 0)), ("net_Rs10cr", ImpactCostModel()), ("net_Rs100cr", ImpactCostModel(capital=1e9)), ("net_Rs1000cr", ImpactCostModel(capital=1e10))):
        r = longshort_returns(pv, q=q, model=model)
        row[label] = {"sharpe": round(sharpe(r["net"]), 3), "ann_ret": round(float(r["net"].mean() * 252), 4), "ann_turnover": round(float(r["turnover"].mean() * 252), 1)}
    res[f"q={q}"] = row
    print(q, json.dumps(row), flush=True)
res["hold_sweep_q0.2"] = {}
for hold in (5, 10, 20):
    for label, model in (("gross", ImpactCostModel(0, 0, 0)), ("net_Rs10cr", ImpactCostModel())):
        r = longshort_returns(pv, q=0.2, hold=hold, model=model)
        res["hold_sweep_q0.2"].setdefault(f"hold={hold}", {})[label] = {"sharpe": round(sharpe(r["net"]), 3), "ann_ret": round(float(r["net"].mean() * 252), 4), "ann_turnover": round(float(r["turnover"].mean() * 252), 1)}
print(json.dumps(res["hold_sweep_q0.2"]), flush=True)
res["flat_cost_sweep_q0.2_hold5"] = {f"{b}bps": round(sharpe(longshort_returns(pv, q=0.2, model=ImpactCostModel(b, 0, 0))["net"]), 3) for b in (0, 2, 5, 8, 10, 15)}
print(json.dumps(res["flat_cost_sweep_q0.2_hold5"]), flush=True)
null = permutation_null(pv, n=200, q=0.2, model=ImpactCostModel(0, 0, 0))
real = res["q=0.2"]["gross"]["sharpe"]
res["permutation_null_q0.2_gross"] = {"null_mean": round(float(null.gross.mean()), 3), "null_sd": round(float(null.gross.std()), 3), "real": real,
                                      "p_value_one_sided": round(float(((null.gross >= real).sum() + 1) / (len(null) + 1)), 4)}
print(json.dumps(res["permutation_null_q0.2_gross"]), flush=True)
Path("reports/longshort.json").write_text(json.dumps(res, indent=1)); print("done", flush=True)
