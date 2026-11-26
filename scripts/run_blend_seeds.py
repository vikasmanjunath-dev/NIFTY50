"""Seed robustness of the faint cross-sectional signal + tree blend (no data given up to calibration slices).
Seeds 1-3: LightGBM, CatBoost, logit-average blend on identical purged walk-forward folds; gross/net long/short Sharpe + daily IC."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.costs import ImpactCostModel
from nifty_analyser.eval.dataset import liquidity_panel, load_pit_dataset
from nifty_analyser.eval.longshort import longshort_returns, pivots, sharpe
from nifty_analyser.models.baseline import fit_catboost, fit_lightgbm
from nifty_analyser.models.ensemble import _logit
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward
OUT, EMB = "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5)
df, cols = load_pit_dataset(5); X, y = df[cols].to_numpy(float), df["y"].to_numpy(); folds = list(purged_embargoed_walk_forward(df, 6, OUT, EMB))
base = pd.concat([df.iloc[f.val_idx][["instrument_id", "event_time", "fwd_ret_1d", "fwd_ret_5d"]] for f in folds], ignore_index=True)
liq = liquidity_panel(load_universe_bars()); res = {}
def evaluate(name, score):
    d = base.assign(score=score); pv = pivots(d, "score", "fwd_ret_1d", liq); row = {}
    for hold in (5, 20):
        for lab, m in (("gross", ImpactCostModel(0, 0, 0)), ("net", ImpactCostModel())):
            row[f"h{hold}_{lab}"] = round(sharpe(longshort_returns(pv, q=0.2, hold=hold, model=m)["net"]), 3)
    ic = d.groupby("event_time").apply(lambda g: g.score.corr(g.fwd_ret_5d, method="spearman")).dropna()
    row["daily_ic"] = round(float(ic.mean()), 4); row["ic_t"] = round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 2)
    res[name] = row; print(name, json.dumps(row), flush=True)
for seed in (1, 2, 3):
    l = np.concatenate([fit_lightgbm(X[f.train_idx], y[f.train_idx], seed).predict_proba_up(X[f.val_idx]) for f in folds])
    c = np.concatenate([fit_catboost(X[f.train_idx], y[f.train_idx], seed).predict_proba_up(X[f.val_idx]) for f in folds])
    evaluate(f"lightgbm_s{seed}", l); evaluate(f"catboost_s{seed}", c); evaluate(f"blend_s{seed}", (_logit(l) + _logit(c)) / 2)
R = pd.DataFrame(res).T
res["summary_mean_by_model"] = {m: R[R.index.str.startswith(m)].mean().round(3).to_dict() for m in ("lightgbm", "catboost", "blend")}
Path("reports/blend_seeds.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["summary_mean_by_model"], indent=1)); print("done")
