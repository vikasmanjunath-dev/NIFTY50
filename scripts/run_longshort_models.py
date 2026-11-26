"""Market-neutral long/short on walk-forward OOS scores for EVERY model family (LightGBM, elastic-net logistic, CatBoost,
stacked calibrated ensemble, rank-average) - closes the 'LightGBM-only portfolio test' caveat. Same folds, cost model, null."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.costs import ImpactCostModel
from nifty_analyser.eval.dataset import liquidity_panel, load_pit_dataset
from nifty_analyser.eval.longshort import longshort_returns, permutation_null, pivots, sharpe
from nifty_analyser.models.baseline import BASE_MODEL_FITTERS
from nifty_analyser.models.ensemble import fit_ensemble
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward

OUT, EMB, SEED = "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5), 42
df, cols = load_pit_dataset(5); X, y = df[cols].to_numpy(float), df["y"].to_numpy()
scores = {k: [] for k in (*BASE_MODEL_FITTERS, "ensemble")}; keep = []
for f in purged_embargoed_walk_forward(df, 6, OUT, EMB):
    keep.append(df.iloc[f.val_idx][["instrument_id", "event_time", "fwd_ret_1d"]].reset_index(drop=True))
    for n, fit in BASE_MODEL_FITTERS.items():
        scores[n].append(fit(X[f.train_idx], y[f.train_idx], SEED).predict_proba_up(X[f.val_idx]))
    scores["ensemble"].append(fit_ensemble(df, cols, "y", f.train_idx, OUT, EMB, random_state=SEED).predict_proba_up(X[f.val_idx]))
    print("fold", f.fold if hasattr(f, "fold") else "", flush=True)
base = pd.concat(keep, ignore_index=True)
for n in list(scores): base[n] = np.concatenate(scores[n])
base["rank_avg"] = base.groupby("event_time")[list(BASE_MODEL_FITTERS)].rank(pct=True).mean(axis=1)
liq = liquidity_panel(load_universe_bars()); res = {}
for n in (*scores, "rank_avg"):
    pv = pivots(base[["instrument_id", "event_time", "fwd_ret_1d"]].assign(score=base[n]), "score", "fwd_ret_1d", liq); row = {}
    for hold in (5, 20):
        for label, model in (("gross", ImpactCostModel(0, 0, 0)), ("net", ImpactCostModel())):
            r = longshort_returns(pv, q=0.2, hold=hold, model=model)
            row[f"hold{hold}_{label}_sharpe"] = round(sharpe(r["net"]), 3)
    g = longshort_returns(pv, q=0.2, hold=5, model=ImpactCostModel(0, 0, 0)); null = permutation_null(pv, n=200, q=0.2, model=ImpactCostModel(0, 0, 0))
    row["hold5_gross_perm_p"] = round(float(((null.gross >= sharpe(g["net"])).sum() + 1) / (len(null) + 1)), 4)
    res[n] = row; print(n, json.dumps(row), flush=True)
Path("reports/longshort_models.json").write_text(json.dumps(res, indent=1)); print("done")
