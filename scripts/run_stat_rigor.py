"""Selection-aware evidence for the faint cross-sectional signal: seed-averaged tree blend (3 seeds x LightGBM+CatBoost), market-neutral
vs SECTOR-neutral long/short, Rs 10 cr impact costs, deflated Sharpe over n_trials in {10,50,100,500}, effective sample size,
stationary-bootstrap CI. OOS scores cached at data/processed/oos_blend_scores.parquet."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.costs import ImpactCostModel
from nifty_analyser.eval.dataset import liquidity_panel, load_pit_dataset
from nifty_analyser.eval.longshort import longshort_returns, pivots, sharpe
from nifty_analyser.eval.stat_rigor import bootstrap_sharpe_ci, deflated_sharpe, effective_n
from nifty_analyser.models.baseline import fit_catboost, fit_lightgbm
from nifty_analyser.models.ensemble import _logit
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward
OUT, EMB, CACHE = "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5), DATA_PROCESSED_DIR / "oos_blend_scores.parquet"
if CACHE.exists():
    base = pd.read_parquet(CACHE)
else:
    df, cols = load_pit_dataset(5); X, y = df[cols].to_numpy(float), df["y"].to_numpy(); parts = []
    for f in purged_embargoed_walk_forward(df, 6, OUT, EMB):
        z = np.mean([_logit(fit(X[f.train_idx], y[f.train_idx], s).predict_proba_up(X[f.val_idx])) for s in (1, 2, 3) for fit in (fit_lightgbm, fit_catboost)], axis=0)
        parts.append(df.iloc[f.val_idx][["instrument_id", "event_time", "fwd_ret_1d"]].assign(score=z)); print("fold", flush=True)
    base = pd.concat(parts, ignore_index=True); base.to_parquet(CACHE)
base["sector"] = base.instrument_id.map(lambda i: get_instrument(i).sector)
base["score_sn"] = base.score - base.groupby(["event_time", "sector"]).score.transform("mean")        # rank within sector: no sector bets
liq = liquidity_panel(load_universe_bars()); res = {"n_oos_days": int(base.event_time.nunique())}
# dispersion of annualised Sharpes over the variants explored so far -> per-period trial variance for the DSR benchmark
prior = [v for k, d in json.load(open("reports/blend_seeds.json")).items() if k != "summary_mean_by_model" for kk, v in d.items() if kk.endswith("net") or kk.endswith("gross")]
trial_var = float(np.var(prior, ddof=1)) / 252.0; res["trial_sharpe_sd_ann"] = round(float(np.sqrt(trial_var * 252)), 3)
for variant, sc in (("market_neutral", "score"), ("sector_neutral", "score_sn")):
    pv = pivots(base.assign(score=base[sc])[["instrument_id", "event_time", "fwd_ret_1d", "score"]], "score", "fwd_ret_1d", liq)
    for hold in (5, 20):
        for lab, m in (("gross", ImpactCostModel(0, 0, 0)), ("net", ImpactCostModel())):
            r = longshort_returns(pv, q=0.2, hold=hold, model=m)["net"].to_numpy()
            row = {"sharpe": round(sharpe(r), 3), "n_eff": round(effective_n(r), 0)}
            if lab == "net" or hold == 20:
                lo, hi, p0 = bootstrap_sharpe_ci(r, n_boot=1000); row["boot_ci95"] = [round(lo, 2), round(hi, 2)]; row["boot_p_sharpe_le_0"] = round(p0, 3)
                row["dsr_by_n_trials"] = {n: round(deflated_sharpe(r, n, trial_var)["dsr"], 3) for n in (10, 50, 100, 500)}
            res[f"{variant}|hold{hold}|{lab}"] = row; print(f"{variant}|hold{hold}|{lab}", json.dumps(row), flush=True)
Path("reports/stat_rigor.json").write_text(json.dumps(res, indent=1)); print("done")
