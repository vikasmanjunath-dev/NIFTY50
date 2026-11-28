"""Robustness suite on real point-in-time data: (1) cost realism, (2) CPCV distribution, (3) learned-regime ablation.
Uses LightGBM for speed (the ensemble was evaluated in run_phase2_pipeline); features = the serving model's scale-free set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.costs import (CostAssumption, ImpactCostModel, _sharpe, portfolio_returns,
                                       portfolio_returns_impact, signal_from_proba)
from nifty_analyser.eval.dataset import liquidity_panel, load_pit_dataset
from nifty_analyser.eval.metrics import evaluate_binary_predictions
from nifty_analyser.features.regime_learned import LearnedRegime, market_state
from nifty_analyser.models.baseline import fit_lightgbm
from nifty_analyser.validation.cpcv import cpcv_evaluate
from nifty_analyser.validation.walk_forward import assert_folds_are_leakage_free, purged_embargoed_walk_forward

H, EMB, SEED = 5, pd.Timedelta(days=5), 42
OUT = f"fwd_ret_{H}d_outcome_available_time"
RULE_REGIME = ["trend_regime_code", "vol_regime_code"]
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def ann(r: pd.Series) -> dict:
    return {"sharpe": round(float(_sharpe(r)), 3), "ann_return": round(float(r.mean() * 252), 4), "max_dd": round(float((r.cumsum() - r.cumsum().cummax()).min()), 4)}


def cost_analysis(df, cols):
    folds = purged_embargoed_walk_forward(df, 4, OUT, EMB)
    assert_folds_are_leakage_free(df, folds, OUT)
    X, parts = df[cols].to_numpy(float), []
    for f in folds:
        m = fit_lightgbm(X[f.train_idx], df["y"].to_numpy()[f.train_idx], random_state=SEED)
        parts.append(df.iloc[f.val_idx][["instrument_id", "event_time", "fwd_ret_1d"]].assign(p=m.predict_proba_up(X[f.val_idx])))
    p = pd.concat(parts, ignore_index=True)
    p["pos"], p["bh"] = signal_from_proba(p["p"].to_numpy()), 1.0
    liq = liquidity_panel(load_universe_bars())
    rows = {"gross (no cost)": ann(portfolio_returns(p, "pos", "fwd_ret_1d", CostAssumption(0, 0))["gross"]),
            "blended 20 bps": ann(portfolio_returns(p, "pos", "fwd_ret_1d", CostAssumption(10, 10))["net"])}
    for cap in (1e7, 1e8, 1e9, 1e10):
        r = portfolio_returns_impact(p, "pos", "fwd_ret_1d", liq, ImpactCostModel(capital=cap))
        rows[f"impact model, Rs {cap/1e7:g} cr"] = {**ann(r["net"]), "avg_cost_bps_per_day": round(float(r["cost"].mean() * 1e4), 2)}
    rows["equal-weight buy&hold (impact, Rs 10 cr)"] = ann(portfolio_returns_impact(p, "bh", "fwd_ret_1d", liq, ImpactCostModel())["net"])
    rows["equal-weight buy&hold (no cost)"] = ann(portfolio_returns(p, "bh", "fwd_ret_1d", CostAssumption(0, 0))["gross"])
    rows["_adv_coverage"] = round(float(p.merge(liq, on=["instrument_id", "event_time"], how="left")["adv_value"].notna().mean()), 3)
    return rows


def cpcv_analysis(df, cols):
    r = cpcv_evaluate(df, cols, fit_lightgbm, 8, 2, OUT, EMB, "fwd_ret_5d", seed=SEED)
    s = r[["accuracy", "roc_auc", "information_coefficient", "log_loss"]]
    return {"n_paths": len(r), "base_rate_up": round(float(df["y"].mean()), 4), "paths": r.round(4).to_dict("records"),
            "mean": s.mean().round(4).to_dict(), "std": s.std().round(4).to_dict(), "min": s.min().round(4).to_dict(), "max": s.max().round(4).to_dict()}


def regime_ablation(df, cols, seeds=(42, 7, 123)):
    folds = purged_embargoed_walk_forward(df, 6, OUT, EMB)
    base = [c for c in cols if c not in RULE_REGIME]
    state = market_state(df)
    rows = []
    for f in folds:
        tr, va = df.iloc[f.train_idx], df.iloc[f.val_idx]
        lr = LearnedRegime(3).fit(state.loc[state.index.isin(tr["event_time"].unique())])    # train dates only
        rp = lr.transform(state)
        learned = [c for c in rp.columns]
        a, b = tr.join(rp, on="event_time"), va.join(rp, on="event_time")
        variants = {"rule_codes": (cols, tr, va), "no_regime": (base, tr, va), "learned_gmm": (base + learned, a, b)}
        for s in seeds:
            for name, (cc, T, V) in variants.items():
                m = fit_lightgbm(T[cc].to_numpy(float), T["y"].to_numpy(), random_state=s)
                mt = evaluate_binary_predictions(V["y"].to_numpy(), m.predict_proba_up(V[cc].to_numpy(float)), V["fwd_ret_5d"].to_numpy())
                rows.append({"fold": f.fold_id, "seed": s, "variant": name, "auc": mt.roc_auc, "ic": mt.information_coefficient, "logloss": mt.log_loss})
    r = pd.DataFrame(rows)
    piv = r.groupby(["fold", "variant"])[["auc", "ic", "logloss"]].mean().unstack("variant")
    out = {"mean_by_variant": r.groupby("variant")[["auc", "ic", "logloss"]].mean().round(4).to_dict("index")}
    for ref in ("rule_codes", "no_regime"):
        d = (piv["auc"]["learned_gmm"] - piv["auc"][ref])
        out[f"learned_minus_{ref}_auc"] = {"mean": round(float(d.mean()), 5), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 5), "folds_better": int((d > 0).sum()), "n_folds": int(len(d)), "per_fold": d.round(4).tolist()}
    return out


def main():
    df, cols = load_pit_dataset(H)
    print(f"dataset {df.shape}, {len(cols)} features, {df['event_time'].min().date()}..{df['event_time'].max().date()}", flush=True)
    res = {"n_rows": len(df), "n_features": len(cols)}
    for name, fn in (("cost_realism", cost_analysis), ("cpcv", cpcv_analysis), ("regime_ablation", regime_ablation)):
        print(f"== {name}", flush=True)
        res[name] = fn(df, cols)
        print(json.dumps(res[name], indent=1, default=str)[:3000], flush=True)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "robustness.json").write_text(json.dumps(res, indent=1, default=str))
    print("done", flush=True)


if __name__ == "__main__":
    main()
