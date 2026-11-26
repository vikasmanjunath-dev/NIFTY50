"""Does replacing level-type inputs with scale-free ones cost predictive information? Paired, identical folds.

OLD = the previous input set (raw price-unit indicators, raw macro levels, raw Rs-crore flows, raw revenue/profit/EPS),
rebuilt here from the same frames; NEW = the scale-free set the model now uses. Non-inferiority margin: mean AUC and IC
must not fall by more than 0.01 / 0.015 for either model family. `--all` then runs the full pipeline, rebuilds the serving
artifacts and recalibrates drift thresholds from ONE shared feature build.
Run: python3 scripts/run_scale_free_validation.py [--all]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nifty_analyser.config import REPORTS_DIR
from nifty_analyser.data.store import load_all_fii_dii_flows, load_all_macro_series, load_combined_fundamentals, load_universe_bars
from nifty_analyser.eval.metrics import evaluate_binary_predictions
from nifty_analyser.features.fundamentals import add_fundamental_ratios, merge_fundamentals_into_features
from nifty_analyser.models.baseline import fit_lightgbm, fit_logistic
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward
from scripts.run_phase2_pipeline import EMBARGO_DAYS, HORIZON, N_SPLITS, OUTCOME, SEED, build_features, build_universe_dataset

SMA, EMA = (5, 10, 20, 50, 100, 200), (5, 9, 12, 21, 26, 50, 100, 200)
MACRO = ("india_vix", "usd_inr", "brent_crude")
REPLACEMENTS = (
    [f"price_over_sma{w}" for w in (5, 10, 100)] + [f"price_over_ema{w}" for w in EMA]
    + ["sma50_over_sma200", "macd_pct", "macd_signal_pct", "macd_hist_pct", "obv_flow_10", "ad_flow_10", "profit_growth_yoy", "earnings_yield"]
    + [f"{m}_{k}" for m in MACRO for k in ("ret_5d", "ret_20d", "dev_252d")]
    + [f"{s}_net_rel{k}" for s in ("fii", "dii") for k in ("", "_5d", "_20d")])


def _levels_one(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("event_time")
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    o = {f"sma_{w}": c.rolling(w).mean() for w in SMA}
    o.update({f"ema_{w}": c.ewm(span=w, adjust=False, min_periods=w).mean() for w in EMA})
    pc = c.shift(1)
    o["atr_14"] = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1).rolling(14).mean()
    macd = o["ema_12"] - o["ema_26"]
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    o["macd"], o["macd_signal"], o["macd_hist"] = macd, sig, macd - sig
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    o["obv"], o["obv_slope_10"] = obv, obv.diff(10)
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    o["ad_line_slope_10"] = (mfm.fillna(0) * v).cumsum().diff(10)
    return pd.DataFrame(o, index=g.index).assign(instrument_id=g["instrument_id"], event_time=g["event_time"])


def add_old_level_features(feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    levels = pd.concat([_levels_one(g) for _, g in load_universe_bars().groupby("instrument_id")], ignore_index=True)
    out = feats.merge(levels, on=["instrument_id", "event_time"], how="left")
    names = [c for c in levels.columns if c not in ("instrument_id", "event_time")]
    left = out.reset_index(drop=True).assign(_o=lambda d: np.arange(len(d))).sort_values("available_time")
    for name, df in load_all_macro_series().items():
        r = df[["available_time", "close"]].rename(columns={"close": f"{name}_level"}).sort_values("available_time")
        left = pd.merge_asof(left, r, on="available_time", direction="backward")
        names.append(f"{name}_level")
    for name, df in load_all_fii_dii_flows().items():
        if df.empty:
            continue
        d = df.sort_values("event_time").copy()
        n = d["net_investment_cr"]
        r = pd.DataFrame({"available_time": d["available_time"], f"{name}_net_cr": n, f"{name}_net_cr_5d_sum": n.rolling(5, min_periods=3).sum(),
                          f"{name}_net_cr_20d_sum": n.rolling(20, min_periods=10).sum()})
        left = pd.merge_asof(left, r.sort_values("available_time"), on="available_time", direction="backward")
        names += [c for c in r.columns if c != "available_time"]
    out = left.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    fund = merge_fundamentals_into_features(out[["instrument_id", "event_time", "available_time"]], add_fundamental_ratios(load_combined_fundamentals()))
    out = out.merge(fund[["instrument_id", "event_time", "revenue", "profit", "eps_basic"]], on=["instrument_id", "event_time"], how="left")
    return out, names + ["revenue", "profit", "eps_basic"]


def compare(feats: pd.DataFrame, cols: list[str]) -> dict:
    feats_old, level_cols = add_old_level_features(feats)
    df, new_cols = build_universe_dataset(feats_old, cols, "pit")
    old_cols = [c for c in new_cols if c not in REPLACEMENTS] + level_cols
    print(f"{len(df)} rows | NEW {len(new_cols)} scale-free features | OLD {len(old_cols)} features ({len(level_cols)} level-type)", flush=True)
    folds = purged_embargoed_walk_forward(df, N_SPLITS, OUTCOME, pd.Timedelta(days=EMBARGO_DAYS))
    ret = df[f"fwd_ret_{HORIZON}d"].to_numpy()
    y = df["y"].to_numpy()
    out = {}
    for model, fit in (("lightgbm", fit_lightgbm), ("logistic", fit_logistic)):
        res = {}
        for tag, cs in (("old", old_cols), ("new", new_cols)):
            X = df[cs].to_numpy(dtype=float)
            res[tag] = pd.DataFrame([evaluate_binary_predictions(y[f.val_idx], fit(X[f.train_idx], y[f.train_idx], random_state=SEED).predict_proba_up(X[f.val_idx]),
                                                                 ret[f.val_idx], f.fold_id, model).to_dict() for f in folds])
        d = {m: (res["new"][m] - res["old"][m]).to_numpy() for m in ("roc_auc", "information_coefficient", "log_loss", "balanced_accuracy")}
        out[model] = {"old": {m: float(res["old"][m].mean()) for m in d}, "new": {m: float(res["new"][m].mean()) for m in d},
                      "mean_diff": {m: float(v.mean()) for m, v in d.items()}, "per_fold_diff": {m: [round(float(x), 4) for x in v] for m, v in d.items()},
                      "non_inferior": bool(d["roc_auc"].mean() >= -0.01 and d["information_coefficient"].mean() >= -0.015)}
        print(model, json.dumps({"old": out[model]["old"], "new": out[model]["new"], "diff": out[model]["mean_diff"], "per_fold_ic": out[model]["per_fold_diff"]["information_coefficient"],
                                 "non_inferior": out[model]["non_inferior"]}, indent=1), flush=True)
    report = {"n_new_features": len(new_cols), "n_old_features": len(old_cols), "level_features_removed": level_cols,
              "replacements_added": [c for c in REPLACEMENTS if c in new_cols], "results": out}
    (REPORTS_DIR / "scale_free_comparison.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    prebuilt = build_features()
    compare(prebuilt[0], prebuilt[1])
    if "--all" in sys.argv:
        from scripts import build_serving_artifacts, calibrate_drift, run_phase2_pipeline

        run_phase2_pipeline.run(prebuilt=prebuilt)
        build_serving_artifacts.run(prebuilt=prebuilt)
        calibrate_drift.run()
