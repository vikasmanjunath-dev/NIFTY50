"""End-to-end Phase 2 pipeline (Blueprint §6.2): every enrichment layer -> purged/embargoed
walk-forward -> champion/challenger ensemble (logistic + LightGBM + CatBoost -> meta -> calibrated)
-> selective prediction, regime-segmented metrics, cost-stressed backtest -> locked report.

Run on BOTH the biased (today's 50) and the point-in-time (real membership) universes; the PIT run
is the headline result. Layers whose cached data is missing are skipped *and recorded in the report*
(never silently). Run: python3 scripts/run_phase2_pipeline.py"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR, REPORTS_DIR
from nifty_analyser.data.ingest import load_cached
from nifty_analyser.data.instrument_master import NIFTY_INDEX
from nifty_analyser.data.store import (assert_no_lookahead, load_all_fii_dii_flows, load_all_macro_series,
                                       load_combined_fundamentals, load_delivery_universe, load_universe_bars)
from nifty_analyser.eval.costs import CostAssumption, cost_stress_table, signal_from_proba
from nifty_analyser.eval.metrics import evaluate_binary_predictions
from nifty_analyser.eval.survivorship import label_dataset, universe_frames
from nifty_analyser.features.build import build_feature_matrix, feature_columns
from nifty_analyser.features.cross_sectional import add_index_relative_features, add_same_day_cross_sectional
from nifty_analyser.features.delivery import add_delivery_indicators, merge_delivery_into_features
from nifty_analyser.features.fii_dii_features import add_all_flow_features
from nifty_analyser.features.fundamentals import add_fundamental_ratios, finalize_fundamental_features, merge_fundamentals_into_features
from nifty_analyser.monitoring.drift import is_scale_dependent
from nifty_analyser.features.macro_features import add_all_macro_features, add_macro_series_indicators
from nifty_analyser.features.regime import add_regime_features
from nifty_analyser.models.ensemble import fit_ensemble
from nifty_analyser.models.selective import coverage_precision_curve
from nifty_analyser.validation.walk_forward import assert_folds_are_leakage_free, purged_embargoed_walk_forward

HORIZON, N_SPLITS, EMBARGO_DAYS, SEED = 5, 4, 5, 42
BASE_COST = CostAssumption(cost_bps=10.0, slippage_bps=10.0)
OUTCOME = f"fwd_ret_{HORIZON}d_outcome_available_time"


def build_features() -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Per-instrument feature matrix over the whole pool + every available layer."""
    layers: dict[str, str] = {}
    feats = build_feature_matrix(load_universe_bars())
    index_bars = load_cached(NIFTY_INDEX.instrument_id)
    feats = add_index_relative_features(feats, index_bars)
    layers["price_technical_index_relative"] = "included"

    macro = load_all_macro_series()
    feats = add_all_macro_features(feats, macro)
    feats = add_regime_features(feats, index_bars, add_macro_series_indicators(macro["india_vix"], "india_vix"))
    layers["macro_and_regime"] = "included"

    if (DATA_RAW_DIR / "delivery_nifty50.parquet").exists():
        feats = merge_delivery_into_features(feats, add_delivery_indicators(load_delivery_universe()))
        layers["delivery"] = "included"
    else:
        layers["delivery"] = "SKIPPED (no cache)"
    if (DATA_RAW_DIR / "fundamentals_nifty50.parquet").exists():
        feats = finalize_fundamental_features(merge_fundamentals_into_features(feats, add_fundamental_ratios(load_combined_fundamentals())))
        layers["fundamentals"] = "included" + ("" if (DATA_RAW_DIR / "insurance_fundamentals_nifty50.parquet").exists() else " (general XBRL only; insurers missing)")
    else:
        layers["fundamentals"] = "SKIPPED (no cache)"
    flows = load_all_fii_dii_flows()
    if not flows["fii"].empty:
        feats = add_all_flow_features(feats, flows)
        layers["fii_dii"] = "included" + ("" if not flows["dii"].empty else " (FII only)")
    else:
        layers["fii_dii"] = "SKIPPED (no cache)"
    inf = {c: int(np.isinf(feats[c]).sum()) for c in feature_columns(feats) if np.isinf(feats[c]).any()}
    assert not inf, f"infinite values in feature columns (fix at the source, don't clip): {inf}"
    levels = [c for c in feature_columns(feats) if is_scale_dependent(c)]
    assert not levels, f"scale-dependent level features must not be model inputs: {levels}"
    cols = feature_columns(feats)          # frozen BEFORE labels exist, so no label column can leak in
    leaked = [c for c in cols if c.startswith(("fwd_ret", "dir_label", "y_")) or c == "y"]
    assert not leaked, f"label-derived columns in the feature list: {leaked}"
    return feats, cols, layers


def build_universe_dataset(feats: pd.DataFrame, cols: list[str], universe: str) -> tuple[pd.DataFrame, list[str]]:
    labelled = label_dataset(feats, HORIZON)
    biased, pit = universe_frames(labelled)
    df = pit if universe == "pit" else biased
    df = add_same_day_cross_sectional(df)   # ranked only against that universe's own members on the day
    xs = [c for c in df.columns if c.startswith(("xs_rank_", "sector_rel_"))]
    df = df.sort_values("event_time").reset_index(drop=True)   # RangeIndex: fold indices are positional
    return df, cols + xs


def evaluate(df: pd.DataFrame, cols: list[str], universe: str) -> dict:
    folds = purged_embargoed_walk_forward(df, N_SPLITS, OUTCOME, pd.Timedelta(days=EMBARGO_DAYS))
    assert_folds_are_leakage_free(df, folds, OUTCOME)
    X = df[cols].to_numpy(dtype=float)
    per_fold, regime_rows, preds, selective_rows = [], [], [], []
    for f in folds:
        assert_no_lookahead(df.iloc[f.train_idx][["instrument_id", "event_time", "available_time"]], f.val_start)
        ens = fit_ensemble(df, cols, "y", f.train_idx, OUTCOME, pd.Timedelta(days=EMBARGO_DAYS), random_state=SEED)
        val = df.iloc[f.val_idx]
        proba = ens.predict_proba_up(X[f.val_idx])
        m = evaluate_binary_predictions(val["y"].to_numpy(), proba, val[f"fwd_ret_{HORIZON}d"].to_numpy(), f.fold_id, "ensemble_calibrated", len(f.train_idx)).to_dict()
        per_fold.append(m)
        print(f"  [{universe}] fold {f.fold_id} [{f.val_start.date()}..{f.val_end.date()}] acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f} "
              f"ic={m['information_coefficient']:+.3f} logloss={m['log_loss']:.3f}", flush=True)
        selective_rows += [{"fold_id": f.fold_id, **r.__dict__} for r in coverage_precision_curve(val["y"].to_numpy(), proba, (0.0, 0.1, 0.2, 0.4))]
        v = val[["instrument_id", "event_time", "trend_regime", "vol_regime", "fwd_ret_1d", "y"]].assign(proba_up=proba, fold_id=f.fold_id)
        preds.append(v)
        for axis in ("trend_regime", "vol_regime"):
            for rv, g in v.groupby(axis):
                if len(g) >= 200 and g["y"].nunique() > 1:
                    mm = evaluate_binary_predictions(g["y"].to_numpy(), g["proba_up"].to_numpy(), fold_id=f.fold_id)
                    regime_rows.append({"fold_id": f.fold_id, "axis": axis, "regime": rv, "n": len(g), "accuracy": mm.accuracy, "roc_auc": mm.roc_auc})
    allp = pd.concat(preds, ignore_index=True)
    allp["position"] = signal_from_proba(allp["proba_up"].to_numpy(), "linear")
    cost = cost_stress_table(allp, "position", "fwd_ret_1d", BASE_COST)
    print(cost.to_string(index=False), flush=True)
    agg = pd.DataFrame(per_fold)[["accuracy", "balanced_accuracy", "log_loss", "brier_score", "roc_auc", "information_coefficient"]].agg(["mean", "std"]).round(4)
    return {"universe": universe, "n_rows": int(len(df)), "n_instruments": int(df["instrument_id"].nunique()), "n_features": len(cols),
            "per_fold": per_fold, "aggregate": json.loads(agg.to_json()), "selective": selective_rows,
            "regime_segmented": regime_rows, "cost_stress": cost.to_dict("records")}


def run(prebuilt=None) -> dict:
    print("Building features for the survivorship-free pool...", flush=True)
    feats, cols, layers = prebuilt or build_features()
    print("Layers:", json.dumps(layers), flush=True)
    results = {}
    for universe in ("pit", "biased"):
        df, ucols = build_universe_dataset(feats, cols, universe)
        print(f"\n=== {universe.upper()} universe: {len(df)} rows, {df['instrument_id'].nunique()} names, {len(ucols)} features ===", flush=True)
        results[universe] = evaluate(df, ucols, universe)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "layers": layers,
              "config": {"horizon": HORIZON, "n_splits": N_SPLITS, "embargo_days": EMBARGO_DAYS, "seed": SEED, "cost_bps": BASE_COST.cost_bps, "slippage_bps": BASE_COST.slippage_bps},
              "results": results}
    (REPORTS_DIR / "phase2_enriched_metrics.json").write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report)
    return report


def write_markdown(report: dict) -> None:
    L = ["# Phase 2 enriched pipeline -- locked metrics", "", f"Generated: {report['generated_at_utc']}", "",
         "## Layers", "", *[f"- {k}: {v}" for k, v in report["layers"].items()], ""]
    for u, r in report["results"].items():
        a = r["aggregate"]
        L += [f"## {u.upper()} universe ({r['n_rows']} rows, {r['n_instruments']} names, {r['n_features']} features)", "",
              "| metric | mean | std |", "|---|---|---|",
              *[f"| {k} | {a[k]['mean']:.4f} | {a[k]['std']:.4f} |" for k in ("accuracy", "balanced_accuracy", "log_loss", "brier_score", "roc_auc", "information_coefficient")], "",
              "Cost-stressed backtest (pooled out-of-sample, linear sizing, equal weight):", "",
              "| stress | cost+slip bps | Sharpe | total return | max DD | turnover |", "|---|---|---|---|---|---|",
              *[f"| {c['cost_stress_multiplier']}x | {c['cost_bps'] + c['slippage_bps']:.0f} | {c['annualised_sharpe']:.3f} | {c['total_return']:+.1%} | {c['max_drawdown']:.1%} | {c['mean_daily_turnover']:.3f} |" for c in r["cost_stress"]], ""]
    L += ["## Reading guide", "", "PIT = real point-in-time Nifty 50 membership (headline); biased = today's 50 projected backward. "
          "Accuracy/AUC/IC near 0.5/0.5/0 mean no detectable edge."]
    (REPORTS_DIR / "phase2_enriched_metrics.md").write_text("\n".join(L))


if __name__ == "__main__":
    run()
