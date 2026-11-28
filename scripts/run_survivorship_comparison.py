"""Survivorship-bias-free vs. biased universe: same features, labels, folds, models (Blueprint §3.4).

Run: python3 scripts/run_survivorship_comparison.py   (needs scripts/backfill_bars.py first)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd

from nifty_analyser.config import REPORTS_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.metrics import evaluate_binary_predictions
from nifty_analyser.eval.survivorship import bias_diagnostics, daily_cross_sectional_ic, label_dataset, universe_frames
from nifty_analyser.features.build import build_feature_matrix, feature_columns
from nifty_analyser.models.baseline import fit_lightgbm, fit_logistic
from nifty_analyser.validation.walk_forward import assert_folds_are_leakage_free, purged_embargoed_walk_forward

HORIZON, N_SPLITS, EMBARGO_DAYS, SEED = 5, 4, 5, 42
OUTCOME = f"fwd_ret_{HORIZON}d_outcome_available_time"
FITTERS = {"lightgbm": fit_lightgbm, "logistic_elasticnet": fit_logistic}


def evaluate_universe(df: pd.DataFrame, cols: list[str], label: str) -> list[dict]:
    folds = purged_embargoed_walk_forward(df, N_SPLITS, OUTCOME, pd.Timedelta(days=EMBARGO_DAYS))
    assert_folds_are_leakage_free(df, folds, OUTCOME)
    X, y = df[cols].to_numpy(dtype=float), df["y"].to_numpy()
    rows = []
    for fold in folds:
        val = df.iloc[fold.val_idx]
        for name, fit in FITTERS.items():
            proba = fit(X[fold.train_idx], y[fold.train_idx], random_state=SEED).predict_proba_up(X[fold.val_idx])
            m = evaluate_binary_predictions(y[fold.val_idx], proba, fwd_return=val[f"fwd_ret_{HORIZON}d"].to_numpy(),
                                            fold_id=fold.fold_id, model_name=name, n_train=len(fold.train_idx)).to_dict()
            m["cross_sectional_ic"] = daily_cross_sectional_ic(val.assign(_p=proba), "_p", f"fwd_ret_{HORIZON}d")
            m["universe"] = label
            rows.append(m)
            print(f"  [{label}] fold {fold.fold_id} {name:20s} acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f} "
                  f"ic={m['information_coefficient']:+.3f} xs_ic={m['cross_sectional_ic']:+.3f}")
    return rows


def run() -> dict:
    print("Loading bars and building features for the full survivorship-free pool...")
    features = build_feature_matrix(load_universe_bars())
    cols = feature_columns(features)
    labelled = label_dataset(features, HORIZON)
    biased, pit = universe_frames(labelled)
    print(f"{len(cols)} features | biased rows={len(biased)} ({biased.instrument_id.nunique()} names) | "
          f"PIT rows={len(pit)} ({pit.instrument_id.nunique()} names)")

    diag = bias_diagnostics(biased, pit)
    print(json.dumps(diag, indent=2))

    results = evaluate_universe(biased, cols, "biased") + evaluate_universe(pit, cols, "pit")
    res = pd.DataFrame(results)
    summary = (res.groupby(["universe", "model_name"])[["accuracy", "roc_auc", "information_coefficient", "cross_sectional_ic", "log_loss"]]
               .agg(["mean", "std"]).round(4))
    print("\n=== Aggregate across folds ===\n", summary)

    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "config": {"horizon": HORIZON, "n_splits": N_SPLITS, "embargo_days": EMBARGO_DAYS, "seed": SEED, "n_features": len(cols)},
              "bias_diagnostics": diag, "per_fold": results, "summary": json.loads(summary.to_json())}
    (REPORTS_DIR / "survivorship_comparison.json").write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report, res)
    return report


def write_markdown(report: dict, res: pd.DataFrame) -> None:
    d = report["bias_diagnostics"]
    ph, om = d["phantom_rows (in biased, not real members)"], d["omitted_rows (real members missing from biased)"]
    agg = res.groupby(["universe", "model_name"])[["accuracy", "roc_auc", "information_coefficient", "cross_sectional_ic"]].mean().round(4)
    lines = ["# Survivorship-bias comparison (biased = today's 50 projected backward vs. PIT = real membership)", "",
             f"Generated: {report['generated_at_utc']}", "", "## Measured bias", "",
             f"- Phantom rows (stock before it joined the index, present only in the biased universe): {ph['n_rows']} rows, "
             f"{ph['n_instruments']} names, annualised 1-day-forward return **{ph['annualised_return']:+.1%}**",
             f"- Omitted rows (real members that later left, absent from the biased universe): {om['n_rows']} rows, "
             f"{om['n_instruments']} names, annualised return **{om['annualised_return']:+.1%}**",
             f"- Equal-weight biased: {d['equal_weight_biased']['annualised_return']:+.1%} (Sharpe {d['equal_weight_biased']['sharpe']:.2f}); "
             f"equal-weight PIT: {d['equal_weight_pit']['annualised_return']:+.1%} (Sharpe {d['equal_weight_pit']['sharpe']:.2f})",
             f"- **Survivorship premium: {d['survivorship_premium_annualised']:+.1%}/yr** (t = {d['survivorship_premium_tstat']:.2f})", "",
             "## Model metrics (mean across 4 purged/embargoed walk-forward folds, identical date blocks)", "",
             agg.to_markdown(), "",
             "Reading guide: IC = pooled Spearman of predicted P(up) vs realised 5d return; cross_sectional_ic = mean per-day rank IC."]
    (REPORTS_DIR / "survivorship_comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run()
