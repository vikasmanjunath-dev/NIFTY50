"""Paired, identical-fold ablation (Blueprint §4.10/§28): does adding fundamentals + FII/DII flow
improve out-of-sample prediction? One dataset, one set of purged/embargoed folds, a single LightGBM
fit with vs. without those columns -- the only difference is the feature set.
Run: python3 scripts/run_phase2_fundamentals_fii_dii_ablation.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.eval.metrics import evaluate_binary_predictions
from nifty_analyser.features.fii_dii_features import flow_feature_columns
from nifty_analyser.models.baseline import fit_lightgbm
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward
from scripts.run_phase2_pipeline import EMBARGO_DAYS, HORIZON, N_SPLITS, OUTCOME, SEED, build_features, build_universe_dataset

NEW = ["profit_margin", "eps_growth_yoy", "revenue_growth_yoy", "profit_growth_yoy", "earnings_yield", *flow_feature_columns("fii"), *flow_feature_columns("dii")]


def run_variant(df, cols, folds):
    X, y = df[cols].to_numpy(dtype=float), df["y"].to_numpy()
    out = []
    for f in folds:
        p = fit_lightgbm(X[f.train_idx], y[f.train_idx], random_state=SEED).predict_proba_up(X[f.val_idx])
        out.append(evaluate_binary_predictions(y[f.val_idx], p, df[f"fwd_ret_{HORIZON}d"].to_numpy()[f.val_idx], f.fold_id, "lightgbm", len(f.train_idx)).to_dict())
    return pd.DataFrame(out)


def run() -> None:
    feats, cols, layers = build_features()
    print("Layers:", layers)
    df, all_cols = build_universe_dataset(feats, cols, "pit")
    new = [c for c in NEW if c in all_cols]
    base = [c for c in all_cols if c not in new]
    print(f"{len(df)} rows | {len(base)} base features | {len(new)} fundamentals/FII-DII features: {new}")
    folds = purged_embargoed_walk_forward(df, N_SPLITS, OUTCOME, pd.Timedelta(days=EMBARGO_DAYS))
    b, t = run_variant(df, base, folds), run_variant(df, base + new, folds)
    for name, r in (("baseline", b), ("treatment", t)):
        print(f"{name}:", [f"ic={x:+.3f}" for x in r["information_coefficient"]])
    for m in ("accuracy", "roc_auc", "information_coefficient", "log_loss"):
        d = t[m].to_numpy() - b[m].to_numpy()
        print(f"  {m}: mean diff {d.mean():+.4f}  per-fold {[round(x, 4) for x in d]}")
    ic = (t["information_coefficient"] - b["information_coefficient"]).mean()
    print("DECISION:", "candidate for promotion (confirm out-of-sample + after costs)" if ic > 0.01 else
          f"no meaningful IC improvement ({ic:+.4f}); not evidence the features add value")


if __name__ == "__main__":
    run()
