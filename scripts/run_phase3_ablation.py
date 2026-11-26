"""RAG feature ablation: does the event layer add incremental value? (Blueprint §4.10, §28)

"RAG does not, by itself, produce a higher directional hit rate... The
correct question is: does RAG provide incremental, point-in-time information
that improves an out-of-sample decision metric after costs?"

This is the test the whole Phase 3 build has been in service of. It reuses
the exact same walk-forward folds, the exact same LightGBM hyperparameters,
and the exact same evaluation metrics as the Phase 2 pipeline -- the ONLY
difference between the two runs below is whether the event/materiality
feature columns are included, so any measured difference is attributable to
those features and nothing else (a paired comparison, per the blueprint's
own ablation-table requirement of "paired dates so treatment and control see
exactly the same market observations").

Run with: python3 scripts/run_phase3_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.documents.extraction import add_materiality, compute_daily_event_features
from nifty_analyser.features.build import build_feature_matrix, feature_columns
from nifty_analyser.features.events import EVENT_FEATURE_COLUMNS, add_rolling_event_intensity, merge_events_into_features
from nifty_analyser.labels.targets import add_direction_label
from nifty_analyser.models.baseline import fit_lightgbm
from nifty_analyser.eval.metrics import evaluate_binary_predictions
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward
from scripts.run_phase3_pipeline import load_announcements

HORIZON = 5
N_SPLITS = 4
EMBARGO_DAYS = 5
RANDOM_STATE = 42
ROLLING_EVENT_COLS = ["announcement_count_7d", "announcement_count_30d", "high_materiality_count_7d", "high_materiality_count_30d"]


def build_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    bars = load_universe_bars()
    features = build_feature_matrix(bars)
    base_cols = feature_columns(features)

    documents = load_announcements()
    docs_with_materiality = add_materiality(documents)
    daily_events = compute_daily_event_features(docs_with_materiality)
    features = merge_events_into_features(features, daily_events)
    features = add_rolling_event_intensity(features, windows=(7, 30))

    event_cols = [*EVENT_FEATURE_COLUMNS, *ROLLING_EVENT_COLS]

    labelled_parts = [
        add_direction_label(g.sort_values("event_time"), horizon=HORIZON, flat_threshold=0.0)
        for _, g in features.groupby("instrument_id", sort=False)
    ]
    labelled = pd.concat(labelled_parts, ignore_index=True)

    label_col = f"dir_label_{HORIZON}d"
    outcome_col = f"fwd_ret_{HORIZON}d_outcome_available_time"
    labelled = labelled.dropna(subset=[label_col, outcome_col])
    labelled = labelled[labelled[label_col] != 0].copy()
    labelled["y"] = (labelled[label_col] == 1).astype(int)
    labelled = labelled.sort_values("event_time").reset_index(drop=True)
    labelled.attrs["outcome_col"] = outcome_col

    return labelled, base_cols, event_cols


def run_variant(df: pd.DataFrame, feat_cols: list[str], outcome_col: str, folds) -> list[dict]:
    X_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df["y"].to_numpy()

    results = []
    for fold in folds:
        X_train, y_train = X_all[fold.train_idx], y_all[fold.train_idx]
        X_val, y_val = X_all[fold.val_idx], y_all[fold.val_idx]

        model = fit_lightgbm(X_train, y_train, random_state=RANDOM_STATE)
        proba = model.predict_proba_up(X_val)
        fwd_ret_val = df.loc[fold.val_idx, f"fwd_ret_{HORIZON}d"].to_numpy(dtype=float)

        metrics = evaluate_binary_predictions(
            y_val, proba, fwd_return=fwd_ret_val, fold_id=fold.fold_id, model_name="lightgbm", n_train=len(fold.train_idx)
        )
        results.append(metrics.to_dict())
    return results


def run() -> None:
    print("Building dataset with RAG event features merged in...")
    df, base_cols, event_cols = build_dataset()
    outcome_col = df.attrs["outcome_col"]
    print(f"Dataset: {len(df)} rows, {len(base_cols)} base features, {len(event_cols)} event features")

    folds = purged_embargoed_walk_forward(
        df, n_splits=N_SPLITS, outcome_time_col=outcome_col, embargo=pd.Timedelta(days=EMBARGO_DAYS)
    )
    print(f"Generated {len(folds)} identical paired folds for both variants.\n")

    print("=== A1: baseline (market features only) ===")
    baseline_results = run_variant(df, base_cols, outcome_col, folds)
    for r in baseline_results:
        print(f"  fold {r['fold_id']}: acc={r['accuracy']:.3f} auc={r['roc_auc']:.3f} ic={r['information_coefficient']:+.3f}")

    print("\n=== A1 treatment: market + RAG event features ===")
    combined_cols = base_cols + event_cols
    treatment_results = run_variant(df, combined_cols, outcome_col, folds)
    for r in treatment_results:
        print(f"  fold {r['fold_id']}: acc={r['accuracy']:.3f} auc={r['roc_auc']:.3f} ic={r['information_coefficient']:+.3f}")

    baseline_df = pd.DataFrame(baseline_results)
    treatment_df = pd.DataFrame(treatment_results)

    print("\n=== Paired comparison (treatment - baseline, per fold) ===")
    for metric in ["accuracy", "roc_auc", "information_coefficient", "log_loss"]:
        diffs = treatment_df[metric].to_numpy() - baseline_df[metric].to_numpy()
        print(f"  {metric}: mean diff = {diffs.mean():+.4f}, per-fold diffs = {[round(d, 4) for d in diffs]}")

    print("\n=== Decision (per Blueprint §4.10 / §28: report incremental value, not a single number) ===")
    mean_ic_diff = (treatment_df["information_coefficient"] - baseline_df["information_coefficient"]).mean()
    if mean_ic_diff > 0.01:
        print(f"Event features show a positive mean IC improvement ({mean_ic_diff:+.4f}) -- candidate for promotion, "
              f"pending further out-of-sample confirmation and cost-aware backtesting.")
    else:
        print(f"Event features show no meaningful IC improvement ({mean_ic_diff:+.4f}) on this sample. "
              f"Per the blueprint's own operating rule: keep as an explanation/research-acceleration tool, "
              f"do not promote into the live trading signal on this evidence alone.")


if __name__ == "__main__":
    run()
