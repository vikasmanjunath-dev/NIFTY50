"""Build everything the /v3 service needs: (1) the point-in-time serving feature store (UNLABELLED, so the
newest day is present; membership-filtered; same-day ranks computed within the real index members),
(2) the final ensemble trained on all labelled point-in-time rows, registered with a checksum.
Run: python3 scripts/build_serving_artifacts.py   (several minutes: full feature build + ensemble fit)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nifty_analyser.config import DATA_PROCESSED_DIR, REPORTS_DIR
from nifty_analyser.data.index_membership import membership_mask
from nifty_analyser.features.cross_sectional import add_same_day_cross_sectional
from nifty_analyser.models.ensemble import fit_ensemble
from nifty_analyser.ops.registry import save_model
from scripts.run_phase2_pipeline import EMBARGO_DAYS, HORIZON, OUTCOME, SEED, build_features, build_universe_dataset

SERVING_FEATURES = DATA_PROCESSED_DIR / "serving_features.parquet"
MODEL_DIR = DATA_PROCESSED_DIR / "models"


def disclaimer_from_report() -> str:
    try:
        a = json.loads((REPORTS_DIR / "phase2_enriched_metrics.json").read_text())["results"]["pit"]["aggregate"]
        return (f"Walk-forward validation (point-in-time universe): balanced accuracy {a['balanced_accuracy']['mean']:.3f}, "
                f"ROC-AUC {a['roc_auc']['mean']:.3f}, IC {a['information_coefficient']['mean']:+.3f} -- no demonstrated predictive "
                f"edge. Informational output only; not investment advice.")
    except Exception:  # noqa: BLE001
        return "No validated predictive edge has been demonstrated. Informational output only; not investment advice."


def run(prebuilt=None) -> None:
    feats, cols, layers = prebuilt or build_features()
    df, ucols = build_universe_dataset(feats, cols, "pit")
    serving = add_same_day_cross_sectional(feats[membership_mask(feats)].reset_index(drop=True))
    keep = ["instrument_id", "event_time", "available_time", "trend_regime", "vol_regime", *ucols]
    serving[keep].to_parquet(SERVING_FEATURES, index=False)
    print(f"serving store: {len(serving)} rows, latest {serving['event_time'].max()}", flush=True)
    model = fit_ensemble(df, ucols, "y", np.arange(len(df)), OUTCOME, pd.Timedelta(days=EMBARGO_DAYS), random_state=SEED)
    meta = {"feature_cols": ucols, "trained_through": str(df["event_time"].max().date()), "layers": layers,
            "horizon_days": HORIZON, "n_training_rows": int(len(df)), "disclaimer": disclaimer_from_report()}
    print("registered model:", save_model(model, meta, MODEL_DIR), flush=True)


if __name__ == "__main__":
    run()
