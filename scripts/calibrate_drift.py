"""Calibrate drift-alert thresholds to this model's own history and save them next to the model.
Run after scripts/build_serving_artifacts.py:  python3 scripts/calibrate_drift.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.monitoring.drift import calibrate_drift_thresholds
from nifty_analyser.ops.registry import load_model


def run() -> dict:
    _, meta = load_model(DATA_PROCESSED_DIR / "models")
    f = pd.read_parquet(DATA_PROCESSED_DIR / "serving_features.parquet")
    for c in ("event_time", "available_time"):
        f[c] = pd.to_datetime(f[c], utc=True).dt.tz_convert("Asia/Kolkata")
    cal = {**calibrate_drift_thresholds(f, meta["feature_cols"]), "model_version": meta["version"]}
    (DATA_PROCESSED_DIR / "models" / "drift_calibration.json").write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))
    return cal


if __name__ == "__main__":
    run()
