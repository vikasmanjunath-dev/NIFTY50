"""Champion/challenger stack (Blueprint §3.5): base models -> out-of-fold meta-model -> calibrator.

Three mutually purged/embargoed slices carved recursively from one fold's training window
(fit -> meta-fit -> calibration): no stage is fit on data an earlier stage saw, and none
overlaps the validation fold. Never stack in-sample predictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nifty_analyser.models.baseline import BASE_MODEL_FITTERS, TrainedModel
from nifty_analyser.models.calibration import Calibrator, fit_calibrator
from nifty_analyser.validation.walk_forward import carve_calibration_slice

DEFAULT_BASE_MODELS = ("logistic_elasticnet", "lightgbm", "catboost")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass
class EnsembleModel:
    base_models: dict[str, TrainedModel]
    meta_model: LogisticRegression
    calibrator: Calibrator
    base_model_names: tuple[str, ...]
    slices: dict[str, np.ndarray]

    def _meta_features(self, X: np.ndarray) -> np.ndarray:
        return _logit(np.column_stack([self.base_models[n].predict_proba_up(X) for n in self.base_model_names]))

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.apply(self.meta_model.predict_proba(self._meta_features(X))[:, 1])


def fit_ensemble(
    df: pd.DataFrame, feature_cols: list[str], label_col: str, train_idx: np.ndarray,
    outcome_time_col: str, embargo: pd.Timedelta, base_models=DEFAULT_BASE_MODELS,
    calibration: str = "sigmoid", random_state: int = 42,
) -> EnsembleModel:
    fit_idx, rest = carve_calibration_slice(df, train_idx, 0.45, outcome_time_col, embargo)
    meta_idx, calib_idx = carve_calibration_slice(df, rest, 0.5, outcome_time_col, embargo)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy()
    base = {n: BASE_MODEL_FITTERS[n](X[fit_idx], y[fit_idx], random_state=random_state) for n in base_models}
    model = EnsembleModel(base, LogisticRegression(C=1.0), None, tuple(base_models),  # type: ignore[arg-type]
                          {"fit": fit_idx, "meta": meta_idx, "calibration": calib_idx})
    model.meta_model.fit(model._meta_features(X[meta_idx]), y[meta_idx])
    raw = model.meta_model.predict_proba(model._meta_features(X[calib_idx]))[:, 1]
    model.calibrator = fit_calibrator(raw, y[calib_idx], method=calibration)
    return model


@dataclass
class BlendModel:
    """Equal-weight logit average of base models, each fit on the FULL training window (no meta/calibration slices, so no
    training data is given up; scores are a ranking signal, not calibrated probabilities)."""
    base_models: dict[str, TrainedModel]

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        z = np.mean([_logit(m.predict_proba_up(X)) for m in self.base_models.values()], axis=0)
        return 1.0 / (1.0 + np.exp(-z))


def fit_blend(X_train: np.ndarray, y_train: np.ndarray, base_models=("lightgbm", "catboost"), random_state: int = 42) -> BlendModel:
    return BlendModel({n: BASE_MODEL_FITTERS[n](X_train, y_train, random_state=random_state) for n in base_models})
