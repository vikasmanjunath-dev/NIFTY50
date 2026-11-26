"""Fixed-hyperparameter baseline models (Blueprint §3.5) -- deliberately NO tuning (§3.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedModel:
    name: str
    model: object

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def make_lightgbm_baseline(random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        random_state=random_state, n_jobs=4, verbose=-1,
    )


def fit_lightgbm(X_train, y_train, random_state: int = 42) -> TrainedModel:
    m = make_lightgbm_baseline(random_state)
    m.fit(X_train, y_train)
    return TrainedModel("lightgbm", m)


def fit_logistic(X_train, y_train, random_state: int = 42) -> TrainedModel:
    m = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=0.05, l1_ratio=0.5, solver="saga", max_iter=300, random_state=random_state),
    )
    m.fit(X_train, y_train)
    return TrainedModel("logistic_elasticnet", m)


def fit_catboost(X_train, y_train, random_state: int = 42) -> TrainedModel:
    from catboost import CatBoostClassifier

    m = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=10.0,
                           random_seed=random_state, verbose=0, thread_count=4)
    m.fit(X_train, y_train)
    return TrainedModel("catboost", m)


BASE_MODEL_FITTERS = {"logistic_elasticnet": fit_logistic, "lightgbm": fit_lightgbm, "catboost": fit_catboost}
