"""Volatility models: a linear HAR benchmark, shallow gradient boosting and their equal-weight blend."""
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_TREE = dict(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=1000, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, random_state=42, n_jobs=4, verbose=-1)


def _linear(estimator):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def fit_predict_classifiers(X_train, y_train, X_val) -> dict[str, np.ndarray]:
    """P(next-horizon volatility above the stock's own median) per model."""
    logit = _linear(LogisticRegression(C=0.05, max_iter=1000)).fit(X_train, y_train).predict_proba(X_val)[:, 1]
    tree = LGBMClassifier(**_TREE).fit(X_train, y_train).predict_proba(X_val)[:, 1]
    return {"har_logit": logit, "lgbm": tree, "blend": (logit + tree) / 2}


def fit_predict_regressors(X_train, y_train, X_val) -> dict[str, np.ndarray]:
    """Log realised volatility per model."""
    ridge = _linear(Ridge(alpha=10.0)).fit(X_train, y_train).predict(X_val)
    tree = LGBMRegressor(**_TREE).fit(X_train, y_train).predict(X_val)
    return {"har_ridge": ridge, "lgbm": tree, "blend": (ridge + tree) / 2}
