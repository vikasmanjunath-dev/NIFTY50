"""Nested hyperparameter search (Blueprint §3.4/§5.2): tuning happens ONLY inside each outer training window, on inner
purged walk-forward folds; the outer fold never influences configuration choice. Reports the tuned-vs-default paired
difference and the selection optimism (best inner score minus honest outer score) -- the bias tuning would otherwise hide."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from nifty_analyser.models.baseline import TrainedModel, make_lightgbm_baseline
from nifty_analyser.validation.walk_forward import assert_folds_are_leakage_free, purged_embargoed_walk_forward

LGBM_SPACE = {
    "n_estimators": ("int", 50, 400), "learning_rate": ("log", 0.005, 0.1), "num_leaves": ("int", 4, 64),
    "min_child_samples": ("log_int", 20, 1000), "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.3, 1.0),
    "reg_lambda": ("log", 0.1, 50.0),
}


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    out = {}
    for k, (kind, lo, hi) in space.items():
        if kind == "int":
            out[k] = int(rng.integers(lo, hi + 1))
        elif kind == "float":
            out[k] = float(rng.uniform(lo, hi))
        elif kind == "log":
            out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:  # log_int
            out[k] = int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
    return out


def _fit(params: dict | None, X, y, seed: int) -> TrainedModel:
    m = make_lightgbm_baseline(seed) if params is None else LGBMClassifier(
        **params, subsample_freq=1, random_state=seed, n_jobs=4, verbose=-1)
    m.fit(X, y)
    return TrainedModel("lightgbm", m)


def _auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5


def inner_score(sub: pd.DataFrame, cols: list[str], params: dict | None, outcome_col: str, embargo: pd.Timedelta,
                inner_splits: int, seed: int) -> float:
    folds = purged_embargoed_walk_forward(sub, inner_splits, outcome_col, embargo)
    X, y = sub[cols].to_numpy(float), sub["y"].to_numpy()
    return float(np.mean([_auc(y[f.val_idx], _fit(params, X[f.train_idx], y[f.train_idx], seed).predict_proba_up(X[f.val_idx])) for f in folds]))


@dataclass
class NestedResult:
    rows: pd.DataFrame           # per outer fold: default_auc, tuned_auc, inner_best, inner_default, best_params

    def summary(self) -> dict:
        r = self.rows
        d = r["tuned_auc"] - r["default_auc"]
        return {"n_outer": len(r), "default_auc": float(r["default_auc"].mean()), "tuned_auc": float(r["tuned_auc"].mean()),
                "tuned_minus_default": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                "folds_tuned_better": int((d > 0).sum()),
                "selection_optimism": float((r["inner_best"] - r["tuned_auc"]).mean())}


def nested_random_search(df: pd.DataFrame, cols: list[str], outcome_col: str, embargo: pd.Timedelta, space: dict = LGBM_SPACE,
                         outer_splits: int = 5, inner_splits: int = 3, n_trials: int = 12, seed: int = 42) -> NestedResult:
    outer = purged_embargoed_walk_forward(df, outer_splits, outcome_col, embargo)
    assert_folds_are_leakage_free(df, outer, outcome_col)
    X, y = df[cols].to_numpy(float), df["y"].to_numpy()
    rows = []
    for f in outer:
        sub = df.iloc[f.train_idx].reset_index(drop=True)          # tuning sees ONLY the outer training window
        rng = np.random.default_rng(seed + f.fold_id)
        trials = [sample_params(space, rng) for _ in range(n_trials)]
        scores = [inner_score(sub, cols, t, outcome_col, embargo, inner_splits, seed) for t in trials]
        best = trials[int(np.argmax(scores))]
        base_inner = inner_score(sub, cols, None, outcome_col, embargo, inner_splits, seed)
        Xv, yv = X[f.val_idx], y[f.val_idx]
        rows.append({"fold": f.fold_id, "default_auc": _auc(yv, _fit(None, X[f.train_idx], y[f.train_idx], seed).predict_proba_up(Xv)),
                     "tuned_auc": _auc(yv, _fit(best, X[f.train_idx], y[f.train_idx], seed).predict_proba_up(Xv)),
                     "inner_best": float(max(scores)), "inner_default": base_inner, "best_params": best})
    return NestedResult(pd.DataFrame(rows))
