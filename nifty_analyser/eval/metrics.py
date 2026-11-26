"""Evaluation metrics (Blueprint §3.1): accuracy, balanced accuracy, log loss, Brier, ROC-AUC,
and the Spearman Information Coefficient (implemented on ranks -- no scipy.stats dependency)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score


def spearman_ic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class BinaryMetrics:
    fold_id: int
    model_name: str
    n_train: int
    n_val: int
    accuracy: float
    balanced_accuracy: float
    log_loss: float
    brier_score: float
    roc_auc: float
    information_coefficient: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_binary_predictions(y_true, proba, fwd_return=None, fold_id: int = 0, model_name: str = "", n_train: int = 0) -> BinaryMetrics:
    y = np.asarray(y_true)
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    both = len(np.unique(y)) > 1
    ic = spearman_ic(p, np.asarray(fwd_return, dtype=float)) if fwd_return is not None else float("nan")
    return BinaryMetrics(
        fold_id, model_name, n_train, len(y),
        float(accuracy_score(y, p > 0.5)),
        float(balanced_accuracy_score(y, p > 0.5)) if both else float("nan"),
        float(log_loss(y, p, labels=[0, 1])),
        float(brier_score_loss(y, p)),
        float(roc_auc_score(y, p)) if both else float("nan"),
        ic,
    )
