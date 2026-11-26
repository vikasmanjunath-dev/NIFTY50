"""Selective prediction / abstention (Blueprint §3.6-3.7): trade only when confident."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def confidence(proba) -> np.ndarray:
    """Distance from maximal ambiguity, scaled to [0, 1]: |p - 0.5| * 2."""
    return np.abs(np.asarray(proba, dtype=float) - 0.5) * 2.0


def abstain_mask(proba, threshold: float) -> np.ndarray:
    return confidence(proba) < threshold


@dataclass(frozen=True)
class CoverageRow:
    threshold: float
    coverage: float
    n_acted: int
    accuracy_on_acted: float


def coverage_precision_curve(y_true, proba, thresholds=(0.0, 0.1, 0.2, 0.4)) -> list[CoverageRow]:
    y, p = np.asarray(y_true), np.asarray(proba, dtype=float)
    pred = (p > 0.5).astype(int)
    rows = []
    for t in thresholds:
        act = ~abstain_mask(p, t)
        n = int(act.sum())
        rows.append(CoverageRow(float(t), n / len(y), n, float((pred[act] == y[act]).mean()) if n else float("nan")))
    return rows


def threshold_for_coverage(proba, target_coverage: float) -> float:
    """Smallest confidence threshold whose coverage is <= target (quantile of confidence)."""
    return float(np.quantile(confidence(proba), 1.0 - target_coverage))
