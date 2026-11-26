"""Probability calibration (Blueprint §3.6): Platt (sigmoid) and isotonic, fit only on a
purged/embargoed calibration slice. Output is clipped to [0.01, 0.99]: isotonic regression can
emit exactly 0/1 "shelves" from one-class-only calibration tails (observed on real data: 703
literal-certainty predictions, 303 wrong, log loss 1.87 > coin-flip 0.693)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CLIP = (0.01, 0.99)


@dataclass
class Calibrator:
    method: str
    model: object

    def apply(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        p = self.model.predict(s) if self.method == "isotonic" else self.model.predict_proba(s.reshape(-1, 1))[:, 1]
        return np.clip(p, *CLIP)


def fit_calibrator(scores, y, method: str = "sigmoid") -> Calibrator:
    s, y = np.asarray(scores, dtype=float), np.asarray(y)
    if method == "sigmoid":
        return Calibrator(method, LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y))
    if method == "isotonic":
        return Calibrator(method, IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y))
    raise ValueError(f"unknown calibration method {method!r}")


def reliability_table(y, proba, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), frac_pos=("y", "mean")).reset_index()
    return t


def expected_calibration_error(y, proba, n_bins: int = 10) -> float:
    t = reliability_table(y, proba, n_bins)
    return float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["n"].sum())
