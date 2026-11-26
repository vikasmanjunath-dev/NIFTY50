"""Learned market regime (Blueprint §3.8, the more complex of its two designs): a Gaussian mixture over a daily
market-state vector, fit ONLY on training dates (fold-wise) so it cannot see the future. Components are ordered by
mean volatility so labels are stable across refits. Whether it helps is an empirical question answered by paired ablation
(scripts/run_robustness.py), not assumed."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

STATE_COLS = ["ret_5d", "ret_20d", "rvol_20d", "drawdown_from_252d_high", "india_vix_percentile_252d", "india_vix_dev_252d"]


def market_state(df: pd.DataFrame, cols: list[str] = STATE_COLS) -> pd.DataFrame:
    """One row per date: cross-sectional mean of each state feature (market-wide by construction)."""
    use = [c for c in cols if c in df.columns]
    return df.groupby("event_time")[use].mean().sort_index()


class LearnedRegime:
    def __init__(self, k: int = 3, seed: int = 42):
        self.k, self.seed = k, seed

    def fit(self, state_train: pd.DataFrame) -> "LearnedRegime":
        self.cols_ = list(state_train.columns)
        self.mu_, self.sd_ = state_train.mean(), state_train.std().replace(0, 1.0)
        z = self._z(state_train)
        self.gmm_ = GaussianMixture(self.k, covariance_type="full", n_init=5, reg_covar=1e-3, random_state=self.seed).fit(z)
        vol_idx = self.cols_.index("rvol_20d") if "rvol_20d" in self.cols_ else 0
        self.order_ = np.argsort(self.gmm_.means_[:, vol_idx])          # component 0 = calmest ... k-1 = most volatile
        return self

    def _z(self, state: pd.DataFrame) -> np.ndarray:
        return ((state[self.cols_] - self.mu_) / self.sd_).fillna(0.0).to_numpy()

    def transform(self, state: pd.DataFrame) -> pd.DataFrame:
        p = self.gmm_.predict_proba(self._z(state))[:, self.order_]
        return pd.DataFrame(p, index=state.index, columns=[f"regime_p{i}" for i in range(self.k)])
