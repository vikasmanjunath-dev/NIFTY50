"""Calibrate ``ImpactCostModel`` from real execution records (the one input this project cannot fabricate).

Model per fill:  slippage_bps = half_spread_bps + impact_coef * sigma_daily * sqrt(participation) * 1e4 + noise,
where slippage is signed so that a positive number is a COST (buys: fill vs arrival price above; sells: below) and
participation = order value / trailing ADV. Explicit fees/taxes (STT, brokerage, exchange charges) are known, so they are
passed in as ``fee_bps`` and become ``fixed_bps`` rather than being estimated. Fit is Huber-robust OLS (fills have heavy
outliers) with a bootstrap CI; ``to_model`` returns a ready ``ImpactCostModel``. Needs >= 30 fills to fit at all.

Expected CSV columns (see ``FILL_COLUMNS``): ``instrument_id, event_time, side (+1 buy/-1 sell), arrival_price, fill_price,
order_value, adv_value, sigma_daily``. ``simulate_fills`` generates synthetic ones for tests/dry runs ONLY.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nifty_analyser.eval.costs import ImpactCostModel

FILL_COLUMNS = ["instrument_id", "event_time", "side", "arrival_price", "fill_price", "order_value", "adv_value", "sigma_daily"]
MIN_FILLS = 30


@dataclass(frozen=True)
class ImpactFit:
    half_spread_bps: float
    impact_coef: float
    ci_half_spread: tuple[float, float]
    ci_impact_coef: tuple[float, float]
    n: int
    r2: float

    def to_model(self, fee_bps: float, capital: float = 1e8) -> ImpactCostModel:
        return ImpactCostModel(fixed_bps=fee_bps, half_spread_bps=max(self.half_spread_bps, 0.0), impact_coef=max(self.impact_coef, 0.0), capital=capital)


def prepare(fills: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """-> (x = sigma*sqrt(participation)*1e4, y = signed slippage in bps). Rows with missing/invalid inputs are dropped."""
    missing = set(FILL_COLUMNS) - set(fills.columns)
    if missing:
        raise ValueError(f"fills missing columns: {sorted(missing)}")
    f = fills.dropna(subset=FILL_COLUMNS)
    f = f[(f.arrival_price > 0) & (f.fill_price > 0) & (f.adv_value > 0) & (f.order_value > 0) & f.side.isin([-1, 1])]
    y = f.side.to_numpy(float) * (f.fill_price.to_numpy(float) / f.arrival_price.to_numpy(float) - 1.0) * 1e4
    x = f.sigma_daily.to_numpy(float) * np.sqrt(np.clip(f.order_value.to_numpy(float) / f.adv_value.to_numpy(float), 0, 1)) * 1e4
    return x, y


def _huber_fit(x: np.ndarray, y: np.ndarray, iters: int = 30, k: float = 1.345) -> np.ndarray:
    A = np.column_stack([np.ones_like(x), x]); w = np.ones_like(x); beta = np.zeros(2)
    for _ in range(iters):
        W = np.sqrt(w)[:, None]
        beta = np.linalg.lstsq(A * W, y * W[:, 0], rcond=None)[0]
        r = y - A @ beta; s = max(np.median(np.abs(r - np.median(r))) / 0.6745, 1e-9)
        u = np.abs(r) / (k * s); w = np.where(u <= 1, 1.0, 1.0 / u)
    return beta


def fit_impact(fills: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> ImpactFit:
    x, y = prepare(fills)
    if len(x) < MIN_FILLS:
        raise ValueError(f"need >= {MIN_FILLS} valid fills to calibrate, got {len(x)}")
    beta = _huber_fit(x, y); rng = np.random.default_rng(seed)
    boots = np.array([_huber_fit(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))])
    resid = y - (beta[0] + beta[1] * x)
    r2 = 1.0 - float(np.sum(resid ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
    ci = lambda j: (float(np.percentile(boots[:, j], 2.5)), float(np.percentile(boots[:, j], 97.5)))
    return ImpactFit(float(beta[0]), float(beta[1]), ci(0), ci(1), len(x), r2)


def simulate_fills(n: int = 400, half_spread_bps: float = 4.0, impact_coef: float = 0.6, noise_bps: float = 15.0, outlier_frac: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Synthetic fills with a known truth (tests / dry runs only - never a substitute for real executions)."""
    rng = np.random.default_rng(seed)
    adv = rng.uniform(5e8, 5e9, n); order = adv * rng.uniform(0.0005, 0.05, n); sigma = rng.uniform(0.008, 0.03, n); side = rng.choice([-1, 1], n)
    slip = half_spread_bps + impact_coef * sigma * np.sqrt(order / adv) * 1e4 + rng.normal(0, noise_bps, n)
    slip = np.where(rng.random(n) < outlier_frac, slip + rng.normal(0, 200, n), slip)
    arrival = rng.uniform(100, 3000, n)
    return pd.DataFrame({"instrument_id": "NSE:SIM", "event_time": pd.date_range("2026-01-01", periods=n, freq="h"), "side": side, "arrival_price": arrival,
                         "fill_price": arrival * (1 + side * slip / 1e4), "order_value": order, "adv_value": adv, "sigma_daily": sigma})
