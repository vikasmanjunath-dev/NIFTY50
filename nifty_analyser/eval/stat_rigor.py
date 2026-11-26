"""Selection-aware statistics for a Sharpe ratio that was picked after many looks and has autocorrelated (overlapping-hold) returns.
- ``effective_n``: T / (1 + 2 sum rho_k) (Bartlett weights) - the number of independent days.
- ``deflated_sharpe``: Bailey & Lopez de Prado (2014) probability that the TRUE Sharpe exceeds the best Sharpe expected from
  ``n_trials`` skill-less trials, given non-normal returns and the effective sample length.
- ``bootstrap_sharpe_ci``: Politis-Romano stationary bootstrap CI (annualised), robust to the autocorrelation.
Per-period Sharpe is used internally; ``annualise`` converts for display."""
from __future__ import annotations

import numpy as np
from scipy import stats

EULER = 0.5772156649015329


def sharpe_per_period(r) -> float:
    r = np.asarray(r, float)
    return float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else float("nan")


def annualise(sr: float, periods: int = 252) -> float:
    return sr * np.sqrt(periods)


def effective_n(r, max_lag: int = 30) -> float:
    r = np.asarray(r, float); r = r - r.mean(); n = len(r); v = float(r @ r) / n
    if v == 0 or n < 3:
        return float(n)
    rho = [float(r[k:] @ r[:-k]) / n / v for k in range(1, min(max_lag, n - 1) + 1)]
    denom = 1.0 + 2.0 * sum((1 - (k + 1) / (len(rho) + 1)) * x for k, x in enumerate(rho))
    return float(n / max(denom, 1.0))


def expected_max_sharpe(n_trials: int, trial_sr_var: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` independent skill-less trials (Gumbel approximation)."""
    if n_trials <= 1:
        return 0.0
    return float(np.sqrt(trial_sr_var) * ((1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials) + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))))


def deflated_sharpe(r, n_trials: int, trial_sr_var: float | None = None, use_effective_n: bool = True) -> dict:
    r = np.asarray(r, float); sr = sharpe_per_period(r); t = effective_n(r) if use_effective_n else float(len(r))
    var = trial_sr_var if trial_sr_var is not None else 1.0 / max(t - 1, 1)       # default: sampling variance of a null Sharpe
    sr0 = expected_max_sharpe(n_trials, var)
    skew, kurt = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(max(t - 1, 1)) / denom
    return {"sharpe_ann": annualise(sr), "n_days": len(r), "n_eff": t, "n_trials": n_trials, "sr0_ann": annualise(sr0),
            "psr_vs_zero": float(stats.norm.cdf(sr * np.sqrt(max(t - 1, 1)) / denom)), "dsr": float(stats.norm.cdf(z))}


def bootstrap_sharpe_ci(r, n_boot: int = 2000, mean_block: float = 20.0, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(lo, hi, P[Sharpe<=0]) from the stationary bootstrap with geometric blocks of mean length ``mean_block``."""
    r = np.asarray(r, float); n = len(r); rng = np.random.default_rng(seed); p = 1.0 / mean_block; out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, int); i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        x = r[idx]; out[b] = annualise(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2)), float((out <= 0).mean())
