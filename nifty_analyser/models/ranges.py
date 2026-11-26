"""Expected-move ranges: the h-day log return divided by a volatility forecast has a roughly stable distribution,
so its past quantiles (only outcomes already realised) turn a forecast into a calibrated range."""
import numpy as np

LEVELS = (0.05, 0.25, 0.75, 0.95)


def fit_quantiles(returns, scale, levels=LEVELS) -> dict[float, float]:
    z = np.asarray(returns, float) / np.asarray(scale, float)
    z = z[np.isfinite(z)]
    return {q: float(np.quantile(z, q)) for q in levels}


def walk_forward_ranges(dates, returns, scale, horizon: int, min_dates: int = 250, levels=LEVELS) -> np.ndarray:
    """One row of return quantiles per input row, from rows whose outcome ended before that date; NaN during warm-up."""
    uniq = np.unique(dates)
    z = np.asarray(returns, float) / np.asarray(scale, float)
    out = np.full((len(z), len(levels)), np.nan)
    for i in range(min_dates, len(uniq)):
        past = z[(dates < uniq[i - horizon]) & np.isfinite(z)]
        now = dates == uniq[i]
        out[now] = np.outer(np.asarray(scale, float)[now], np.quantile(past, levels))
    return out


def pinball(returns, quantile, level: float) -> float:
    diff = np.asarray(returns) - np.asarray(quantile)
    return float(np.mean(np.maximum(level * diff, (level - 1) * diff)))


def adaptive_ranges(dates, returns, scale, horizon: int, min_dates: int = 250, levels=LEVELS, gamma: float = 0.005):
    """Adaptive conformal version: each tail level moves with the miss rate of ranges whose outcome has since been realised.
    Returns (ranges per row, quantile multipliers to apply to a forecast made after the last date)."""
    uniq = np.unique(dates)
    returns, scale = np.asarray(returns, float), np.asarray(scale, float)
    z = returns / scale
    target = np.asarray(levels, float)
    level = target.copy()
    out = np.full((len(z), len(levels)), np.nan)
    for i in range(min_dates, len(uniq)):
        j = i - horizon - 1
        done = dates == uniq[j]
        if j >= min_dates:
            below = np.mean(returns[done][:, None] < out[done], axis=0)
            level = np.clip(level + gamma * (target - below), 0.002, 0.998)
        now = dates == uniq[i]
        past = z[(dates < uniq[i - horizon]) & np.isfinite(z)]
        out[now] = np.outer(scale[now], np.quantile(past, level))
    final = np.quantile(z[np.isfinite(z)], level)
    return out, {q: float(v) for q, v in zip(levels, final)}


def walk_forward_recalibrate(dates, prob, outcome, horizon: int, min_dates: int = 250, every: int = 20) -> np.ndarray:
    """Platt recalibration refit every `every` dates on predictions whose outcome is already realised; NaN during warm-up."""
    from sklearn.linear_model import LogisticRegression

    uniq = np.unique(dates)
    logit = np.log(np.clip(prob, 1e-4, 1 - 1e-4) / (1 - np.clip(prob, 1e-4, 1 - 1e-4)))[:, None]
    out = np.full(len(prob), np.nan)
    for i in range(min_dates, len(uniq), every):
        past = dates < uniq[i - horizon]
        model = LogisticRegression(C=1e6).fit(logit[past], outcome[past])
        now = (dates >= uniq[i]) & (dates <= uniq[min(i + every, len(uniq)) - 1])
        out[now] = model.predict_proba(logit[now])[:, 1]
    return out
