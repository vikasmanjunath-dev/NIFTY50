"""Cost-aware backtest (Blueprint §3.9): turnover-proportional blended cost+slippage with a
1x/2x/5x stress table. Deliberately simplified (single blended bps figure, no market impact)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostAssumption:
    cost_bps: float = 10.0
    slippage_bps: float = 10.0

    def scaled(self, k: float) -> "CostAssumption":
        return CostAssumption(self.cost_bps * k, self.slippage_bps * k)

    @property
    def total_bps(self) -> float:
        return self.cost_bps + self.slippage_bps


def signal_from_proba(proba, method: str = "linear") -> np.ndarray:
    """Probability -> position in [-1, 1]: 'linear' = 2p-1, 'sign' = +/-1."""
    p = np.asarray(proba, dtype=float)
    return 2 * p - 1 if method == "linear" else np.sign(p - 0.5)


def portfolio_returns(preds: pd.DataFrame, position_col: str, ret_col: str, cost: CostAssumption) -> pd.DataFrame:
    """Daily equal-weight net returns. Turnover = mean |position change| per instrument-day;
    the first day's position is charged as a full entry."""
    pos = preds.pivot_table(index="event_time", columns="instrument_id", values=position_col)
    ret = preds.pivot_table(index="event_time", columns="instrument_id", values=ret_col)
    pos, ret = pos.sort_index(), ret.reindex_like(pos)
    gross = (pos * ret).mean(axis=1)
    turnover = pos.fillna(0).diff().abs().mean(axis=1)
    turnover.iloc[0] = pos.fillna(0).iloc[0].abs().mean()
    net = gross - turnover * cost.total_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover})


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else float("nan")


def cost_stress_table(preds, position_col, ret_col, base: CostAssumption, stress_multipliers=(1.0, 2.0, 5.0)) -> pd.DataFrame:
    rows = []
    for k in stress_multipliers:
        c = base.scaled(k)
        r = portfolio_returns(preds, position_col, ret_col, c)
        equity = (1 + r["net"]).cumprod()
        rows.append({"cost_stress_multiplier": k, "cost_bps": c.cost_bps, "slippage_bps": c.slippage_bps,
                     "annualised_sharpe": _sharpe(r["net"]), "total_return": float(equity.iloc[-1] - 1),
                     "max_drawdown": float((equity / equity.cummax() - 1).min()),
                     "mean_daily_turnover": float(r["turnover"].mean())})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ImpactCostModel:
    """Fuller execution-cost model (Blueprint §3.9): per name and day, cost (bps of traded value) =
    fixed fees/taxes + half the bid-ask spread + square-root market impact  Y * sigma_daily * sqrt(participation),
    participation = traded value / trailing average daily traded value (capped at 100%; unknown liquidity is charged
    as 100% -- conservative, never optimistic). Capital is split across the day's active names by |position|."""
    fixed_bps: float = 10.0
    half_spread_bps: float = 5.0
    impact_coef: float = 0.5
    capital: float = 1e8                      # Rs 10 crore

    def scaled(self, k: float) -> "ImpactCostModel":
        return ImpactCostModel(self.fixed_bps * k, self.half_spread_bps * k, self.impact_coef * k, self.capital)


def portfolio_returns_impact(preds: pd.DataFrame, position_col: str, ret_col: str, liq: pd.DataFrame, model: ImpactCostModel) -> pd.DataFrame:
    """``liq``: instrument_id, event_time, adv_value, sigma (see eval.dataset.liquidity_panel)."""
    lcols = ["adv_value", "sigma"]
    d = preds.drop(columns=[c for c in lcols if c in preds.columns]).merge(
        liq[["instrument_id", "event_time", *lcols]], on=["instrument_id", "event_time"], how="left")
    pos = d.pivot_table(index="event_time", columns="instrument_id", values=position_col).sort_index()
    ret = d.pivot_table(index="event_time", columns="instrument_id", values=ret_col).reindex_like(pos)
    adv = d.pivot_table(index="event_time", columns="instrument_id", values="adv_value").reindex_like(pos)
    sig = d.pivot_table(index="event_time", columns="instrument_id", values="sigma").reindex_like(pos)
    w = pos.div(pos.notna().sum(axis=1), axis=0).fillna(0.0)
    dw = w.diff().abs()
    dw.iloc[0] = w.iloc[0].abs()
    part = (dw * model.capital / adv.where(adv > 0)).clip(upper=1.0).fillna(1.0)
    sig_f = sig.fillna(sig.stack().median() if sig.notna().any().any() else 0.02)
    cost_bps = model.fixed_bps + model.half_spread_bps + model.impact_coef * sig_f * np.sqrt(part) * 1e4
    gross = (w * ret.fillna(0.0)).sum(axis=1)
    cost = (dw * cost_bps / 1e4).sum(axis=1)
    return pd.DataFrame({"gross": gross, "net": gross - cost, "turnover": dw.sum(axis=1), "cost": cost})
