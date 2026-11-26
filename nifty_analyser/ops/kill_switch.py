"""Kill switch + live health (Blueprint §5.5). Auto rules turn data/feature health into
GREEN / AMBER / RED; a manual trip is persistent and only a deliberate reset clears it
(automatic recovery never un-trips a human decision)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from nifty_analyser.monitoring.drift import scale_free_drift_table

GREEN, AMBER, RED = "GREEN", "AMBER", "RED"
_RANK = {GREEN: 0, AMBER: 1, RED: 2}


@dataclass(frozen=True)
class Thresholds:
    stale_amber_days: float = 4.0     # NSE holiday clusters can span ~4 days
    stale_red_days: float = 7.0
    psi_frac_amber: float = 0.15      # share of features with PSI >= 0.25
    psi_frac_red: float = 0.40
    min_features_checked: int = 10

    @classmethod
    def calibrated(cls, cal: dict, **kw) -> "Thresholds":
        """Drift alert levels learned from the model's own history (``calibrate_drift_thresholds``)."""
        return cls(psi_frac_amber=cal["psi_frac_amber"], psi_frac_red=cal["psi_frac_red"], **kw)


@dataclass
class Health:
    state: str
    reasons: list[str] = field(default_factory=list)
    data_age_days: float | None = None
    psi_significant_frac: float | None = None

    def to_dict(self) -> dict:
        return {"state": self.state, "reasons": self.reasons, "data_age_days": self.data_age_days,
                "psi_significant_frac": self.psi_significant_frac}


def evaluate_health(data_age_days: float | None, psi_table: pd.DataFrame | None, th: Thresholds = Thresholds()) -> Health:
    state, reasons = GREEN, []

    def raise_to(level: str, why: str):
        nonlocal state
        reasons.append(why)
        if _RANK[level] > _RANK[state]:
            state = level

    if data_age_days is None:
        raise_to(RED, "no data available to assess freshness")
    elif data_age_days >= th.stale_red_days:
        raise_to(RED, f"data is {data_age_days:.1f} days old (>= {th.stale_red_days})")
    elif data_age_days >= th.stale_amber_days:
        raise_to(AMBER, f"data is {data_age_days:.1f} days old (>= {th.stale_amber_days})")
    frac = None
    if psi_table is None or len(psi_table) < th.min_features_checked:
        raise_to(AMBER, "feature drift could not be assessed (too few features checked)")
    else:
        frac = float((psi_table["psi"] >= 0.25).mean())
        if frac >= th.psi_frac_red:
            raise_to(RED, f"{frac:.0%} of features show significant drift (>= {th.psi_frac_red:.0%})")
        elif frac >= th.psi_frac_amber:
            raise_to(AMBER, f"{frac:.0%} of features show significant drift (>= {th.psi_frac_amber:.0%})")
    return Health(state, reasons, data_age_days, frac)


def compute_live_health(features: pd.DataFrame, cols: list[str], trained_through: pd.Timestamp, now: pd.Timestamp,
                        th: Thresholds = Thresholds(), **drift_kw) -> tuple[Health, pd.DataFrame | None]:
    """Freshness of the newest feature row + scale-free PSI (see ``scale_free_drift_table``). Only features
    that were actually assessable count toward the drift share; too few assessable -> AMBER, never "healthy"."""
    if features.empty:
        return evaluate_health(None, None, th), None
    age = (now - features["available_time"].max()).total_seconds() / 86400
    table = scale_free_drift_table(features, cols, trained_through, now, **drift_kw)
    assessed = table[table["psi"].notna()]
    return evaluate_health(age, assessed if len(assessed) else None, th), table


class KillSwitch:
    """Persistent manual override on top of automatic health."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {"tripped": False}

    def trip(self, reason: str, by: str) -> dict:
        state = {"tripped": True, "reason": reason, "by": by}
        self.path.write_text(json.dumps(state))
        return state

    def reset(self, by: str) -> dict:
        state = {"tripped": False, "reset_by": by}
        self.path.write_text(json.dumps(state))
        return state

    def manual(self) -> dict:
        return self._load()

    def effective(self, auto: Health) -> Health:
        m = self._load()
        if m.get("tripped"):
            return Health(RED, [f"manual kill switch: {m.get('reason', '')} (by {m.get('by', '?')})", *auto.reasons],
                          auto.data_age_days, auto.psi_significant_frac)
        return auto
