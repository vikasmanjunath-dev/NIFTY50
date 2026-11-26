"""Point-in-time Nifty 50 index membership (Blueprint §3.4 survivorship-bias fix).

Backtesting only *today's* constituents silently drops stocks that were removed
for underperforming and includes stocks only because they later performed well
enough to join. This module reconstructs the real membership on any date by
starting from the known current 50 and reverse-applying every sourced
replacement since the project's data start (2021-09-20).

Source: Wikipedia's "NIFTY 50" article -- fully-cited "List of replacements
since 2005" (NSE circulars + contemporaneous news), cross-checked against the
same article's per-company "Date added" column. NSE/NSE Indices publish no
clean structured history (checked directly). ``announced_date`` is the real
publish date of the cited news item; the one marked estimate uses NSE's
documented 4-week notice period.

``members_as_of`` uses the EFFECTIVE date (what determines whether a stock was
really in the index on a trading day). ``announced_date`` is kept as the
point-in-time ``available_time`` of the change for any future feature built
on "about to join/leave" information.

Honest limitation: HDFC (member until its 13 Jul 2023 merger) has no data left
in yfinance, so its seat is correctly *reported* as a member but cannot have
bars -- that window is 49/50 in practice, not silently patched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nifty_analyser.data.instrument_master import CURRENT_SYMBOLS
from nifty_analyser.data.timestamps import to_ist

# No replacement is recorded after 30 Sep 2025, so the current 50 is valid
# for any date from then through the present.
ANCHOR_DATE = pd.Timestamp("2025-09-30")
ANCHOR_MEMBERS: frozenset[str] = CURRENT_SYMBOLS


@dataclass(frozen=True)
class ReplacementEvent:
    effective_date: pd.Timestamp
    symbol_out: str
    symbol_in: str
    announced_date: pd.Timestamp
    announced_is_estimate: bool
    reason: str


def _e(eff, out, inn, ann, est=False, reason="Inadequate market capitalization"):
    return ReplacementEvent(pd.Timestamp(eff), out, inn, pd.Timestamp(ann), est, reason)


REPLACEMENT_EVENTS: list[ReplacementEvent] = [
    _e("2022-03-31", "IOC", "APOLLOHOSP", "2022-03-03", est=True),
    _e("2022-09-30", "SHREECEM", "ADANIENT", "2022-09-01"),
    _e("2023-07-13", "HDFC", "LTIM", "2023-07-04", reason="Merger (HDFC Ltd into HDFC Bank)"),
    _e("2024-03-28", "UPL", "SHRIRAMFIN", "2024-02-29"),
    _e("2024-09-30", "DIVISLAB", "BEL", "2024-08-23"),
    _e("2024-09-30", "LTIM", "TRENT", "2024-08-23"),
    _e("2025-03-28", "BPCL", "JIOFIN", "2025-02-21"),
    _e("2025-03-28", "BRITANNIA", "ETERNAL", "2025-02-21"),
    _e("2025-09-30", "HEROMOTOCO", "INDIGO", "2025-08-23"),
    _e("2025-09-30", "INDUSINDBK", "MAXHEALTH", "2025-08-23"),
]
_EVENTS_DESC = sorted(REPLACEMENT_EVENTS, key=lambda e: e.effective_date, reverse=True)

TRACKED_SYMBOLS: frozenset[str] = frozenset(ANCHOR_MEMBERS) | {
    s for e in REPLACEMENT_EVENTS for s in (e.symbol_in, e.symbol_out)
}


def members_as_of(date: pd.Timestamp | str) -> frozenset[str]:
    """Exact Nifty 50 symbols on ``date`` (reverse-apply events newer than it)."""
    date = pd.Timestamp(date)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    members = set(ANCHOR_MEMBERS)
    for e in _EVENTS_DESC:
        if e.effective_date <= date:
            break
        members.discard(e.symbol_in)
        members.add(e.symbol_out)
    return frozenset(members)


def instrument_ids_as_of(date: pd.Timestamp | str) -> frozenset[str]:
    return frozenset(f"NSE:{s}" for s in members_as_of(date))


MEMBERSHIP_COLUMNS = ["instrument_id", "event_time", "available_time", "action", "reason"]


def build_membership_change_log() -> pd.DataFrame:
    rows = [
        {"instrument_id": f"NSE:{sym}", "event_time": to_ist(e.effective_date),
         "available_time": to_ist(e.announced_date), "action": action, "reason": e.reason}
        for e in REPLACEMENT_EVENTS
        for sym, action in ((e.symbol_out, "removed"), (e.symbol_in, "added"))
    ]
    return pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS).sort_values("event_time").reset_index(drop=True)


def membership_mask(bars: pd.DataFrame) -> pd.Series:
    """Boolean mask: row's instrument was a real constituent on the row's date.
    Instruments this module doesn't track (e.g. the index series) pass through.
    Computed per rebalancing epoch (membership only changes on ~8 dates)."""
    ids = bars["instrument_id"].to_numpy()
    tracked = bars["instrument_id"].isin({f"NSE:{s}" for s in TRACKED_SYMBOLS}).to_numpy()
    mask = np.ones(len(bars), dtype=bool)
    boundaries = sorted({e.effective_date for e in REPLACEMENT_EVENTS})
    starts = [pd.Timestamp("1990-01-01"), *boundaries]
    ends = [*boundaries, pd.Timestamp("2100-01-01")]
    for s, e in zip(starts, ends):
        in_epoch = ((bars["event_time"] >= to_ist(s)) & (bars["event_time"] < to_ist(e))).to_numpy()
        valid = instrument_ids_as_of(s if s > starts[0] else e - pd.Timedelta(days=1))
        sel = in_epoch & tracked
        mask[sel] = np.isin(ids[sel], list(valid))
    return pd.Series(mask, index=bars.index)


def filter_bars_to_members(bars: pd.DataFrame) -> pd.DataFrame:
    """The survivorship-bias fix: keep only rows whose instrument was a real
    Nifty 50 constituent on that row's own date."""
    return bars[membership_mask(bars)].sort_values(["instrument_id", "event_time"]).reset_index(drop=True)
