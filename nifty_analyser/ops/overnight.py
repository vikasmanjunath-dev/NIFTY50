"""Pre-open direction signal for the Nifty from the US close (point-in-time: the US session closes ~02:00 IST, before the Indian open).
Rule with NO fitted parameters: predict the sign of the Indian move as the sign of the S&P 500 return; abstain when |S&P| < threshold.
Measured walk-forward-free (no training) on 2021-2026 (reports/overnight_direction_by_move.json): opening-gap direction 68% on all days, 78% when |S&P| >= 0.5%,
85% at >= 1%; close-to-close 58% / 65% / 68%; **open-to-close (the only part tradable after the open) ~47-48%, i.e. chance**. The US move is already in the opening
price, so this describes the gap, it is not a trading edge."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_THRESHOLD = 0.005          # fixed in advance (0.5%), not tuned
# measured accuracy of 'follow the S&P sign' on the Nifty index by |S&P| bucket (reports/overnight_direction_by_move.json): (min |move|, coverage, gap, close_to_close, open_to_close)
ACCURACY_BY_MOVE = [(0.0, 1.00, 0.684, 0.580, 0.476), (0.0025, 0.741, 0.729, 0.605, 0.474), (0.005, 0.537, 0.781, 0.647, 0.483), (0.01, 0.261, 0.845, 0.682, 0.458), (0.015, 0.119, 0.861, 0.723, 0.468)]


@dataclass(frozen=True)
class OvernightSignal:
    direction: str | None            # "UP" | "DOWN" | None (abstain)
    spx_return: float
    expected_accuracy: dict | None   # measured historical accuracy for this move size
    note: str


def signal(spx_return: float, threshold: float = DEFAULT_THRESHOLD) -> OvernightSignal:
    if abs(spx_return) < threshold:
        return OvernightSignal(None, spx_return, None, f"abstain: |S&P move| {abs(spx_return):.2%} below {threshold:.2%}")
    row = max((r for r in ACCURACY_BY_MOVE if abs(spx_return) >= r[0]), key=lambda r: r[0])
    acc = {"bucket_min_move": row[0], "historical_coverage": row[1], "opening_gap": row[2], "close_to_close": row[3], "open_to_close": row[4]}
    return OvernightSignal("UP" if spx_return > 0 else "DOWN", spx_return, acc, "direction of the Indian OPENING GAP/close-to-close; open-to-close accuracy is ~chance (the move is priced in at the open)")
