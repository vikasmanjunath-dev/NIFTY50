"""Point-in-time data contract (Blueprint §3.2).

Every record carries ``event_time`` (when it happened) AND ``available_time``
(the earliest moment a real system could have known it). A record is rejected
on construction if it claims to be knowable before it exists.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class QualityFlag(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class PointInTimeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    instrument_id: str
    event_time: datetime
    available_time: datetime
    source_id: str
    revision_id: str = "v1"
    quality_flag: QualityFlag = QualityFlag.PASS
    missing_reason: str | None = None

    @model_validator(mode="after")
    def _check_point_in_time(self) -> "PointInTimeRecord":
        if (self.event_time.tzinfo is None) != (self.available_time.tzinfo is None):
            raise ValueError("event_time and available_time must both be tz-aware or both naive")
        if self.available_time < self.event_time:
            raise ValueError("available_time must not precede event_time")
        if self.quality_flag == QualityFlag.FAIL and not (self.missing_reason or "").strip():
            raise ValueError("FAIL-quality records must carry a missing_reason")
        return self


class MarketBar(PointInTimeRecord):
    open: float
    high: float
    low: float
    close: float
    volume: float

    @model_validator(mode="after")
    def _check_ohlc(self) -> "MarketBar":
        if self.quality_flag != QualityFlag.FAIL:
            if min(self.open, self.high, self.low, self.close) <= 0:
                raise ValueError("prices must be positive")
            if self.high < max(self.open, self.close, self.low) or self.low > min(self.open, self.close, self.high):
                raise ValueError("inconsistent OHLC")
        return self
