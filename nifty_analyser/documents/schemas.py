"""Point-in-time document contract (Blueprint §4.3, §4.4).

Mirrors the market-data point-in-time contract (``data/schemas.py``) exactly
on purpose: the blueprint's whole Part 4 argument is that documents need the
*same* discipline as prices, not a looser one. A document's ``available_time``
is NSE's own exchange-dissemination timestamp (``exchdisstime`` in the raw
API — the moment the announcement became public), which is what makes the
point-in-time retrieval firewall in ``retrieval.py`` possible at all.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from nifty_analyser.data.schemas import PointInTimeRecord, QualityFlag


class DocumentRecord(PointInTimeRecord):
    """One NSE corporate announcement, point-in-time tagged.

    ``event_time`` here is the announcement's own timestamp (when NSE says it
    was disseminated) -- for this source, event_time and available_time are
    very close (seconds apart, per NSE's own "difference" field observed
    during investigation), unlike fundamentals (weeks) or delivery (hours).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    doc_id: str
    category: str  # NSE's own "desc" field, e.g. "Acquisition", "Updates", "Board Meeting"
    text: str
    attachment_url: str | None = None
    isin: str | None = None

    @field_validator("doc_id")
    @classmethod
    def _doc_id_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("doc_id must not be blank")
        return v

    @model_validator(mode="after")
    def _text_required_unless_fail(self) -> "DocumentRecord":
        if self.quality_flag == QualityFlag.PASS and not self.text.strip():
            raise ValueError("PASS-quality document must have non-empty text")
        return self
