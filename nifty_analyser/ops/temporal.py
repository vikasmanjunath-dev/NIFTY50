"""Temporal intent of a question. A question with no explicit time reference means 'as of when I ask' (the freshest information wins); a question that
names a month/year ('announced in Nov 2023', 'during 2022') is about that period, so the corpus is restricted to it and recency is NOT boosted.
Anchors refer to the filing's PUBLICATION time. All results are timezone-aware IST and never extend past ``as_of`` (point-in-time safety)."""
from __future__ import annotations

import re

import pandas as pd

TZ = "Asia/Kolkata"
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH_YEAR = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?,?\s+(?:of\s+)?(20\d{2}|19\d{2})\b", re.I)
_YEAR = re.compile(r"\b(?:in|during|for|of|from)\s+(?:the\s+year\s+|fy\s*)?(20\d{2})\b", re.I)
_PAST_PERIOD = re.compile(r"\b(last|previous|prior)\s+(year|quarter|month|week)\b", re.I)


def parse_time_anchor(query: str, as_of: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(start, end) of the period the question names, clipped to ``as_of``; None when the question has no explicit time reference."""
    m = _MONTH_YEAR.search(query)
    if m:
        start = pd.Timestamp(year=int(m.group(2)), month=_MONTHS[m.group(1)[:3].lower()], day=1, tz=TZ)
        end = start + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)
    else:
        y = _YEAR.search(query)
        if not y:
            return None
        start = pd.Timestamp(year=int(y.group(1)), month=1, day=1, tz=TZ); end = pd.Timestamp(year=int(y.group(1)), month=12, day=31, hour=23, minute=59, second=59, tz=TZ)
    if start > as_of:
        return None
    return start, min(end, as_of)


def has_explicit_time(query: str) -> bool:
    return bool(_MONTH_YEAR.search(query) or _YEAR.search(query) or _PAST_PERIOD.search(query))
