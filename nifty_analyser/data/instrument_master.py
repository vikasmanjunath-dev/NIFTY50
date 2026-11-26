"""Instrument master: the survivorship-bias-free POOL of every symbol that was a
real Nifty 50 constituent at any point in this project's data window (Blueprint §3.4).

``NIFTY50`` is the *pool* (current 50 + historical leavers), not "today's 50":
which of them were actually index members on a given date is the job of
``data/index_membership.py``. Using the pool everywhere ingestion/announcement/
fundamentals code looks up an instrument is what lets those pipelines cover
stocks that have since left the index.

Known data gaps (confirmed against yfinance, not assumed): see
``_NOT_INGESTIBLE`` and ``_YAHOO_OVERRIDES``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    instrument_id: str  # stable internal id, e.g. "NSE:RELIANCE"
    symbol: str         # NSE trading symbol
    yahoo_ticker: str   # ticker used for yfinance ingestion
    sector: str
    ingestible: bool = True
    aliases: tuple[str, ...] = ()  # earlier NSE symbols (pre-rename), still present in historical records


# (symbol, sector): approximate NSE sector groupings, adequate for
# sector-relative features, not a licensed taxonomy.
_CURRENT_50: list[tuple[str, str]] = [
    ("ADANIENT", "Diversified"), ("ADANIPORTS", "Industrials"), ("APOLLOHOSP", "Healthcare"),
    ("ASIANPAINT", "Consumer"), ("AXISBANK", "Financials"), ("BAJAJ-AUTO", "Auto"),
    ("BAJFINANCE", "Financials"), ("BAJAJFINSV", "Financials"), ("BEL", "Industrials"),
    ("BHARTIARTL", "Telecom"), ("CIPLA", "Pharma"), ("COALINDIA", "Energy"),
    ("DRREDDY", "Pharma"), ("EICHERMOT", "Auto"), ("ETERNAL", "Consumer"),
    ("GRASIM", "Materials"), ("HCLTECH", "IT"), ("HDFCBANK", "Financials"),
    ("HDFCLIFE", "Financials"), ("HINDALCO", "Materials"), ("HINDUNILVR", "Consumer"),
    ("ICICIBANK", "Financials"), ("INDIGO", "Industrials"), ("INFY", "IT"), ("ITC", "Consumer"),
    ("JIOFIN", "Financials"), ("JSWSTEEL", "Materials"), ("KOTAKBANK", "Financials"),
    ("LT", "Industrials"), ("M&M", "Auto"), ("MARUTI", "Auto"), ("MAXHEALTH", "Healthcare"),
    ("NESTLEIND", "Consumer"), ("NTPC", "Utilities"), ("ONGC", "Energy"),
    ("POWERGRID", "Utilities"), ("RELIANCE", "Energy"), ("SBILIFE", "Financials"),
    ("SHRIRAMFIN", "Financials"), ("SBIN", "Financials"), ("SUNPHARMA", "Pharma"),
    ("TCS", "IT"), ("TATACONSUM", "Consumer"), ("TATAMOTORS", "Auto"), ("TATASTEEL", "Materials"),
    ("TECHM", "IT"), ("TITAN", "Consumer"), ("TRENT", "Consumer"), ("ULTRACEMCO", "Materials"),
    ("WIPRO", "IT"),
]

# Real Nifty 50 members for part of the data window that have since left.
_HISTORICAL_LEAVERS: list[tuple[str, str]] = [
    ("IOC", "Energy"), ("SHREECEM", "Materials"), ("HDFC", "Financials"), ("LTIM", "IT"),
    ("UPL", "Materials"), ("DIVISLAB", "Pharma"), ("BPCL", "Energy"),
    ("BRITANNIA", "Consumer"), ("HEROMOTOCO", "Auto"), ("INDUSINDBK", "Financials"),
]

# HDFC.NS returns no data at all from yfinance (full history removed after the
# July 2023 merger into HDFC Bank) -- confirmed directly. Its index seat is
# still recorded in index_membership; its bars are honestly absent.
_NOT_INGESTIBLE = {"HDFC"}

# Symbols whose Yahoo ticker differs from "<SYMBOL>.NS" (set after live checks).
_YAHOO_OVERRIDES: dict[str, str] = {
    # Tata Motors demerged; Yahoo serves the continuous (demerger-adjusted)
    # passenger-vehicle history under TMPV.NS, while TATAMOTORS.NS now 404s.
    "TATAMOTORS": "TMPV.NS",
    # LTIMindtree trades as LTM since its rename; LTIM.NS 404s (probed 2026-11-24).
    "LTIM": "LTM.NS",
}


# NSE trading symbols that changed after a corporate action: the instrument_id stays stable
# (NSE:LTIM), the live NSE symbol differs, and the old symbol is kept as an alias for old records.
_NSE_SYMBOL_OVERRIDES = {"LTIM": "LTM", "TATAMOTORS": "TMPV"}


def _yahoo_ticker(symbol: str) -> str:
    return _YAHOO_OVERRIDES.get(symbol, f"{symbol}.NS")


NIFTY50: dict[str, Instrument] = {
    f"NSE:{symbol}": Instrument(
        instrument_id=f"NSE:{symbol}",
        symbol=_NSE_SYMBOL_OVERRIDES.get(symbol, symbol),
        aliases=(symbol,) if symbol in _NSE_SYMBOL_OVERRIDES else (),
        yahoo_ticker=_yahoo_ticker(symbol),
        sector=sector,
        ingestible=symbol not in _NOT_INGESTIBLE,
    )
    for symbol, sector in [*_CURRENT_50, *_HISTORICAL_LEAVERS]
}

CURRENT_SYMBOLS: frozenset[str] = frozenset(s for s, _ in _CURRENT_50)


def all_instrument_ids() -> list[str]:
    return list(NIFTY50)


def ingestible_instrument_ids() -> list[str]:
    return [i.instrument_id for i in NIFTY50.values() if i.ingestible]


# The index itself, deliberately outside the constituent pool.
NIFTY_INDEX = Instrument("NSE:NIFTY50_INDEX", "NIFTY50_INDEX", "^NSEI", "Index")


def get_instrument(instrument_id: str) -> Instrument:
    if instrument_id == NIFTY_INDEX.instrument_id:
        return NIFTY_INDEX
    try:
        return NIFTY50[instrument_id]
    except KeyError as exc:
        raise KeyError(f"Unknown instrument_id {instrument_id!r}") from exc


# Company names that changed (filings before/after use different names than today's symbol suggests). Used to hint the answering model that
# evidence about "Zomato Limited" is about today's ETERNAL, instead of letting it abstain on a name mismatch.
FORMER_NAMES: dict[str, tuple[str, ...]] = {"NSE:ETERNAL": ("Zomato",), "NSE:TMPV": ("Tata Motors", "Tata Motors Passenger Vehicles"), "NSE:LTM": ("LTIMindtree", "LTI Mindtree")}
