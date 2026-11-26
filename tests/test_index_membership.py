"""Tests for point-in-time Nifty 50 index membership reconstruction."""
from __future__ import annotations

import pandas as pd

from nifty_analyser.data.index_membership import (
    ANCHOR_MEMBERS, REPLACEMENT_EVENTS, build_membership_change_log, filter_bars_to_members,
    instrument_ids_as_of, members_as_of,
)


def test_anchor_has_exactly_fifty_members():
    assert len(ANCHOR_MEMBERS) == 50


def test_membership_count_is_always_fifty_across_the_window():
    for date in pd.date_range("2021-09-20", "2026-11-24", freq="7D"):
        assert len(members_as_of(date)) == 50, date


def test_after_last_event_equals_anchor():
    assert members_as_of("2026-01-01") == ANCHOR_MEMBERS


def _swap(before_date, after_date, out_syms, in_syms):
    b, a = members_as_of(before_date), members_as_of(after_date)
    assert set(out_syms) <= b and not set(in_syms) & b
    assert set(in_syms) <= a and not set(out_syms) & a


def test_every_sourced_event_flips_exactly_on_its_effective_date():
    for e in REPLACEMENT_EVENTS:
        _swap(e.effective_date - pd.Timedelta(days=1), e.effective_date, [e.symbol_out], [e.symbol_in])


def test_start_universe_matches_independent_hand_derivation():
    """2021-09-20 universe = current 50 with every in-window change reversed,
    derived by hand (not by the code under test)."""
    expected = (set(ANCHOR_MEMBERS) - {
        "APOLLOHOSP", "ADANIENT", "LTIM", "SHRIRAMFIN", "BEL", "TRENT",
        "JIOFIN", "ETERNAL", "INDIGO", "MAXHEALTH",
    }) | {"IOC", "SHREECEM", "HDFC", "UPL", "DIVISLAB", "BPCL", "BRITANNIA", "HEROMOTOCO", "INDUSINDBK"}
    assert members_as_of("2021-09-20") == expected


def test_instrument_ids_prefixed():
    assert "NSE:ONGC" in instrument_ids_as_of("2021-09-20")


def test_change_log_point_in_time():
    log = build_membership_change_log()
    assert len(log) == 2 * len(REPLACEMENT_EVENTS)
    assert (log["available_time"] <= log["event_time"]).all()
    assert set(log["action"]) == {"added", "removed"}


def _bars(iid, dates):
    return pd.DataFrame({"instrument_id": iid, "event_time": pd.to_datetime(dates).tz_localize("Asia/Kolkata"), "close": 1.0})


def test_filter_drops_after_exit_and_before_entry():
    out = filter_bars_to_members(pd.concat([
        _bars("NSE:IOC", ["2022-03-25", "2022-04-05"]),
        _bars("NSE:APOLLOHOSP", ["2022-03-25", "2022-04-05"]),
        _bars("NSE:ONGC", ["2022-03-25", "2022-04-05"]),
        _bars("NSE:NIFTY50_INDEX", ["2022-03-25"]),
    ]))
    got = {(r.instrument_id, r.event_time.strftime("%m-%d")) for r in out.itertuples()}
    assert got == {("NSE:IOC", "03-25"), ("NSE:APOLLOHOSP", "04-05"),
                   ("NSE:ONGC", "03-25"), ("NSE:ONGC", "04-05"), ("NSE:NIFTY50_INDEX", "03-25")}
