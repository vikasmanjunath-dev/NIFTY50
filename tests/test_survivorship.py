"""Tests for survivorship-bias measurement on synthetic data with a KNOWN planted bias."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_analyser.eval.survivorship import bias_diagnostics, daily_cross_sectional_ic, universe_frames


def _frame(rows):
    df = pd.DataFrame(rows, columns=["instrument_id", "date", "fwd_ret_1d"])
    df["event_time"] = pd.to_datetime(df.pop("date")).dt.tz_localize("Asia/Kolkata")
    df["y"] = 1
    return df


def test_universe_frames_phantom_and_omitted_rows_are_exactly_the_membership_delta():
    # IOC left the index 2022-03-31 (omitted from 'biased': not a current member);
    # APOLLOHOSP joined that day (phantom before it in 'biased'); ONGC always a member.
    rows = [(i, d, 0.0) for i in ("NSE:IOC", "NSE:APOLLOHOSP", "NSE:ONGC") for d in ("2022-03-25", "2022-04-05")]
    biased, pit = universe_frames(_frame(rows), start="2021-09-20")
    b = {(r.instrument_id, r.event_time.strftime("%m-%d")) for r in biased.itertuples()}
    p = {(r.instrument_id, r.event_time.strftime("%m-%d")) for r in pit.itertuples()}
    assert b == {("NSE:APOLLOHOSP", "03-25"), ("NSE:APOLLOHOSP", "04-05"), ("NSE:ONGC", "03-25"), ("NSE:ONGC", "04-05")}
    assert p == {("NSE:IOC", "03-25"), ("NSE:APOLLOHOSP", "04-05"), ("NSE:ONGC", "03-25"), ("NSE:ONGC", "04-05")}


def test_bias_diagnostics_recovers_a_planted_survivorship_premium():
    """Planted: the future joiner (phantom) earns +0.2%/day before joining, the leaver
    (omitted) earns -0.1%/day -> the biased universe must score higher, by the planted amount."""
    dates = pd.bdate_range("2022-02-01", "2022-03-30")
    rows = []
    for d in dates:
        rows += [("NSE:APOLLOHOSP", d, 0.002), ("NSE:IOC", d, -0.001), ("NSE:ONGC", d, 0.0)]
    biased, pit = universe_frames(_frame(rows))
    diag = bias_diagnostics(biased, pit)
    assert diag["phantom_rows (in biased, not real members)"]["mean_daily_return"] == np.float64(0.002)
    assert diag["omitted_rows (real members missing from biased)"]["mean_daily_return"] == np.float64(-0.001)
    # EW biased = mean(0.002, 0) = 0.001 ; EW pit = mean(-0.001, 0) = -0.0005 ; gap 0.0015/day
    assert abs(diag["survivorship_premium_annualised"] - 0.0015 * 252) < 1e-9


def test_no_bias_when_universes_coincide():
    rows = [("NSE:ONGC", d, 0.001) for d in pd.bdate_range("2025-11-03", periods=5)]
    biased, pit = universe_frames(_frame(rows))
    diag = bias_diagnostics(biased, pit)
    assert diag["survivorship_premium_annualised"] == 0.0
    assert diag["phantom_rows (in biased, not real members)"]["n_rows"] == 0


def test_cross_sectional_ic_perfect_signal():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.bdate_range("2022-01-03", periods=20):
        for i in range(12):
            rows.append({"event_time": d, "score": float(i), "ret": float(i) + rng.normal(0, 1e-6)})
    assert daily_cross_sectional_ic(pd.DataFrame(rows), "score", "ret") > 0.99
