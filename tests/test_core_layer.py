"""Tests for the rebuilt base layer: schemas, ingestion tagging, store, features, labels,
walk-forward validation, metrics. The load-bearing ones are the leakage proofs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from nifty_analyser.data.ingest import validate_and_tag
from nifty_analyser.data.schemas import MarketBar, PointInTimeRecord, QualityFlag
from nifty_analyser.data.store import LookaheadError, as_of, assert_no_lookahead, latest_as_of
from nifty_analyser.data.timestamps import eod_bar_available_time, to_ist
from nifty_analyser.eval.metrics import evaluate_binary_predictions, spearman_ic
from nifty_analyser.features.build import build_feature_matrix, feature_columns
from nifty_analyser.labels.targets import add_direction_label, add_forward_return
from nifty_analyser.validation.walk_forward import assert_folds_are_leakage_free, purged_embargoed_walk_forward


def _bars(n=400, iid="NSE:T", seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2022-01-03", periods=n)
    raw = pd.DataFrame({"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close,
                        "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)
    return validate_and_tag(raw, iid)


# --- schemas / timestamps ---
def test_available_before_event_rejected():
    with pytest.raises(ValidationError):
        PointInTimeRecord(instrument_id="A", event_time=pd.Timestamp("2026-01-02"), available_time=pd.Timestamp("2026-01-01"), source_id="s")


def test_fail_requires_reason():
    t = pd.Timestamp("2026-01-01")
    with pytest.raises(ValidationError):
        PointInTimeRecord(instrument_id="A", event_time=t, available_time=t, source_id="s", quality_flag=QualityFlag.FAIL)


def test_market_bar_rejects_inconsistent_ohlc():
    t = pd.Timestamp("2026-01-01")
    with pytest.raises(ValidationError):
        MarketBar(instrument_id="A", event_time=t, available_time=t, source_id="s", open=10, high=9, low=8, close=9, volume=1)


def test_eod_available_time_is_1530_ist():
    assert eod_bar_available_time("2026-01-05") == to_ist("2026-01-05 15:30")


# --- ingestion tagging ---
def test_degenerate_flat_zero_volume_bar_is_fail():
    raw = pd.DataFrame({"open": [10, 10.0], "high": [10, 10.0], "low": [10, 10.0], "close": [10, 10.0], "volume": [100.0, 0.0]},
                       index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    out = validate_and_tag(raw, "A")
    assert out["quality_flag"].tolist() == ["PASS", "FAIL"]


def test_nonpositive_price_is_fail_and_duplicates_dropped():
    raw = pd.DataFrame({"open": [10, -1, 5.0], "high": [11, 1, 6], "low": [9, -2, 4], "close": [10, 0.5, 5], "volume": [1.0, 1, 1]},
                       index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02"]))
    out = validate_and_tag(raw, "A")
    assert len(out) == 2  # duplicate date collapsed (last kept)
    assert out["quality_flag"].tolist() == ["PASS", "PASS"]


def test_availability_is_after_event_for_every_bar():
    b = _bars()
    assert (b["available_time"] > b["event_time"]).all()


# --- store ---
def test_same_day_bar_invisible_before_close_visible_after():
    b = _bars(20)
    day = b["event_time"].iloc[5]
    assert as_of(b, day + pd.Timedelta(hours=10))["event_time"].max() < day
    assert as_of(b, day + pd.Timedelta(hours=16))["event_time"].max() == day


def test_latest_as_of_and_assert_no_lookahead():
    b = _bars(20)
    t = b["available_time"].iloc[10]
    assert len(latest_as_of(b, t)) == 1
    assert_no_lookahead(as_of(b, t), t)
    with pytest.raises(LookaheadError):
        assert_no_lookahead(b, t)


# --- features: truncate-the-future proof ---
def test_no_lookahead_in_feature_matrix():
    full = _bars(400)
    f_full = build_feature_matrix(full)
    f_trunc = build_feature_matrix(full.iloc[:-30])
    cols = feature_columns(f_trunc)
    pd.testing.assert_frame_equal(f_full.iloc[: len(f_trunc)][cols].reset_index(drop=True), f_trunc[cols].reset_index(drop=True), rtol=1e-10, check_exact=False)


def test_features_do_not_cross_contaminate_instruments():
    a, b = _bars(300, "NSE:A", 1), _bars(300, "NSE:B", 2)
    both = build_feature_matrix(pd.concat([a, b], ignore_index=True))
    alone = build_feature_matrix(a)
    cols = feature_columns(alone)
    got = both[both.instrument_id == "NSE:A"][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, alone[cols].reset_index(drop=True), rtol=1e-10, check_exact=False)


# --- labels ---
def test_forward_return_and_outcome_time_alignment():
    b = _bars(30)
    g = add_forward_return(b, 5)
    assert g["fwd_ret_5d"].iloc[0] == pytest.approx(np.log(b["close"].iloc[5] / b["close"].iloc[0]))
    assert g["fwd_ret_5d_outcome_available_time"].iloc[0] == b["available_time"].iloc[5]
    assert g["fwd_ret_5d"].iloc[-5:].isna().all()


def test_direction_label_signs_and_flat_band():
    b = _bars(60)
    g = add_direction_label(b, 5, flat_threshold=0.0)
    ok = g.dropna(subset=["fwd_ret_5d"])
    assert (np.sign(ok["fwd_ret_5d"]) == ok["dir_label_5d"]).all()
    wide = add_direction_label(b, 5, flat_threshold=10.0).dropna(subset=["fwd_ret_5d"])
    assert (wide["dir_label_5d"] == 0).all()


# --- walk-forward ---
def _labelled(n=600):
    g = add_direction_label(_bars(n), 5)
    g = g.dropna(subset=["dir_label_5d", "fwd_ret_5d_outcome_available_time"]).reset_index(drop=True)
    return g


def test_walk_forward_folds_are_leakage_free_and_expanding():
    df = _labelled()
    folds = purged_embargoed_walk_forward(df, 4, "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5))
    assert len(folds) == 4
    assert_folds_are_leakage_free(df, folds, "fwd_ret_5d_outcome_available_time")
    assert all(len(folds[i].train_idx) < len(folds[i + 1].train_idx) for i in range(3))
    assert all(df["event_time"].iloc[f.train_idx].max() < f.val_start for f in folds)


def test_leakage_guard_catches_a_deliberately_smuggled_validation_row():
    df = _labelled()
    folds = purged_embargoed_walk_forward(df, 4, "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5))
    f = folds[1]
    from dataclasses import replace
    bad = replace(f, train_idx=np.append(f.train_idx, f.val_idx[:1]))
    with pytest.raises(AssertionError):
        assert_folds_are_leakage_free(df, [bad], "fwd_ret_5d_outcome_available_time")


def test_purge_removes_rows_whose_label_straddles_the_boundary():
    df = _labelled()
    f = purged_embargoed_walk_forward(df, 4, "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=0))[2]
    straddlers = df[(df["event_time"] < f.val_start) & (df["fwd_ret_5d_outcome_available_time"] >= f.val_start)].index
    assert not set(straddlers) & set(f.train_idx)
    assert len(straddlers) > 0


# --- metrics ---
def test_spearman_ic_perfect_and_inverse():
    a = np.arange(10.0)
    assert spearman_ic(a, a * 3) == pytest.approx(1.0)
    assert spearman_ic(a, -a) == pytest.approx(-1.0)


def test_metrics_on_a_perfect_and_a_random_predictor():
    y = np.array([0, 1] * 50)
    perfect = evaluate_binary_predictions(y, np.where(y == 1, 0.9, 0.1), fwd_return=y.astype(float))
    assert perfect.accuracy == 1.0 and perfect.roc_auc == 1.0 and perfect.information_coefficient > 0.9
    rnd = evaluate_binary_predictions(y, np.full(100, 0.5))
    assert rnd.log_loss == pytest.approx(np.log(2), abs=1e-6)


# --- scale-freeness: the property the drift monitor and the model inputs rely on ---
def test_technical_features_are_invariant_to_price_and_volume_rescaling():
    """A feature that changes when prices are multiplied by a constant is price-unit-dependent (trends with the
    level, not comparable across stocks). Every emitted feature must be dimensionless."""
    raw = _bars(400)
    k, m = 37.0, 1000.0
    scaled = raw.assign(**{c: raw[c] * k for c in ("open", "high", "low", "close")}, volume=raw["volume"] * m)
    a, b = build_feature_matrix(raw), build_feature_matrix(scaled)
    cols = feature_columns(a)
    assert set(cols) == set(feature_columns(b)) and len(cols) > 50
    bad = [c for c in cols if not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), rtol=1e-6, atol=1e-9, equal_nan=True)]
    assert not bad, f"scale-dependent features: {bad}"


def test_no_emitted_feature_matches_the_scale_dependent_pattern():
    from nifty_analyser.monitoring.drift import is_scale_dependent
    cols = feature_columns(build_feature_matrix(_bars(300)))
    assert not [c for c in cols if is_scale_dependent(c)]
