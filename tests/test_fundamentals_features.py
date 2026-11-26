"""Tests for fundamental feature engineering and its point-in-time as-of join."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_analyser.features.fundamentals import (
    add_fundamental_ratios,
    merge_fundamentals_into_features,
)


def _fundamentals(n_quarters: int = 8, instrument_id: str = "NSE:TEST") -> pd.DataFrame:
    filing_dates = pd.date_range("2023-01-20", periods=n_quarters, freq="QS", tz="Asia/Kolkata")
    period_ends = filing_dates - pd.Timedelta(days=20)  # filed ~20 days after period end
    revenue = np.linspace(1000.0, 1800.0, n_quarters)
    profit = revenue * 0.15
    eps = np.linspace(10.0, 18.0, n_quarters)
    return pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "event_time": period_ends,
            "available_time": filing_dates,
            "revenue": revenue,
            "profit": profit,
            "eps_basic": eps,
        }
    )


def test_profit_margin_computed_correctly():
    f = _fundamentals(4)
    out = add_fundamental_ratios(f)
    expected = out["profit"] / out["revenue"]
    pd.testing.assert_series_equal(out["profit_margin"], expected, check_names=False)


def test_yoy_growth_uses_four_quarters_back():
    f = _fundamentals(8)
    out = add_fundamental_ratios(f)
    row4 = out.iloc[4]
    row0 = out.iloc[0]
    expected_growth = row4["eps_basic"] / row0["eps_basic"] - 1.0
    assert out["eps_growth_yoy"].iloc[4] == pytest.approx(expected_growth)


def test_first_four_quarters_have_no_yoy_growth():
    f = _fundamentals(8)
    out = add_fundamental_ratios(f)
    assert out["eps_growth_yoy"].iloc[:4].isna().all()


def _daily_features(n_days: int, instrument_id: str = "NSE:TEST") -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "event_time": dates,
            "available_time": dates + pd.Timedelta(hours=15, minutes=30),
        }
    )


def test_as_of_merge_carries_forward_last_known_fundamentals():
    fundamentals = add_fundamental_ratios(_fundamentals(4))
    features = _daily_features(200)
    merged = merge_fundamentals_into_features(features, fundamentals)

    first_filing_time = fundamentals["available_time"].min()
    before = merged[merged["available_time"] < first_filing_time]
    after = merged[merged["available_time"] >= first_filing_time]

    assert before["revenue"].isna().all()  # nothing known yet
    assert after["revenue"].notna().all()  # carried forward once published


def test_as_of_merge_never_uses_a_not_yet_published_quarter():
    """Direct point-in-time proof: for every daily row, the merged fundamentals
    row's available_time must be <= the daily row's own available_time.
    """
    fundamentals = add_fundamental_ratios(_fundamentals(6))
    features = _daily_features(400)
    merged = merge_fundamentals_into_features(features, fundamentals)

    # Reconstruct which fundamentals row was actually used by matching eps_basic
    # back to the source, then confirm its filing date precedes the daily row.
    fund_lookup = fundamentals.set_index("eps_basic")["available_time"]
    matched = merged.dropna(subset=["eps_basic"])
    used_filing_times = matched["eps_basic"].map(fund_lookup)
    assert (used_filing_times <= matched["available_time"]).all()


def test_as_of_merge_updates_to_newer_quarter_after_next_filing():
    fundamentals = add_fundamental_ratios(_fundamentals(4))
    features = _daily_features(400)
    merged = merge_fundamentals_into_features(features, fundamentals)

    second_filing_time = fundamentals["available_time"].iloc[1]
    third_filing_time = fundamentals["available_time"].iloc[2]

    just_before_third = merged[
        (merged["available_time"] >= second_filing_time) & (merged["available_time"] < third_filing_time)
    ]
    just_after_third = merged[merged["available_time"] >= third_filing_time]

    assert just_before_third["eps_basic"].nunique() == 1
    assert just_after_third["eps_basic"].iloc[0] != just_before_third["eps_basic"].iloc[0]


def test_no_fundamentals_at_all_leaves_columns_nan_not_crash():
    features = _daily_features(50)
    empty_fundamentals = add_fundamental_ratios(_fundamentals(0))
    merged = merge_fundamentals_into_features(features, empty_fundamentals)
    assert merged["revenue"].isna().all()
    assert len(merged) == 50


def test_multiple_instruments_do_not_cross_contaminate():
    fund_a = add_fundamental_ratios(_fundamentals(4, instrument_id="NSE:AAA"))
    fund_b = add_fundamental_ratios(_fundamentals(4, instrument_id="NSE:BBB"))
    fund_b["eps_basic"] = fund_b["eps_basic"] * 100  # make them obviously distinguishable
    fundamentals = pd.concat([fund_a, fund_b], ignore_index=True)

    features = pd.concat(
        [_daily_features(300, "NSE:AAA"), _daily_features(300, "NSE:BBB")], ignore_index=True
    )
    merged = merge_fundamentals_into_features(features, fundamentals)

    a_eps = merged[merged.instrument_id == "NSE:AAA"]["eps_basic"].dropna()
    b_eps = merged[merged.instrument_id == "NSE:BBB"]["eps_basic"].dropna()
    assert a_eps.max() < b_eps.min()  # b's values are 100x, should never overlap a's


def test_yoy_growth_is_nan_not_infinite_when_prior_year_value_is_zero():
    """Regression (real data): a quarter reported as exactly 0.0 produced inf four quarters later,
    which crashed the linear model's imputer once fundamentals reached a real fit."""
    f = _fundamentals(8)
    f.loc[0, ["eps_basic", "revenue"]] = 0.0
    out = add_fundamental_ratios(f)
    assert pd.isna(out["eps_growth_yoy"].iloc[4]) and pd.isna(out["revenue_growth_yoy"].iloc[4])
    assert not np.isinf(out[["eps_growth_yoy", "revenue_growth_yoy", "profit_margin"]].to_numpy(dtype=float)).any()


def test_scale_free_fundamentals_and_levels_are_dropped():
    from nifty_analyser.features.fundamentals import finalize_fundamental_features
    f = add_fundamental_ratios(_fundamentals(8))
    assert f["eps_ttm"].iloc[:3].isna().all() and f["eps_ttm"].iloc[3] == pytest.approx(f["eps_basic"].iloc[:4].sum())
    feats = _daily_features(300).assign(close=100.0)
    out = finalize_fundamental_features(merge_fundamentals_into_features(feats, f))
    assert {"earnings_yield", "profit_margin", "eps_growth_yoy", "profit_growth_yoy"} <= set(out.columns)
    assert not {"revenue", "profit", "eps_basic", "eps_ttm"} & set(out.columns)
    known = f[f["available_time"] <= feats["available_time"].iloc[-1]].iloc[-1]      # latest filing published by the last day
    assert out["earnings_yield"].dropna().iloc[-1] == pytest.approx(known["eps_ttm"] / 100.0)


def test_profit_growth_is_nan_not_misleading_for_a_loss_making_base():
    f = _fundamentals(8)
    f.loc[0, "profit"] = -50.0           # loss four quarters ago: "growth" vs a negative base is meaningless
    f.loc[1, "profit"] = 0.0
    out = add_fundamental_ratios(f)
    assert pd.isna(out["profit_growth_yoy"].iloc[4]) and pd.isna(out["profit_growth_yoy"].iloc[5])
    assert not np.isinf(out["profit_growth_yoy"].to_numpy(dtype=float)).any()
