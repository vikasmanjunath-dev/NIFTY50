import numpy as np
import pandas as pd
import pytest
from nifty_analyser.labels.triple_barrier import add_triple_barrier_label, meta_labels, uniqueness_weights


def _g(closes):
    d = pd.bdate_range("2024-01-01", periods=len(closes), tz="Asia/Kolkata")
    return pd.DataFrame({"event_time": d, "available_time": d + pd.Timedelta(hours=16), "close": closes})


def _noisy_base(n=40):
    return 100 * np.exp(np.cumsum(np.tile([0.01, -0.01], n // 2)))     # sigma ~ 0.01 -> width ~ 0.022 at h=5,k=1


def test_upper_lower_and_vertical_barriers_and_outcome_time():
    base = list(_noisy_base())
    up = base[:30] + [base[29] * 1.08] * 10                              # jump on bar 30
    out = add_triple_barrier_label(_g(up), 5, 1.0)
    i = 28                                                               # jump is 2 bars ahead (bar 30)
    assert out["tb_label"].iloc[i] == 1 and out["tb_y"].iloc[i] == 1 and out["tb_holding_days"].iloc[i] == 2
    assert out["tb_outcome_available_time"].iloc[i] == out["available_time"].iloc[30]
    dn = base[:30] + [base[29] * 0.92] * 10
    assert add_triple_barrier_label(_g(dn), 5, 1.0)["tb_label"].iloc[28] == -1
    flat = add_triple_barrier_label(_g(base), 5, 5.0)                    # unreachable barrier -> vertical, sign-resolved
    r = flat.dropna(subset=["tb_label"]).iloc[0]
    assert r["tb_label"] == 0 and r["tb_holding_days"] == 5 and r["tb_y"] in (0.0, 1.0)


def test_barrier_width_uses_only_past_data_and_tail_is_unlabelled():
    base = _noisy_base()
    a = add_triple_barrier_label(_g(base), 5, 1.0)
    b_closes = base.copy(); b_closes[25:] *= 3                           # change the future of bar 20
    b = add_triple_barrier_label(_g(b_closes), 5, 1.0)
    assert a["tb_width"].iloc[20] == b["tb_width"].iloc[20]              # width at 20 unaffected by later prices
    assert a["tb_label"].iloc[-5:].isna().all() and a["tb_outcome_available_time"].iloc[-5:].isna().all()
    assert (a["tb_outcome_available_time"].dropna() > a.loc[a["tb_outcome_available_time"].notna(), "event_time"]).all()


def test_uniqueness_weights_known_values():
    # labels: [0,1], [1,2], [5,5]; concurrency: bar0=1, bar1=2, bar2=1, bar5=1
    w = uniqueness_weights(np.array([0, 1, 5]), np.array([1, 2, 5]))
    assert w == pytest.approx([0.75, 0.75, 1.0])
    assert np.isnan(uniqueness_weights(np.array([0, 3]), np.array([2, -1]))[1])


def test_uniqueness_fully_overlapping_labels_share_weight_and_weights_in_unit_interval():
    w = uniqueness_weights(np.zeros(4, int), np.full(4, 3))
    assert w == pytest.approx([0.25] * 4)
    rng = np.random.default_rng(0)
    s = np.sort(rng.integers(0, 200, 100)); e = s + rng.integers(1, 6, 100)
    u = uniqueness_weights(s, e)
    assert ((u > 0) & (u <= 1)).all()


def test_meta_labels():
    assert meta_labels(np.array([1, 1, -1, -1]), np.array([1, 0, 0, 1])).tolist() == [1, 0, 1, 0]
