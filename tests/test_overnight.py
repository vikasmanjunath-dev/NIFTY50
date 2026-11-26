from nifty_analyser.ops.overnight import ACCURACY_BY_MOVE, signal


def test_follows_sign_and_abstains_below_threshold():
    assert signal(0.012).direction == "UP" and signal(-0.012).direction == "DOWN" and signal(0.004).direction is None and signal(-0.001).direction is None


def test_expected_accuracy_uses_the_right_bucket_and_states_open_to_close_is_chance():
    a = signal(0.011).expected_accuracy; assert a["bucket_min_move"] == 0.01 and a["opening_gap"] == 0.845 and a["open_to_close"] < 0.5
    assert signal(0.006).expected_accuracy["bucket_min_move"] == 0.005 and "priced in" in signal(0.02).note
    assert all(r[4] < 0.52 for r in ACCURACY_BY_MOVE)          # the tradable part must never be advertised as skill
