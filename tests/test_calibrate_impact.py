import pandas as pd, pytest
from nifty_analyser.eval.calibrate_impact import FILL_COLUMNS, fit_impact, prepare, simulate_fills


def test_recovers_known_parameters_despite_outliers():
    fit = fit_impact(simulate_fills(n=3000, half_spread_bps=4.0, impact_coef=0.6, seed=1), n_boot=100)
    assert fit.half_spread_bps == pytest.approx(4.0, abs=3.0) and fit.impact_coef == pytest.approx(0.6, abs=0.25)
    assert fit.ci_impact_coef[0] < 0.6 < fit.ci_impact_coef[1] + 0.25 and fit.n == 3000


def test_sell_slippage_sign_is_cost_positive():
    f = simulate_fills(n=50).iloc[:1].assign(side=-1, arrival_price=100.0, fill_price=99.0)   # sold below arrival = cost
    assert prepare(f)[1][0] == pytest.approx(100.0)


def test_too_few_or_malformed_fills_are_rejected_loudly():
    with pytest.raises(ValueError, match="need >="):
        fit_impact(simulate_fills(n=10))
    with pytest.raises(ValueError, match="missing columns"):
        fit_impact(simulate_fills(n=50).drop(columns=["adv_value"]))


def test_invalid_rows_dropped_and_model_built():
    f = simulate_fills(n=400); f.loc[:5, "adv_value"] = 0; f.loc[6, "side"] = 0
    assert len(prepare(f)[0]) == 393
    m = fit_impact(f, n_boot=20).to_model(fee_bps=8.0, capital=5e8)
    assert m.fixed_bps == 8.0 and m.capital == 5e8 and m.impact_coef >= 0 and set(FILL_COLUMNS) <= set(f.columns)


def test_blend_is_monotone_mean_of_base_logits():
    import numpy as np
    from nifty_analyser.models.ensemble import BlendModel, _logit
    class M:
        def __init__(s, p): s.p = np.asarray(p)
        def predict_proba_up(s, X): return s.p
    b = BlendModel({"a": M([0.2, 0.6, 0.9]), "b": M([0.3, 0.5, 0.8])})
    out = b.predict_proba_up(None)
    assert (np.diff(out) > 0).all() and out[1] == pytest.approx(1 / (1 + np.exp(-(_logit(0.6) + _logit(0.5)) / 2)))
