import numpy as np, pytest
from nifty_analyser.eval.stat_rigor import bootstrap_sharpe_ci, deflated_sharpe, effective_n, expected_max_sharpe


def test_effective_n_shrinks_for_autocorrelated_returns():
    rng = np.random.default_rng(0); e = rng.normal(size=3000)
    ar = np.zeros(3000)
    for i in range(1, 3000): ar[i] = 0.8 * ar[i - 1] + e[i]
    assert effective_n(e) > 0.7 * 3000 and effective_n(ar) < 0.35 * 3000


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 0.01) == 0 < expected_max_sharpe(10, 0.01) < expected_max_sharpe(1000, 0.01)


def test_dsr_penalises_more_trials_and_separates_skill_from_noise():
    rng = np.random.default_rng(1); noise = rng.normal(0, 0.01, 1500); skill = rng.normal(0.0012, 0.01, 1500)
    assert deflated_sharpe(noise, 100)["dsr"] < 0.5
    assert deflated_sharpe(skill, 100)["dsr"] > 0.95
    assert deflated_sharpe(skill, 5)["dsr"] >= deflated_sharpe(skill, 5000)["dsr"]


def test_bootstrap_ci_brackets_point_estimate_and_zero_for_noise():
    rng = np.random.default_rng(2); r = rng.normal(0.0005, 0.01, 800)
    lo, hi, p0 = bootstrap_sharpe_ci(r, n_boot=300)
    assert lo < np.mean(r) / np.std(r, ddof=1) * np.sqrt(252) < hi and 0 <= p0 <= 1
