# Robustness suite — real point-in-time data (2026-11-24)

Data: 71,697 labelled rows, 59 instruments, 2021-01-01..2026-11-13, 101 scale-free features (the serving model's set).
Model: LightGBM (speed; the calibrated ensemble was evaluated in `run_phase2_pipeline.py`). Script: `scripts/run_robustness.py`; raw numbers: `robustness.json`.

## 1. Cost realism (4-fold purged walk-forward OOS predictions, 1-day holding of a 2p−1 position, equal-weight across names)
| Strategy | Sharpe | Ann. return | Max DD |
|---|---|---|---|
| Strategy, gross (no cost) | 0.68 | +2.9% | −7.2% |
| Strategy, blended 20 bps | −0.27 | −1.1% | −11.6% |
| Strategy, impact model Rs 1 cr / 10 cr / 100 cr / 1000 cr | −0.21 / −0.24 / −0.33 / −0.62 | −0.9% / −1.0% / −1.4% / −2.6% | −11% … −17% |
| Equal-weight buy & hold, no cost | 0.82 | +11.2% | −22.3% |
| Equal-weight buy & hold, impact model Rs 10 cr | 0.77 | +10.4% | −23.9% |

Impact model: 10 bps fixed + 5 bps half-spread + 0.5·σ·√participation (ADV from trailing 20 days, lagged; 100% ADV coverage). Cost per day is only 1.5–2.2 bps of NAV because the position is small and turnover moderate, **yet the strategy still turns negative net of costs at every capital level and never approaches buy-and-hold.** The result is insensitive to the impact model's capital assumption; fixed cost + spread dominate until ~Rs 1000 cr.

## 2. Combinatorial purged CV (N=8 groups, k=2 → 28 splits, 7 assembled out-of-sample paths)
Pooled up-day base rate 0.534.
| Metric | mean | std | min | max |
|---|---|---|---|---|
| ROC-AUC | **0.494** | 0.006 | 0.488 | 0.502 |
| Information coefficient | −0.009 | 0.013 | −0.024 | +0.010 |
| Log loss | 0.722 | 0.004 | 0.716 | 0.727 |
**No path has AUC meaningfully above 0.5; the mean is marginally below.** The earlier 4-fold walk-forward numbers (AUC ≈ 0.51–0.52) sit above the whole CPCV distribution, i.e. they were a favourable draw, not evidence of edge. (CPCV trains on data both before and after each test group, so it measures model quality across regimes, not a deployable chronology; AUC is pooled across dates, so shifts in the up-day base rate between periods add noise.)

## 3. Learned (GMM) market regime vs rule-based regime — paired, 6 purged walk-forward folds × 3 seeds
| Variant | AUC | IC | Log loss |
|---|---|---|---|
| Rule-based regime codes (current) | 0.5111 | +0.0256 | 0.7209 |
| No regime features | 0.5091 | +0.0224 | 0.7202 |
| Learned GMM posteriors (fit on training dates only) | 0.5073 | +0.0180 | 0.7210 |
Learned − rule-based AUC: mean −0.0038 (SE 0.0036; better in 3/6 folds). Learned − none: −0.0018 (SE 0.0026; 2/6). **The learned regime gives no measurable benefit** (all differences within about one standard error); it is not promoted. The rule-based codes also add nothing detectable over no regime features.

## Conclusion
Three independent robustness checks agree with the earlier results: no detectable predictive edge, strategy net returns below equal-weight buy-and-hold at every cost assumption, and added model complexity (learned regime) does not help.
