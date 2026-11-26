# Survivorship-bias comparison (biased = today's 50 projected backward vs. PIT = real membership)

Generated: 2026-11-23T18:52:11.590843+00:00

## Measured bias

- Phantom rows (stock before it joined the index, present only in the biased universe): 5756 rows, 9 names, annualised 1-day-forward return **+32.4%**
- Omitted rows (real members that later left, absent from the biased universe): 5785 rows, 9 names, annualised return **+3.4%**
- Equal-weight biased: +11.8% (Sharpe 0.82); equal-weight PIT: +9.2% (Sharpe 0.65)
- **Survivorship premium: +2.6%/yr** (t = 3.13)

## Model metrics (mean across 4 purged/embargoed walk-forward folds, identical date blocks)

|                                   |   accuracy |   roc_auc |   information_coefficient |   cross_sectional_ic |
|:----------------------------------|-----------:|----------:|--------------------------:|---------------------:|
| ('biased', 'lightgbm')            |     0.5036 |    0.5075 |                    0.0128 |               0.0186 |
| ('biased', 'logistic_elasticnet') |     0.5109 |    0.5071 |                    0.0107 |               0.0037 |
| ('pit', 'lightgbm')               |     0.5029 |    0.5099 |                    0.0169 |               0.0217 |
| ('pit', 'logistic_elasticnet')    |     0.5075 |    0.5092 |                    0.0178 |               0.0144 |

Reading guide: IC = pooled Spearman of predicted P(up) vs realised 5d return; cross_sectional_ic = mean per-day rank IC.