# Phase 2 enriched pipeline -- locked metrics

Generated: 2026-11-24T08:55:28.769154+00:00

## Layers

- price_technical_index_relative: included
- macro_and_regime: included
- delivery: included
- fundamentals: included
- fii_dii: included

## PIT universe (63083 rows, 59 names, 101 features)

| metric | mean | std |
|---|---|---|
| accuracy | 0.5420 | 0.0267 |
| balanced_accuracy | 0.5072 | 0.0169 |
| log_loss | 0.6912 | 0.0036 |
| brier_score | 0.2490 | 0.0018 |
| roc_auc | 0.5198 | 0.0369 |
| information_coefficient | 0.0475 | 0.0816 |

Cost-stressed backtest (pooled out-of-sample, linear sizing, equal weight):

| stress | cost+slip bps | Sharpe | total return | max DD | turnover |
|---|---|---|---|---|---|
| 1.0x | 20 | 0.512 | +4.3% | -4.1% | 0.029 |
| 2.0x | 40 | -0.199 | -1.7% | -5.4% | 0.029 |
| 5.0x | 100 | -2.298 | -17.9% | -18.6% | 0.029 |

## BIASED universe (63054 rows, 50 names, 101 features)

| metric | mean | std |
|---|---|---|
| accuracy | 0.5393 | 0.0393 |
| balanced_accuracy | 0.5013 | 0.0025 |
| log_loss | 0.6936 | 0.0059 |
| brier_score | 0.2501 | 0.0029 |
| roc_auc | 0.5116 | 0.0339 |
| information_coefficient | 0.0309 | 0.0736 |

Cost-stressed backtest (pooled out-of-sample, linear sizing, equal weight):

| stress | cost+slip bps | Sharpe | total return | max DD | turnover |
|---|---|---|---|---|---|
| 1.0x | 20 | 1.061 | +10.5% | -3.8% | 0.021 |
| 2.0x | 40 | 0.621 | +6.0% | -4.5% | 0.021 |
| 5.0x | 100 | -0.686 | -6.5% | -12.7% | 0.021 |

## Reading guide

PIT = real point-in-time Nifty 50 membership (headline); biased = today's 50 projected backward. Accuracy/AUC/IC near 0.5/0.5/0 mean no detectable edge.