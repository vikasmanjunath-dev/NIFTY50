# Volatility AUC: plan and experiment tracker

Goal: highest **honest** AUC for "will the next h-day volatility be above usual". Direction is tracked but has shown no edge.

## Rules
1. No future information; targets and rows stay fixed so numbers are comparable.
2. Write the candidates and the adoption rule before running.
3. Select on data not used to judge: from round 4, selection uses 2008-2020 only and 2021+ is the single confirmation.
4. Adopt only if the 95% date-block interval of the gain is above zero.
5. Log every trial, including failures. The forward ledger is the only clean test.

## Results so far (all 2021+ out-of-sample AUC)
| Round | Change | 5-day | 20-day | Outcome |
|---|---|---|---|---|
| 0 | persistence rule | 0.598 | 0.669 | benchmark |
| 1 | HAR + LightGBM blend, 21 features | 0.671 | 0.702 | adopted |
| 2a | delivery %, announcement counts, FII/FX/Brent | no gain or worse | same | rejected |
| 2b | AUC search 1: refits, extra features, Chronos, ensemble | +0.004 holdout | +0.007 holdout | rejected |
| 3a | 2008-2020 training history | 0.684 | 0.739 | adopted |
| 3b | AUC search 2: extra features with history | not adopted | 0.752 | 20-day adopted |

Trials against 2021+ data so far: about 15 model variants per horizon, so small gains are suspect.

## Round 4 (done, nothing adopted)
Select on 2008-2020 walk-forward (8 folds) among 3 feature sets x 5 models; confirm one winner per horizon on 2021+ against the live model.
- Features: `base`, `extra`, `extra+stock` (own past exceedance rate, volatility versus sector and market).
- Models: `blend`, `big_trees`, `cat3` (adds CatBoost), `reg_target` (learn the size of the change, rank by it), `combo` (classifier + reg_target).

Results (`reports/vol_auc_search3.json`):

| | 2008-2020 winner | 2021+ AUC live → winner | 95% interval of gain | Outcome |
|---|---|---|---|---|
| 5-day | `extra_stock/blend` | 0.6843 → 0.6890 | -0.0003 to +0.0095 | rejected (narrowly) |
| 20-day | `extra_stock/combo` | 0.7502 → 0.6385 | -0.146 to -0.074 | rejected |

Lesson: `combo` was a flawed design. It ranked scores inside each validation block, which uses other rows from that block (not possible live) and throws away
differences in level between periods. Do not reuse it. Runners-up were not tried on 2021+ afterwards, which would be picking after seeing the answer.
Learning the size of the change (`reg_target`) is clearly worse than the direct classifier at 20 days. Model changes are now worth about 0.005 at most: the data is the limit.

## Round 5 (done, nothing adopted): overnight US and index features
Downloaded (approved) `^GSPC`, `^VIX`, `^NSEI` from 2008. Candidates: live features, +US, +index, +both; each with `blend` and `big_trees`.

| `reports/vol_auc_search4.json` | 2008-2020 winner | 2021+ AUC live → winner | 95% interval of gain | Outcome |
|---|---|---|---|---|
| 5-day | live features (nothing new helped) | 0.6843 → 0.6843 | none | no change |
| 20-day | live features, larger trees | 0.7502 → 0.7513 | -0.0063 to +0.0087 | rejected |

Every US or index feature set scored **below** the live features on 2008-2020 (except a tie at 5 days).

## Current best and status
**5-day 0.684, 20-day 0.750** (2021+ out-of-sample). Rounds 4 and 5 added nothing, including genuinely new data. Stop searching on daily data.

## Round 6 (done, nothing adopted): intraday realised volatility
Source: Hugging Face `xxparthparekhxx/indian-stock-market-minute-data` (MIT, from Upstox; approved, 2 GB cap). 1.3 GB streamed, 4.8 MB of daily summaries kept.
59 symbols 2022-01 to 2026-11-13 (TATAMOTORS absent); daily returns correlate 0.98 with ours, 0.2% of days differ by more than 2%.

| `reports/vol_intraday.json`, scored from 2023 | first-half pick | second half live → pick | 95% interval | Outcome |
|---|---|---|---|---|
| 5-day | intraday, trained 2022+ only | 0.700 → 0.701 | -0.007 to +0.010 | rejected |
| 20-day | live (intraday was worse) | 0.767 | none | no change |

The dataset refreshes about monthly, so it could not feed a daily live forecast even if it had helped.
Standing result unchanged: **5-day 0.684, 20-day 0.750**.

## Backlog (in order)
| Idea | Needs |
|---|---|
| Score the forward ledger | months of new data (`scripts/daily_refresh.py`) |
