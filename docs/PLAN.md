# Project plan

The working plan. Item-level history is in [TRACKER.md](../TRACKER.md); model experiments are in [AUC_PLAN.md](AUC_PLAN.md).

## Where things stand (2026-11-27)
| Area | State |
|---|---|
| Direction forecasting | No demonstrated edge (AUC about 0.50). Closed. |
| Volatility forecasting | Live. AUC 0.691 (5-day), 0.750 (20-day) on 2021+ out-of-sample rows. |
| Filing intelligence | Retrieval, guarded answers, event classifier. |
| Web app | Today, stock, compare, portfolio, filings, ask, model, system, guide. |
| Forward record | Started; no forecast has matured yet. |

## Now
| # | Item | Done when | State |
|---|---|---|---|
| 1 | Docs for every folder, API and operations guide, kept current by a test | `tests/test_docs.py` passes | done |
| 2 | Watchlist highlights on Today | watched stocks with high chance or an announced meeting are called out | done |
| 3 | Filter filings by category | Filings page has a category menu | done |

| 9 | Per-stock forward record on stock pages | each saved forecast shows pending or its graded outcome | done |
| 10 | One-click download of the forward records | a zip of the ledgers with their chain status | done |

| 11 | Data freshness per source on the System page | each input shows its newest date and whether it lags | done |

## Next (in order)
| # | Item | Why | Needs |
|---|---|---|---|
| 4 | Enough graded forecasts to judge the live model | only clean evidence | weeks of daily refreshes |
| 5 | Scheduled daily refresh (launchd) | keeps the record growing unattended | owner installs `scripts/launchd` job |
| 6 | Back up the forward ledgers off this machine | they cannot be recreated | owner chooses a destination |
| 7 | Retest option-implied features | little option history today | about a year more data |
| 8 | Human relevance labels | LLM judges are a proxy | owner runs `scripts/label_cli.py` |

## Not planned
- Buy/sell signals: nothing tested predicts direction.
- Public hosting: the app has no login; it stays on localhost.
- More model searching on the same history: it would mostly fit noise.

## Working rules
No look-ahead; adoption rules written before experiments; failures recorded; no secrets or raw market data in git.
