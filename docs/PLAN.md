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

| 12 | New company filings in the daily refresh | only unseen filings appended; search index updated | done |

| 13 | Recent filings for watched stocks on Today | newest filings across the watchlist, linked | done |

| 14 | Page titles, skip link, focus on navigation | keyboard and screen-reader users land on the new page | done |

| 15 | Paste a list of holdings into Portfolio | readable lines added, others reported | done |

| 16 | Jump to any stock with Cmd-K / Ctrl-K | dialog opens, Enter goes to the stock | done |
| 17 | Price chart range: 1M, 6M, 1Y | chart redraws for the chosen span | done |
| 18 | Reload data when returning to a tab left open | page redraws after 10 minutes away | done |

| 19 | How much a stock moved on past results days | compares those days with an ordinary day | done |
| 20 | Stock's chance beside its sector average | shown on the stock page | done |

| 21 | Printable one-page risk summary | watchlist and holdings on one page, prints cleanly | done |
| 22 | Keep pages fast | forward grading and filings do not reread files on every request | done |

| 23 | Save and restore watchlist and holdings as a file | browser data can be moved or recovered | done |
| 24 | Choose the watchlist alert level | 60, 70 or 80% | done |

| 25 | Smoke check of every route on real data | one command, fails loudly | done |
| 26 | Security and privacy notes | protections and limits written down | done |

| 27 | Remember board horizon, sort and watchlist view | choice survives a reload | done |

| 28 | Complete dependency lists | every import is listed; a test enforces it | done |

| 29 | Launcher uses the project environment and explains missing packages | clear message instead of a crash | done |

| 30 | Market mood: India VIX level, how high it is for the past year, history | shown on Today | done |
| 31 | How unusual a stock's recent volatility is | percentile of its last 20 days within 5 years, with chart | done |

| 32 | Charts usable from the keyboard | arrow keys move the readout | done |

| 33 | Portfolio: which holdings move together, and risk if one is removed | listed under the risk breakdown | done |
| 34 | Saved forecasts show whether the price stayed inside its range | graded rows say inside or outside | done |

| 35 | Loading message and retry on failed pages | slow or failed pages say so | done |

| 36 | Sector page: its stocks, average chance, how closely they move together | linked from each stock page | done |

| 37 | Security headers on the web app | only own scripts run; no framing | done |

| 38 | Export portfolio risk as CSV | one click on the Portfolio page | done |

| 39 | Recent audit activity on System | last ten entries, no request contents | done |

| 40 | Refresh reminder on Today when data is 3+ days old | notice with a link | done |

| 41 | Show more filings, up to 100 | button under the list | done |

| 42 | Command-line report of the forward record | matured and pending stated plainly | done |

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
- Hosting the live server publicly: it has no login. Only the read-only snapshot is public.
- More model searching on the same history: it would mostly fit noise.

## Working rules
No look-ahead; adoption rules written before experiments; failures recorded; no secrets or raw market data in git.
