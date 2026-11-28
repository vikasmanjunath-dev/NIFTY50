# Operations

## Install
`pip install -r requirements.txt` (research studies also need `requirements-optional.txt`).

## Start the app
Double-click `Nifty Analyser.command`, or `python3 scripts/serve.py` and open http://127.0.0.1:8310/.
Add `--evidence --explain` for the Ask page (needs Ollama).

Shortcuts: `/` focuses the search box; Cmd-K or Ctrl-K jumps to a stock; a focused chart steps with the arrow keys.

## Daily (after 15:30 IST, ideally evening)
`python3 scripts/daily_refresh.py`, or the button on the System page.

## First-time data setup
| Order | Command | Gets |
|---|---|---|
| 1 | `python3 scripts/backfill_bars.py` | prices since 2021 |
| 2 | `python3 scripts/backfill_history.py` | 2008-2020 training history |
| 3 | `python3 scripts/backfill_announcements.py` | company filings |
| 4 | `python3 scripts/backfill_board_meetings.py` | meeting dates |
| 5 | `python3 scripts/build_serving_artifacts.py` | direction model and serving features |
| 6 | `python3 scripts/run_vol_forecast_v2.py`, `run_vol_history.py`, `run_vol_ranges.py` | backtests, ranges |
| 7 | `python3 scripts/forecast_volatility.py` | live forecasts |

Macro series are refreshed by the daily job. Optional research data: `backfill_global.py`, `backfill_intraday.py`, `backfill_options.py`.

The System page lists each data source with its newest date. The daily job also adds new company filings (`refresh_announcements.py`) and embeds them if Ollama is running; restart the server for the Ask page to see them. `backfill_announcements.py` rebuilds the whole set.

`python3 scripts/forward_report.py` summarises the forward record and its scores.

The daily publish (`scripts/daily_publish.sh`) refuses to run while there are uncommitted changes or failing tests; commit first.

## Kill switch
`POST /v3/admin/kill` (server started with `STM_ADMIN_TOKEN`). Forecasts are withheld until reset.

## When something looks wrong
| Symptom | Meaning | Action |
|---|---|---|
| "Out of date" notice, AMBER | data older than the limit | refresh |
| Fewer stocks or an older date than expected | the price vendor is late | refresh later |
| Audit log "BROKEN" | a log line was changed | stop and inspect `data/processed/ops/audit.jsonl` |
| Ask page says it is off | server started without the flags | restart with `--evidence --explain` |

## Backups
System page → *Download a backup of these records* (a zip with each chain's status), then keep it off this machine; they cannot be recreated.

## Tests and docs
`python3 -m pytest -q`; `python3 scripts/smoke_web.py` checks every app route against your real data. After adding or renaming code run `python3 scripts/build_docs.py`.
