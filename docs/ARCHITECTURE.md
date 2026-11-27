# Architecture

```
sources ──> data/ (ingest, point-in-time stamps) ──> features/ ──> models/ ──> ops/ ──> web app + /v3 API
   NSE filings ──> documents/ (retrieval, guarded answers, classifier) ───────────┘
                 validation/ (purged walk-forward)  eval/ (metrics, costs)  monitoring/ (drift)
```

## Contracts
- Every record has `event_time` and `available_time`; nothing is used before it was knowable (`data/store.py`).
- Ingestion never stores the unfinished bar of an open session.
- Live forecasts need data for at least 95% of that date's index members.

## Volatility pipeline (live)
1. `features/volatility*.py`: range-based HAR terms, extra features (20-day), announced board meetings (5-day).
2. `models/volatility.py`: blend of a linear model and shallow gradient boosting; `models/ranges.py`: price-move ranges.
3. `ops/vol_forecast.py`: trains on realised outcomes only (2008 onwards), forecasts, appends to a hash-chained ledger, scores it later.

## Direction pipeline (kept, no edge)
Ensemble served by `/v3/analyse`, which mostly abstains.

## Web app
`web/` is plain HTML, CSS and JavaScript with no build step; `ops/web.py` serves read-only JSON plus a protected refresh.
Server text is inserted with `textContent` only; foreign `Host` names are rejected.

## Files not in git
`data/raw/` (downloads), `data/processed/` (models, features, ledgers).
