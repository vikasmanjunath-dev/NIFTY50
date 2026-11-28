# Goals (why, and what counts as success)
Owns vision, metrics, scope. Design: [blueprint.md](blueprint.md). Order: [PLAN.md](PLAN.md). State: [TRACKER.md](TRACKER.md#execution-tracker).

**Vision:** tell one investor, honestly, how big the next moves in Nifty 50 stocks are likely to be, and prove it on data saved before the outcome.

**Problem:** direction forecasts are not predictable here `[F]` (AUC 0.50). Size of moves is `[F]` (back-test AUC 0.691 / 0.750), but that history was searched many times, so it is not yet proof `[R]` R-01.

## Personas
| ID | Who | Job to be done | Pain |
|---|---|---|---|
| P1 | Owner-investor (primary) | each evening: which holdings may swing, and by how much | raw data, false confidence |
| P2 | Visitor to the public snapshot `[A]` A-01 | browse a read-only view | stale or misleading figures |

## North Star
**Forward AUC of the 5-day forecast** on graded ledger entries. Inputs: data dates recorded; refresh success rate; price-range coverage.

## OKRs
| ID | Objective / key result | Baseline → target | By | Measured by |
|---|---|---|---|---|
| O1 | Prove the forecast forward | | | |
| KR1.1 | data dates recorded in the ledger | 3 → 60 | 2027-03-31 | `scripts/forward_report.py` |
| KR1.2 | forward 5-day AUC | none graded → ≥0.65 | 2027-03-31 | same |
| KR1.3 | forward 90% range coverage | none graded → 85–93% | 2027-03-31 | `/api/record` |
| O2 | Keep it running unattended | | | |
| KR2.1 | trading days with a successful refresh | unmeasured → ≥95% | 2027-01-31 | run log (T-005) |
| KR2.2 | off-machine ledger backups | 0 → weekly, 1 restore tested | 2026-12-12 | restore check (T-003) |
| O3 | Publish only what may be published | | | |
| KR3.1 | data-licence position decided and acted on | decided 2026-11-28 (D-005) | 2026-12-05 | ADR D-005 |
| KR3.2 | open known defects of priority P0/P1 | 0 → 0 | always | [BUGS.md](BUGS.md) |

## Guardrails (must not get worse)
No direction signal shown as usable · forecasts withheld when data > 5 days old · all tests pass before any push · no secrets or holdings in the repo.

## Scope
| MoSCoW | Items |
|---|---|
| Must | daily refresh, forward ledger and grading, backup, licence decision |
| Should | CI on push, fresh-install check, scorecard with uncertainty |
| Could | split `web/app.js`, retest option features after 12 more months |
| Won't | buy/sell signals, order execution, multi-user accounts, public hosting of the live server, more model search on the same history |

## Success and kill criteria
| ID | Rule |
|---|---|
| S-1 | success: KR1.2 and KR1.3 met on 2027-03-31 |
| K-1 | forward 5-day AUC < 0.55 with ≥60 data dates (quote its interval) → stop presenting forecasts as useful; show history only |
| K-2 | KR1.1 below 30 dates on 2027-01-31 → fix automation before any feature work |
| K-3 | source terms forbid republication → take the public snapshot down within 7 days |
