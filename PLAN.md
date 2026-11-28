# Current plan
Milestone: **hardening** (feature list and history: [docs/PLAN.md](docs/PLAN.md); item log: [TRACKER.md](TRACKER.md); defects: [BUGS.md](BUGS.md); actions: [LOG.md](LOG.md)).

## Next 5, in order
1. H-06 Audit-log and ledger reads tolerate a half-written last line.
2. H-07 Concurrent refresh guard across processes (daily publish task versus the refresh button).
3. H-08 Fuzz the portfolio, ask and restore inputs in the hosted export's browser shim.
4. H-09 Remove duplicated test-bar builders and long one-line markup in `web/app.js`.
5. H-10 Split `web/app.js` into modules (one per page).

Approach for H-06: skip an unparsable final line in `AuditLog.read`, still report it as a broken chain in `verify`; test with a truncated file.

## Open risks
- The public snapshot republishes Yahoo and NSE data; terms not checked (owner decision).
- Forward ledgers exist only on this computer.
- A daily task deploys the working folder, so uncommitted work can reach the public site.
