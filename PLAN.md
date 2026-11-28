# Current plan
Milestone: **hardening** (feature list and history: [docs/PLAN.md](docs/PLAN.md); item log: [TRACKER.md](TRACKER.md); defects: [BUGS.md](BUGS.md); actions: [LOG.md](LOG.md)).

## Next 5, in order
1. H-17 If a split or bonus restates old prices, confirm the shrink guard and ledger grading still behave.
2. H-09 Tidy long one-line markup in `web/app.js`.
3. H-10 Split `web/app.js` into modules.
4. H-12 Fresh-environment install check (BLOCKED: needs a large download, owner's yes).
5. Regression and security pass.

## Open risks
- The public snapshot republishes Yahoo and NSE data; terms not checked (owner decision).
- Forward ledgers exist only on this computer.
- A daily task deploys the working folder, so uncommitted work can reach the public site.
