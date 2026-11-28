# Current plan
Milestone: **hardening** (feature list and history: [docs/PLAN.md](docs/PLAN.md); item log: [TRACKER.md](TRACKER.md); defects: [BUGS.md](BUGS.md); actions: [LOG.md](LOG.md)).

## Next 5, in order
1. H-09 Remove duplicated test-bar builders; tidy long one-line markup in `web/app.js`.
2. H-10 Split `web/app.js` into modules (one per page); keep the hosted export's text edits working.
3. H-11 Performance pass: time every route on real data, cache anything over 200 ms.
4. H-12 Dependency check: install `requirements.txt` into a fresh environment (BLOCKED if downloads are not approved).
5. H-13 Dark-theme contrast audit.

Approach for H-09: move the shared bar builder into `tests/conftest.py`; no behaviour change, suite must stay green.

## Open risks
- The public snapshot republishes Yahoo and NSE data; terms not checked (owner decision).
- Forward ledgers exist only on this computer.
- A daily task deploys the working folder, so uncommitted work can reach the public site.
