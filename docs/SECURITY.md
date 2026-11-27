# Security and privacy

## Intended use
One person, on their own computer. The server listens on `127.0.0.1` and **has no login**. Do not expose it to a network or the internet.

## Protections in place
| Risk | Protection |
|---|---|
| Another website reaching the local server by DNS rebinding | foreign `Host` names are rejected |
| Another website starting a refresh | requires a custom header and matching origin |
| Hostile text inside company filings | inserted with `textContent` only; non-HTTPS links dropped; language-model answers pass guardrails |
| Spreadsheet formula injection from CSV export | text cells beginning with `= + - @` are neutralised |
| Injected or foreign scripts; framing | Content-Security-Policy, no-sniff, no-referrer on the web app |
| Edited history | audit log and forecast ledgers are hash-chained |
| A hostile restore file | only well-formed ids and positive amounts are kept |

## Known limits
- Anyone with access to this computer's user account can read and change everything.
- A hash chain shows edits, but someone could rewrite the whole file; keep off-machine backups.
- The admin kill switch needs `STM_ADMIN_TOKEN`; keep the token out of the repository.

## Data
Watchlist and holdings stay in the browser; holdings are sent only to the local server to compute risk.
Downloads come from Yahoo Finance, NSE and Hugging Face; none sends personal information. The repository is public, so never commit keys or personal data.
